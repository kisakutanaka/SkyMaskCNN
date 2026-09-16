/**
 * sky-mask.js
 *
 * 空マスクの計算そのもの。DOM も Worker も ONNX も知らない純粋関数だけ。
 * 呼ぶ順番は sky-segmenter.worker.js が、可視化は steps.html が持つ。
 * （計算と配管を分けてあるので、説明用のページが同じ関数を呼び直せる）
 *
 * 出力先の配列は呼び出し側が渡す。毎フレーム確保するとモバイルで GC が
 * 時々 1 フレーム分の時間を丸ごと奪うため。
 */

const MEAN = [0.485, 0.456, 0.406]; // ImageNet 正規化（学習時と同じ前処理）
const STD = [0.229, 0.224, 0.225];

/**
 * 設定から実際に使う解像度を決める。呼び出し側の値は丸めて受ける。
 * refineSize をそのまま使うと inputSize の整数倍でない値（384 など）で
 * 縮小ループが取り込み範囲の外を読み、NaN が入ったまま動き続ける。
 * coeffSize は 2 の冪に丸めて inputSize を割り切らせる。
 * steps.html も同じ関数を使うので、説明と実物がずれない。
 */
export function plan(cfg) {
  const size = cfg.inputSize;
  const pool = Math.max(1, Math.round(cfg.refineSize / size));
  const shrink = 2 ** Math.max(0, Math.round(Math.log2(size / Math.max(1, cfg.coeffSize))));
  const coeffSize = Math.max(1, size / shrink);
  return {
    size,
    capture: size * pool,
    coeffSize,
    // 半径は画像上の大きさを保つ（係数解像度に合わせて縮める）
    coeffRadius: Math.max(1, Math.round(cfg.refineRadius / shrink)),
    shrink,
  };
}

/**
 * 取り込んだ RGBA を、モデル入力（NCHW・正規化済み）と輝度ガイドにする。
 * pool×pool の面積平均で縮小しながら両方を同時に作る。
 * @param {Uint8ClampedArray} rgba capture×capture の RGBA
 * @param {number} capture 取り込み解像度
 * @param {number} size モデル入力の一辺（capture = size * pool）
 * @param {Float32Array} input 出力: [1,3,size,size]
 * @param {Float32Array} guide 出力: size×size の輝度（0..1）
 */
export function toInput(rgba, capture, size, input, guide) {
  const pool = capture / size;
  const plane = size * size;
  const inv = 1 / (pool * pool);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let dy = 0; dy < pool; dy++) {
        let p = ((y * pool + dy) * capture + x * pool) * 4;
        for (let dx = 0; dx < pool; dx++, p += 4) {
          r += rgba[p];
          g += rgba[p + 1];
          b += rgba[p + 2];
        }
      }
      r *= inv;
      g *= inv;
      b *= inv;
      const i = y * size + x;
      input[i] = (r / 255 - MEAN[0]) / STD[0];
      input[i + plane] = (g / 255 - MEAN[1]) / STD[1];
      input[i + plane * 2] = (b / 255 - MEAN[2]) / STD[2];
      guide[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    }
  }
}

/**
 * ロジットを「空である確率」にする。
 * 多クラスのときは「空」対「空以外の最大」の 2 値ソフトマックス
 * = sigmoid(sky - maxOther - margin)。150 クラス全部の softmax より安く、
 * 境界がなめらかな確率になる。
 */
export function toProbability(values, numClasses, area, skyClassIndex, margin, dst) {
  if (numClasses === 1) {
    // 空/非空の二値モデル: ロジットをそのまま sigmoid するだけ
    for (let i = 0; i < area; i++) dst[i] = 1 / (1 + Math.exp(-values[i]));
    return dst;
  }
  for (let i = 0; i < area; i++) {
    const sky = values[skyClassIndex * area + i];
    let other = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      if (c === skyClassIndex) continue;
      const v = values[c * area + i];
      if (v > other) other = v;
    }
    dst[i] = 1 / (1 + Math.exp(other - sky + margin));
  }
  return dst;
}

/** 輝度ガイドを shrink×shrink の箱平均で縮める（shrink=1 ならそのまま返す） */
export function shrinkGuide(src, size, shrink, dst) {
  if (shrink === 1) return src;
  const out = size / shrink;
  const inv = 1 / (shrink * shrink);
  for (let y = 0; y < out; y++) {
    for (let x = 0; x < out; x++) {
      let sum = 0;
      for (let dy = 0; dy < shrink; dy++) {
        const row = (y * shrink + dy) * size + x * shrink;
        for (let dx = 0; dx < shrink; dx++) sum += src[row + dx];
      }
      dst[y * out + x] = sum * inv;
    }
  }
  return dst;
}

/**
 * ガイデッドフィルタ (He et al., 2010) の線形係数 a, b を求める。
 * 出力は「マスク ≒ a * ガイド輝度 + b」の形でガイドの輪郭に沿う。
 * a, b は元画像より滑らかなので、低解像度で求めて拡大しても品質がほとんど
 * 落ちない（= fast guided filter）。
 */
export function guidedCoeffs(I, p, size, r, eps, buf = (key, len) => new Float32Array(len)) {
  const n = size * size;
  const tmp = buf('box.tmp', n); // boxFilter の作業用。呼ぶたびに丸ごと上書きされる
  const Ip = buf('gc.Ip', n);
  const II = buf('gc.II', n);
  for (let i = 0; i < n; i++) {
    Ip[i] = I[i] * p[i];
    II[i] = I[i] * I[i];
  }
  const meanI = boxFilter(I, size, r, buf('gc.meanI', n), tmp);
  const meanP = boxFilter(p, size, r, buf('gc.meanP', n), tmp);
  const meanIp = boxFilter(Ip, size, r, buf('gc.meanIp', n), tmp);
  const meanII = boxFilter(II, size, r, buf('gc.meanII', n), tmp);

  const a = buf('gc.a', n);
  const b = buf('gc.b', n);
  for (let i = 0; i < n; i++) {
    const cov = meanIp[i] - meanI[i] * meanP[i];
    const varI = meanII[i] - meanI[i] * meanI[i];
    a[i] = cov / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  return {
    a: boxFilter(a, size, r, buf('gc.meanA', n), tmp),
    b: boxFilter(b, size, r, buf('gc.meanB', n), tmp),
  };
}

/** from → to のバイリニア拡大で使う、位置と重みの表（正方形なので x と y で共用） */
export function upscaleTable(from, to) {
  const s = from / to;
  const i0 = new Int32Array(to);
  const i1 = new Int32Array(to);
  const w = new Float64Array(to); // float32 に落とすと bilinear() と丸めがずれる
  for (let i = 0; i < to; i++) {
    const f = Math.min((i + 0.5) * s - 0.5, from - 1);
    i0[i] = Math.max(0, Math.floor(f));
    i1[i] = Math.min(i0[i] + 1, from - 1);
    w[i] = f - i0[i];
  }
  return { i0, i1, w };
}

/**
 * 係数の拡大・輝度ガイドの計算・合成を 1 パスで解く。
 * 素直に書くと aHi / bHi / guideHi という capture×capture の配列を 3 本
 * 書いて読み直すことになり、実機（熱ダレ後）ではここが 15ms かかっていた。
 * 式は同じで、途中の配列を作らないだけ。
 * @param {number} k 確率の 0→1 遷移をどれだけ立てるか（edgeSharpness）
 */
export function refine(a, b, rgba, coeffSize, capture, k, table, dst) {
  const { i0, i1, w } = table;
  for (let y = 0, i = 0; y < capture; y++) {
    const y0 = i0[y] * coeffSize;
    const y1 = i1[y] * coeffSize;
    const wy = w[y];
    for (let x = 0; x < capture; x++, i++) {
      const x0 = i0[x];
      const x1 = i1[x];
      const wx = w[x];
      const a0 = a[y0 + x0] * (1 - wx) + a[y0 + x1] * wx;
      const a1 = a[y1 + x0] * (1 - wx) + a[y1 + x1] * wx;
      const b0 = b[y0 + x0] * (1 - wx) + b[y0 + x1] * wx;
      const b1 = b[y1 + x0] * (1 - wx) + b[y1 + x1] * wx;
      // 輝度ガイド（ここでしか使わないので配列にしない）
      const p = i * 4;
      const g = (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) / 255;
      const v = ((a0 * (1 - wy) + a1 * wy) * g + (b0 * (1 - wy) + b1 * wy) - 0.5) * k + 0.5;
      dst[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }
  return dst;
}

/** 0.5 を境に確率のコントラストを立てる（遷移帯の幅を 1/k にする） */
export function sharpen(src, k, dst = new Float32Array(src.length)) {
  if (!k || k === 1) { dst.set(src); return dst; }
  for (let i = 0; i < src.length; i++) {
    const v = (src[i] - 0.5) * k + 0.5;
    dst[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return dst;
}

/** 正方形マスクのバイリニア拡大。dst を渡せばそこへ書く。 */
export function bilinear(src, w, h, size, dst = new Float32Array(size * size)) {
  const sx = w / size;
  const sy = h / size;
  for (let y = 0; y < size; y++) {
    const fy = Math.min((y + 0.5) * sy - 0.5, h - 1);
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(y0 + 1, h - 1);
    const wy = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = Math.min((x + 0.5) * sx - 0.5, w - 1);
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(x0 + 1, w - 1);
      const wx = fx - x0;
      const a = src[y0 * w + x0] * (1 - wx) + src[y0 * w + x1] * wx;
      const b = src[y1 * w + x0] * (1 - wx) + src[y1 * w + x1] * wx;
      dst[y * size + x] = a * (1 - wy) + b * wy;
    }
  }
  return dst;
}

/**
 * 移動平均（累積和による O(N) 実装）
 * dst / tmp を渡せばそこへ書く。横方向は src→tmp、縦方向は tmp→dst なので、
 * dst は src と同じ配列でも構わない（tmp だけは別の配列にすること）。
 */
export function boxFilter(src, size, r, dst = new Float32Array(size * size), tmp = new Float32Array(size * size)) {
  for (let y = 0; y < size; y++) {
    const row = y * size;
    let sum = 0;
    for (let x = 0; x < r && x < size; x++) sum += src[row + x];
    for (let x = 0; x < size; x++) {
      const lo = x - r - 1;
      const hi = x + r;
      if (hi < size) sum += src[row + hi];
      if (lo >= 0) sum -= src[row + lo];
      tmp[row + x] = sum / (Math.min(hi, size - 1) - Math.max(lo + 1, 0) + 1);
    }
  }
  for (let x = 0; x < size; x++) {
    let sum = 0;
    for (let y = 0; y < r && y < size; y++) sum += tmp[y * size + x];
    for (let y = 0; y < size; y++) {
      const lo = y - r - 1;
      const hi = y + r;
      if (hi < size) sum += tmp[hi * size + x];
      if (lo >= 0) sum -= tmp[lo * size + x];
      dst[y * size + x] = sum / (Math.min(hi, size - 1) - Math.max(lo + 1, 0) + 1);
    }
  }
  return dst;
}
