/**
 * sky-segmenter.worker.js
 *
 * 分離の全工程をここで走らせる。メインスレッドに残すのは
 * createImageBitmap（非同期。GPU の完了を待たない）だけ。
 *
 * 実機 (iPhone/Safari) の内訳で、いちばん高いのは推論ではなく
 * getImageData の 18〜19ms だった。これは GPU の描画完了を待つ同期で、
 * 解像度を 1/4 にしても減らない固定費（512:19ms / 256:18ms）。
 * 減らせないなら、せめてメインスレッドの外へ出す、というのがこのファイル。
 *
 * sky-segmenter.js からだけ使われる。プロトコルは 2 種類:
 *   { type:'init',  cfg }               → { type:'ready', inputSize, maskSize }
 *   { type:'frame', bitmap, out }       → { type:'mask', width, height, data, timings }
 * out は前のフレームで返した Float32Array。転送で返してもらって使い回す
 * （毎フレーム確保すると 1 フレーム 7MB 近いゴミになる）。
 */

const MEAN = [0.485, 0.456, 0.406]; // ImageNet 正規化（学習時と同じ前処理）
const STD = [0.229, 0.224, 0.225];

let cfg = null;
let ort = null;
let session = null;
let inputName = '';
let outputName = '';
let size = 0; // モデル入力の一辺
let pool = 1; // 何画素を 1 画素に畳むか
let capture = 0; // 取り込み解像度 = size * pool
let coeffSize = 0; // ガイデッドフィルタの係数を求める解像度
let coeffRadius = 1; // その解像度での半径
let shrink = 1; // size / coeffSize
let ctx = null;
let guideLo = null;
let guideCo = null;

// 作業用バッファ。毎フレーム確保するとモバイルで GC が時々 1 フレーム分の
// 時間を丸ごと奪う。Worker 側に置けるものは全部ここで 1 回だけ取る。
const scratch = new Map();
const buf = (key, len) => {
  let a = scratch.get(key);
  if (!a || a.length !== len) scratch.set(key, (a = new Float32Array(len)));
  return a;
};

// 受け取った順に 1 枚ずつ処理する。onmessage を直接 async にすると、
// session.run を待っている間に次のメッセージが割り込んで、共有している
// 作業用バッファを壊す（呼び出し側が 2 枚以上同時に投げられるため）。
let chain = Promise.resolve();
let lastEnd = 0; // 前の 1 枚を終えた時刻。次が来るまでの空き時間を測る

self.onmessage = (e) => {
  const msg = e.data;
  chain = chain.then(async () => {
    try {
      if (msg.type === 'init') await init(msg.cfg, msg.id);
      else if (msg.type === 'frame') await frame(msg);
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: err?.message ?? String(err) });
    }
  });
};

async function init(options, id) {
  cfg = options;
  // ESM 版の ort をそのまま読む。ここは Worker なので、これ自体の取得も
  // メインスレッドを止めない。wasm の実体は下の wasmPaths から取る。
  ort = await import(cfg.ortUrl);
  ort.env.wasm.wasmPaths = cfg.ortWasmPaths;
  ort.env.wasm.numThreads = 1; // GitHub Pages は COOP/COEP を付けられない
  ort.env.wasm.proxy = false; // すでに Worker の中なので、さらに Worker を作らせない

  session = await ort.InferenceSession.create(cfg.modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];

  size = cfg.inputSize;
  // 畳む画素数を先に決め、取り込み解像度をそこから導く。refineSize をそのまま
  // 使うと inputSize の整数倍でない値（384 など）で縮小ループが範囲外を読む。
  pool = Math.max(1, Math.round(cfg.refineSize / size));
  capture = size * pool;

  // ガイデッドフィルタの係数 a, b を求める解像度。a, b は元画像より滑らかなので、
  // 粗く求めて拡大しても品質がほとんど落ちない（fast guided filter の肝）。
  // 論文の推奨は入力に対して 1/4（= 128）。2 の冪に丸めて size を割り切らせる。
  shrink = 2 ** Math.max(0, Math.round(Math.log2(size / Math.max(1, cfg.coeffSize))));
  coeffSize = Math.max(1, size / shrink);
  // 半径は画像上の大きさを保つ（係数解像度に合わせて縮める）
  coeffRadius = Math.max(1, Math.round(cfg.refineRadius / shrink));

  const canvas = new OffscreenCanvas(capture, capture);
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  guideLo = new Float32Array(size * size); // 統計量計算用に縮小した輝度
  guideCo = shrink === 1 ? guideLo : new Float32Array(coeffSize * coeffSize);
  buildUpscaleTable();

  self.postMessage({ type: 'ready', id, inputSize: size, maskSize: capture, coeffSize });
}

async function frame({ id, bitmap, out }) {
  const timings = {};
  let t = performance.now();
  // 前の 1 枚を終えてから次が届くまでの空き。ここが大きいと、Worker は
  // 仕事を待って遊んでいる＝呼び出し側の往復がボトルネックということ。
  timings['Worker の空き'] = lastEnd ? t - lastEnd : 0;
  const mk = (k) => { timings[k] = performance.now() - t; t = performance.now(); };
  const whole = performance.now();

  // 1. 受け取ったビットマップを取り込む（呼び出し側で capture×capture に
  //    潰してあるので、ここは等倍のコピー）
  ctx.drawImage(bitmap, 0, 0, capture, capture);
  bitmap.close();
  mk('drawImage');
  const { data: rgba } = ctx.getImageData(0, 0, capture, capture);
  mk('getImageData');

  // 2. pool×pool の面積平均でモデル入力サイズへ縮小しつつ、NCHW / 正規化 /
  //    低解像度ガイドを同時に作る。proxy=false なので input も使い回せる
  //    （ort が転送しないため、次のフレームでも生きている）。
  const input = buf('input', 3 * size * size);
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
      guideLo[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    }
  }
  mk('縮小と正規化');

  // 4. 推論。出力は低解像度のロジット
  const outputs = await session.run({
    [inputName]: new ort.Tensor('float32', input, [1, 3, size, size]),
  });
  mk('推論');
  const logits = outputs[outputName];
  const [, numClasses, h, w] = logits.dims;
  const values = logits.data;

  // 5. 「空」と「空以外の最大」の 2 値ソフトマックス = sigmoid(sky - maxOther - margin)
  //    150 クラス全部の softmax より安く、境界がなめらかな確率になる。
  const area = h * w;
  const coarse = buf('coarse', area);
  if (numClasses === 1) {
    // 空/非空の二値モデル: ロジットをそのまま sigmoid するだけ
    for (let i = 0; i < area; i++) coarse[i] = 1 / (1 + Math.exp(-values[i]));
  } else {
    for (let i = 0; i < area; i++) {
      const sky = values[cfg.skyClassIndex * area + i];
      let other = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        if (c === cfg.skyClassIndex) continue;
        const v = values[c * area + i];
        if (v > other) other = v;
      }
      coarse[i] = 1 / (1 + Math.exp(other - sky + cfg.skyMargin));
    }
  }
  mk('sigmoid');

  if (!cfg.refineRadius) {
    const flat = sharpen(coarse, cfg.edgeSharpness, take(out, area));
    timings['合計'] = performance.now() - whole;
    lastEnd = performance.now();
    self.postMessage({ type: 'mask', id, width: w, height: h, data: flat, timings }, [flat.buffer]);
    return;
  }

  // 6. fast guided filter:
  //    線形係数 a, b は低解像度（inputSize）で求め、それを拡大して
  //    高解像度のガイドに当てる。box filter の計算量は据え置きのまま、
  //    輪郭だけ capture の解像度で吸着する（He et al. 2010 の 4.1 節）。
  // 係数解像度の輝度ガイド。shrink×shrink の箱平均で作る（shrink=1 ならそのまま）
  if (shrink !== 1) {
    const inv = 1 / (shrink * shrink);
    for (let y = 0; y < coeffSize; y++) {
      for (let x = 0; x < coeffSize; x++) {
        let sum = 0;
        for (let dy = 0; dy < shrink; dy++) {
          const row = (y * shrink + dy) * size + x * shrink;
          for (let dx = 0; dx < shrink; dx++) sum += guideLo[row + dx];
        }
        guideCo[y * coeffSize + x] = sum * inv;
      }
    }
  }
  // モデル出力がちょうど係数解像度なら、拡大せずそのまま使える（既定はこの経路）
  const pCo = w === coeffSize && h === coeffSize
    ? coarse
    : bilinear(coarse, w, h, coeffSize, buf('pCo', coeffSize * coeffSize));
  const { a, b } = guidedCoeffs(guideCo, pCo, coeffSize, coeffRadius, cfg.refineEps, buf);

  // a, b の拡大・輝度ガイド・合成を 1 パスにまとめる。素直に書くと
  // aHi / bHi / guideHi という 512×512 の配列を 3 本書いて読み直すことになり、
  // 実機（熱ダレ後）ではここが 15ms かかっていた。式は同じで、
  // 途中の配列を作らないだけ（出力はビット単位で一致する）。
  // 拡大の重みは x 方向・y 方向とも位置だけで決まるので、表にして使い回す。
  const mask = take(out, capture * capture);
  const k = cfg.edgeSharpness;
  for (let y = 0, i = 0; y < capture; y++) {
    const y0 = upY0[y] * coeffSize;
    const y1 = upY1[y] * coeffSize;
    const wy = upWy[y];
    for (let x = 0; x < capture; x++, i++) {
      const x0 = upX0[x];
      const x1 = upX1[x];
      const wx = upWx[x];
      // a を拡大
      const a0 = a[y0 + x0] * (1 - wx) + a[y0 + x1] * wx;
      const a1 = a[y1 + x0] * (1 - wx) + a[y1 + x1] * wx;
      // b を拡大
      const b0 = b[y0 + x0] * (1 - wx) + b[y0 + x1] * wx;
      const b1 = b[y1 + x0] * (1 - wx) + b[y1 + x1] * wx;
      // 輝度ガイド（ここでしか使わないので配列にしない）
      const p = i * 4;
      const g = (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) / 255;
      const v = ((a0 * (1 - wy) + a1 * wy) * g + (b0 * (1 - wy) + b1 * wy) - 0.5) * k + 0.5;
      mask[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }
  mk('ガイデッドフィルタ');
  timings['合計'] = performance.now() - whole;
  lastEnd = performance.now();

  self.postMessage({ type: 'mask', id, width: capture, height: capture, data: mask, timings }, [mask.buffer]);
}

// coeffSize → capture の拡大で使う、位置と重みの表（bilinear() と同じ式）
let upX0 = null; let upX1 = null; let upWx = null;
let upY0 = null; let upY1 = null; let upWy = null;
function buildUpscaleTable() {
  const s = coeffSize / capture;
  upX0 = new Int32Array(capture); upX1 = new Int32Array(capture); upWx = new Float64Array(capture);
  upY0 = new Int32Array(capture); upY1 = new Int32Array(capture); upWy = new Float64Array(capture);
  for (let i = 0; i < capture; i++) {
    const f = Math.min((i + 0.5) * s - 0.5, coeffSize - 1);
    const i0 = Math.max(0, Math.floor(f));
    const i1 = Math.min(i0 + 1, coeffSize - 1);
    upX0[i] = upY0[i] = i0;
    upX1[i] = upY1[i] = i1;
    upWx[i] = upWy[i] = f - i0;
  }
}

/** 返ってきたバッファが使えればそれを、駄目なら新しく確保する */
function take(out, len) {
  return out && out.length === len ? out : new Float32Array(len);
}

/** 0.5 を境に確率のコントラストを立てる（遷移帯の幅を 1/k にする） */
function sharpen(src, k, dst = new Float32Array(src.length)) {
  if (!k || k === 1) { dst.set(src); return dst; }
  for (let i = 0; i < src.length; i++) {
    const v = (src[i] - 0.5) * k + 0.5;
    dst[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return dst;
}

/** 正方形マスクのバイリニア拡大。dst を渡せばそこへ書く。 */
function bilinear(src, w, h, size, dst = new Float32Array(size * size)) {
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
function boxFilter(src, size, r, dst = new Float32Array(size * size), tmp = new Float32Array(size * size)) {
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

/**
 * ガイデッドフィルタ (He et al., 2010) の線形係数 a, b を求める。
 * 出力は「マスク ≒ a * ガイド輝度 + b」の形でガイドの輪郭に沿う。
 * a, b は元画像より滑らかなので、低解像度で求めて拡大しても品質がほとんど落ちない
 * （= fast guided filter）。
 */
function guidedCoeffs(I, p, size, r, eps, buf = (key, len) => new Float32Array(len)) {
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
