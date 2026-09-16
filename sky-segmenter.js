/**
 * sky-segmenter.js
 *
 * カメラ映像 / 画像から「空である確率マップ」だけを返す最小モジュール。
 * 依存はグローバルの `ort` (onnxruntime-web) だけなので、このファイルを
 * そのままコピーすれば他プロジェクトでも動きます。
 *
 * 使い方:
 *   const seg = await createSkySegmenter();
 *   const mask = await seg.segment(videoElement);   // { width, height, data }
 *   // mask.data[y * mask.width + x] = 0.0(空でない) 〜 1.0(空)
 */

export const SKY_SEGMENTER_DEFAULTS = {
  // ---- ここを差し替えればモデルを変更できます ----
  modelUrl: './models/tinyskynet_skyseg_256.onnx',
  inputSize: 256, // モデルの入力解像度（この ONNX は固定 256x256）
  skyClassIndex: 2, // ADE20K 系のモデルに差し替えたとき、150 クラス中 2 番が "sky"
  // ---------------------------------------------
  // 「空」が他クラスにこれだけ差をつけて勝ったときだけ空とみなすマージン。
  // 0 だと単純な argmax と同じで、霞んだ遠景の地面を空と誤判定しやすい
  // （実測では sky=+5.65 に対し land=+3.33 で空が勝ってしまう）。
  // 2 前後にすると、その帯だけが前景に戻り、本当の空（差が 8 前後）は影響を受けない。
  skyMargin: 2,
  // モデル出力は 128x128 と粗いので、映像そのものをガイドにして
  // マスクの境界を被写体の輪郭へ吸着させる（0 にすると無効）。
  refineRadius: 4, // ガイデッドフィルタの半径（inputSize 上の画素数）
  refineEps: 1e-4,
  // 出力マスクの解像度。inputSize の整数倍にすること。
  // ここを上げるほど輪郭がシャープになる（コストは後述の fast guided filter で微増のみ）。
  refineSize: 512,
  // 確率の 0→1 遷移をどれだけ立てるか。1 でそのまま、大きいほど輪郭がくっきりする。
  // 0.5 を境に (p-0.5)*k+0.5 で伸ばすだけなので、位置はずらさず境界の幅だけ縮む。
  edgeSharpness: 6,
  ortWasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/',
};

const MEAN = [0.485, 0.456, 0.406]; // ImageNet 正規化（学習時と同じ前処理）
const STD = [0.229, 0.224, 0.225];

export async function createSkySegmenter(options = {}) {
  const cfg = { ...SKY_SEGMENTER_DEFAULTS, ...options };

  // GitHub Pages は COOP/COEP ヘッダを付けられない = SharedArrayBuffer が使えないため、
  // wasm はシングルスレッドに固定する（自動判定に任せると初期化に失敗することがある）。
  ort.env.wasm.wasmPaths = cfg.ortWasmPaths;
  ort.env.wasm.numThreads = 1;
  // 推論を Web Worker 側で実行する。これがないと推論中(数百ms)メインスレッドが
  // 止まり、カメラ映像の描画がその間フリーズする。
  ort.env.wasm.proxy = true;

  const session = await ort.InferenceSession.create(cfg.modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  const size = cfg.inputSize;
  // 映像はマスク解像度で取り込み、モデル入力はそこから面積平均で縮小する。
  // getImageData が 1 回で済み、縮小も単純間引きよりきれいになる。
  const capture = Math.max(size, cfg.refineSize);
  const pool = Math.max(1, Math.round(capture / size)); // 何画素を 1 画素に畳むか
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = capture;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const guideHi = new Float32Array(capture * capture); // 高解像度の輝度ガイド
  const guideLo = new Float32Array(size * size); // 統計量計算用に縮小した輝度
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  /**
   * @param {CanvasImageSource} source video / canvas / img / ImageBitmap
   * @returns {Promise<{width:number, height:number, data:Float32Array}>} 空である確率(0..1)
   */
  async function segment(source) {
    // 1. マスク解像度で取り込む
    ctx.drawImage(source, 0, 0, capture, capture);
    const { data: rgba } = ctx.getImageData(0, 0, capture, capture);

    // 2. pool×pool の面積平均でモデル入力サイズへ縮小しつつ、NCHW / 正規化 / 低解像度ガイドを同時に作る
    //    ※ Worker 実行時に ArrayBuffer が転送されるため、input は毎回確保し直す
    const input = new Float32Array(3 * size * size);
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

    // 3. 高解像度のガイド（輝度）。輪郭の細さはここの解像度で決まる
    for (let i = 0, p = 0; i < guideHi.length; i++, p += 4) {
      guideHi[i] = (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) / 255;
    }

    // 4. 推論。出力は低解像度のロジット
    const outputs = await session.run({
      [inputName]: new ort.Tensor('float32', input, [1, 3, size, size]),
    });
    const logits = outputs[outputName];
    const [, numClasses, h, w] = logits.dims;
    const values = logits.data;

    // 5. 「空」と「空以外の最大」の 2 値ソフトマックス = sigmoid(sky - maxOther - margin)
    //    150 クラス全部の softmax より安く、境界がなめらかな確率になる。
    const area = h * w;
    const coarse = new Float32Array(area);
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

    if (!cfg.refineRadius) {
      return { width: w, height: h, data: sharpen(coarse, cfg.edgeSharpness) };
    }

    // 6. fast guided filter:
    //    線形係数 a, b は低解像度（inputSize）で求め、それを拡大して
    //    高解像度のガイドに当てる。box filter の計算量は据え置きのまま、
    //    輪郭だけ capture の解像度で吸着する（He et al. 2010 の 4.1 節）。
    const pLo = bilinear(coarse, w, h, size);
    const { a, b } = guidedCoeffs(guideLo, pLo, size, cfg.refineRadius, cfg.refineEps);
    const aHi = bilinear(a, size, size, capture);
    const bHi = bilinear(b, size, size, capture);

    const out = new Float32Array(guideHi.length);
    const k = cfg.edgeSharpness;
    for (let i = 0; i < out.length; i++) {
      const v = (aHi[i] * guideHi[i] + bHi[i] - 0.5) * k + 0.5;
      out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return { width: capture, height: capture, data: out };
  }

  return {
    segment,
    inputSize: size,
    maskSize: capture,
    dispose: () => session.release?.(),
  };
}

/** 0.5 を境に確率のコントラストを立てる（遷移帯の幅を 1/k にする） */
function sharpen(src, k) {
  if (!k || k === 1) return src;
  const dst = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = (src[i] - 0.5) * k + 0.5;
    dst[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return dst;
}

/** 正方形マスクのバイリニア拡大 */
function bilinear(src, w, h, size) {
  const dst = new Float32Array(size * size);
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

/** 移動平均（累積和による O(N) 実装） */
function boxFilter(src, size, r) {
  const tmp = new Float32Array(size * size);
  const dst = new Float32Array(size * size);
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
function guidedCoeffs(I, p, size, r, eps) {
  const n = size * size;
  const Ip = new Float32Array(n);
  const II = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    Ip[i] = I[i] * p[i];
    II[i] = I[i] * I[i];
  }
  const meanI = boxFilter(I, size, r);
  const meanP = boxFilter(p, size, r);
  const meanIp = boxFilter(Ip, size, r);
  const meanII = boxFilter(II, size, r);

  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const cov = meanIp[i] - meanI[i] * meanP[i];
    const varI = meanII[i] - meanI[i] * meanI[i];
    a[i] = cov / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  return { a: boxFilter(a, size, r), b: boxFilter(b, size, r) };
}
