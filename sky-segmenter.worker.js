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
let ctx = null;
let guideHi = null;
let guideLo = null;

// 作業用バッファ。毎フレーム確保するとモバイルで GC が時々 1 フレーム分の
// 時間を丸ごと奪う。Worker 側に置けるものは全部ここで 1 回だけ取る。
const scratch = new Map();
const buf = (key, len) => {
  let a = scratch.get(key);
  if (!a || a.length !== len) scratch.set(key, (a = new Float32Array(len)));
  return a;
};

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') await init(msg.cfg, msg.id);
    else if (msg.type === 'frame') await frame(msg);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: err?.message ?? String(err) });
  }
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

  const canvas = new OffscreenCanvas(capture, capture);
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  guideHi = new Float32Array(capture * capture); // 高解像度の輝度ガイド
  guideLo = new Float32Array(size * size); // 統計量計算用に縮小した輝度

  self.postMessage({ type: 'ready', id, inputSize: size, maskSize: capture });
}

async function frame({ id, bitmap, out }) {
  const timings = {};
  let t = performance.now();
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

  // 3. 高解像度のガイド（輝度）。輪郭の細さはここの解像度で決まる
  for (let i = 0, p = 0; i < guideHi.length; i++, p += 4) {
    guideHi[i] = (rgba[p] * 0.299 + rgba[p + 1] * 0.587 + rgba[p + 2] * 0.114) / 255;
  }
  mk('高解像度ガイド');

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
    self.postMessage({ type: 'mask', id, width: w, height: h, data: flat, timings }, [flat.buffer]);
    return;
  }

  // 6. fast guided filter:
  //    線形係数 a, b は低解像度（inputSize）で求め、それを拡大して
  //    高解像度のガイドに当てる。box filter の計算量は据え置きのまま、
  //    輪郭だけ capture の解像度で吸着する（He et al. 2010 の 4.1 節）。
  const pLo = bilinear(coarse, w, h, size, buf('pLo', size * size));
  const { a, b } = guidedCoeffs(guideLo, pLo, size, cfg.refineRadius, cfg.refineEps, buf);
  const aHi = bilinear(a, size, size, capture, buf('aHi', capture * capture));
  const bHi = bilinear(b, size, size, capture, buf('bHi', capture * capture));

  // 返す配列だけは転送で手放すので、前のフレームで返ってきたものを受け取って使う
  const mask = take(out, guideHi.length);
  const k = cfg.edgeSharpness;
  for (let i = 0; i < mask.length; i++) {
    const v = (aHi[i] * guideHi[i] + bHi[i] - 0.5) * k + 0.5;
    mask[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  mk('ガイデッドフィルタ');
  timings['合計'] = performance.now() - whole;

  self.postMessage({ type: 'mask', id, width: capture, height: capture, data: mask, timings }, [mask.buffer]);
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
