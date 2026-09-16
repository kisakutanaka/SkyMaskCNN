/**
 * sky-segmenter.worker.js
 *
 * 分離の配管。ONNX セッションと取り込み用のキャンバスを持ち、計算そのものは
 * sky-mask.js の純粋関数に任せる。メインスレッドに残すのは
 * createImageBitmap（非同期。GPU の完了を待たない）だけ。
 *
 * 実機 (iPhone/Safari) の内訳で、いちばん高いのは推論ではなく
 * getImageData の 18〜19ms だった。これは GPU の描画完了を待つ同期で、
 * 解像度を 1/4 にしても減らない固定費（512:19ms / 256:18ms）。
 * ここへ移したら、読む相手が出来上がった ImageBitmap になり 1〜2ms になった。
 *
 * sky-segmenter.js からだけ使われる。プロトコルは 2 種類:
 *   { type:'init',  cfg }               → { type:'ready', inputSize, maskSize, coeffSize }
 *   { type:'frame', bitmap, out }       → { type:'mask', width, height, data }
 * out は前のフレームで返した Float32Array。転送で返してもらって使い回す
 * （毎フレーム確保すると 1 フレーム 7MB 近いゴミになる）。
 */

import {
  bilinear, guidedCoeffs, plan, refine, sharpen, shrinkGuide, toInput, toProbability, upscaleTable,
} from './sky-mask.js?v=5';

let cfg = null;
let ort = null;
let session = null;
let inputName = '';
let outputName = '';
let size = 0; // モデル入力の一辺
let capture = 0; // 取り込み解像度 = size * pool
let coeffSize = 0; // ガイデッドフィルタの係数を求める解像度
let coeffRadius = 1; // その解像度での半径
let shrink = 1; // size / coeffSize
let table = null; // coeffSize → capture の拡大表
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
/** 返ってきたバッファが使えればそれを、駄目なら新しく確保する */
const take = (out, len) => (out && out.length === len ? out : new Float32Array(len));

// 受け取った順に 1 枚ずつ処理する。onmessage を直接 async にすると、
// session.run を待っている間に次のメッセージが割り込んで、共有している
// 作業用バッファを壊す。
let chain = Promise.resolve();

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

  ({ size, capture, coeffSize, coeffRadius, shrink } = plan(cfg));
  table = upscaleTable(coeffSize, capture);

  const canvas = new OffscreenCanvas(capture, capture);
  ctx = canvas.getContext('2d', { willReadFrequently: true });
  guideLo = new Float32Array(size * size); // 統計量計算用に縮小した輝度
  guideCo = shrink === 1 ? guideLo : new Float32Array(coeffSize * coeffSize);

  self.postMessage({ type: 'ready', id, inputSize: size, maskSize: capture, coeffSize });
}

async function frame({ id, bitmap, out }) {
  // 1. 受け取ったビットマップを取り込む（呼び出し側で capture×capture に
  //    潰してあるので、ここは等倍のコピー）
  ctx.drawImage(bitmap, 0, 0, capture, capture);
  bitmap.close();
  const { data: rgba } = ctx.getImageData(0, 0, capture, capture);

  // 2. モデル入力（NCHW・正規化済み）と輝度ガイドを作る。proxy=false なので
  //    input も使い回せる（ort が転送しないため、次のフレームでも生きている）。
  const input = buf('input', 3 * size * size);
  toInput(rgba, capture, size, input, guideLo);

  // 3. 推論。出力は低解像度のロジット
  const outputs = await session.run({
    [inputName]: new ort.Tensor('float32', input, [1, 3, size, size]),
  });
  const logits = outputs[outputName];
  const [, numClasses, h, w] = logits.dims;

  // 4. 「空である確率」へ
  const area = h * w;
  const coarse = toProbability(
    logits.data, numClasses, area, cfg.skyClassIndex, cfg.skyMargin, buf('coarse', area),
  );

  if (!cfg.refineRadius) {
    const flat = sharpen(coarse, cfg.edgeSharpness, take(out, area));
    self.postMessage({ type: 'mask', id, width: w, height: h, data: flat }, [flat.buffer]);
    return;
  }

  // 5. fast guided filter。係数 a, b は coeffSize で求め、拡大して
  //    capture の輝度に当てる（He et al. 2010 の 4.1 節）。
  const guide = shrinkGuide(guideLo, size, shrink, guideCo);
  // モデル出力がちょうど係数解像度なら、拡大せずそのまま使える（既定はこの経路）
  const pCo = w === coeffSize && h === coeffSize
    ? coarse
    : bilinear(coarse, w, h, coeffSize, buf('pCo', coeffSize * coeffSize));
  const { a, b } = guidedCoeffs(guide, pCo, coeffSize, coeffRadius, cfg.refineEps, buf);

  const mask = refine(
    a, b, rgba, coeffSize, capture, cfg.edgeSharpness, table, take(out, capture * capture),
  );
  self.postMessage({ type: 'mask', id, width: capture, height: capture, data: mask }, [mask.buffer]);
}
