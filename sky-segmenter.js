/**
 * sky-segmenter.js
 *
 * カメラ映像 / 画像から「空である確率マップ」だけを返す最小モジュール。
 * 分離は丸ごと Worker (sky-segmenter.worker.js) で走る。メインスレッドに
 * 残るのは createImageBitmap だけなので、描画は分離に巻き込まれない。
 * この 2 ファイルをコピーすれば他プロジェクトでも動きます（依存は
 * onnxruntime-web だけで、それも Worker 側が CDN から読む）。
 *
 * 使い方:
 *   const seg = await createSkySegmenter();
 *   const mask = await seg.segment(videoElement);   // { width, height, data }
 *   // mask.data[y * mask.width + x] = 0.0(空でない) 〜 1.0(空)
 *
 * 必要なもの: module worker と OffscreenCanvas（Safari 16.4+ / Chrome 69+）。
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
  // 係数 a, b を求める解像度。a, b は元画像より滑らかなので、粗く求めて拡大しても
  // 品質がほとんど落ちない（fast guided filter の肝で、論文の推奨は入力の 1/4）。
  // モデル出力がちょうど 128×128 なので、128 にすると拡大そのものも要らなくなる。
  // 2 の冪に丸められる。実際の値は生成後に coeffSize で確認できる。
  coeffSize: 128,
  // 出力マスクの解像度。inputSize の整数倍に丸められる（384 を渡せば 512 になる）。
  // ここを上げるほど輪郭がシャープになるが、取り込みと読み戻しもこの解像度で
  // 行う。実際の値は生成後に maskSize で確認できる。
  refineSize: 512,
  // 確率の 0→1 遷移をどれだけ立てるか。1 でそのまま、大きいほど輪郭がくっきりする。
  // 0.5 を境に (p-0.5)*k+0.5 で伸ばすだけなので、位置はずらさず境界の幅だけ縮む。
  edgeSharpness: 6,
  // onnxruntime-web。ESM 版を Worker 側が import する。
  ortUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.wasm.min.mjs',
  ortWasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/',
};

export async function createSkySegmenter(options = {}) {
  const cfg = { ...SKY_SEGMENTER_DEFAULTS, ...options };
  // ?v= は index.html 側と揃える（古いキャッシュとの食い違いを防ぐ）
  const worker = new Worker(new URL('./sky-segmenter.worker.js?v=5', import.meta.url), {
    type: 'module',
  });

  // Worker とのやりとりは常に 1 往復ずつ。投げっぱなしにすると推論が重なり、
  // 古いフレームの結果を待つ分だけ遅れが積み上がる。
  let seq = 0;
  const waiting = new Map();
  worker.onmessage = ({ data }) => {
    const done = waiting.get(data.id);
    if (!done) return;
    waiting.delete(data.id);
    if (data.type === 'error') done.reject(new Error(data.message));
    else done.resolve(data);
  };
  worker.onerror = (e) => {
    const err = new Error(e.message || 'sky-segmenter.worker.js が読み込めませんでした');
    for (const { reject } of waiting.values()) reject(err);
    waiting.clear();
  };
  const ask = (msg, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve, reject });
      worker.postMessage({ ...msg, id }, transfer);
    });

  const ready = await ask({ type: 'init', cfg });

  // 取り込み用。メインスレッドに残すのはこの 1 枚への drawImage だけで、
  // 読み戻し (getImageData) は Worker 側にある。
  // 元の要素から直接 createImageBitmap すると、実測で 1958×1290 の静止画に
  // 14.8ms かかる（フル解像度のビットマップを作るため。resizeWidth を付けても
  // 変わらない）。小さいキャンバスへ一度描いてから渡すと 0.6ms で済む。
  const capture = Object.assign(document.createElement('canvas'), {
    width: ready.maskSize,
    height: ready.maskSize,
  });
  const captureCtx = capture.getContext('2d');

  // 前のフレームで返ってきたマスク。次のフレームで Worker に返して使い回す。
  let spare = null;

  /**
   * @param {CanvasImageSource} source video / canvas / img / ImageBitmap
   * @returns {Promise<{width:number, height:number, data:Float32Array}>} 空である確率(0..1)
   *
   * ※ data は Worker と往復して使い回しているバッファです。次の segment() で
   *   持っていかれるので、フレームを跨いで持つ場合はコピーしてください。
   *   同時に呼べるのは 1 枚までです（前の 1 枚を待ってから次を呼ぶ）。
   */
  async function segment(source) {
    // 取り込みはここだけ。drawImage は GPU に積むだけで完了を待たず、
    // createImageBitmap は非同期なので、GPU の描画完了を待つ間もメインスレッドは
    // 止まらない（元の要素から直接 createImageBitmap すると、1958×1290 の静止画で
    // 14.8ms かかる。小さいキャンバスへ一度描いてから渡すと 0.6ms で済む）。
    // 正方形に潰すのもここ（縦横比は呼び出し側が拡大で戻す）。
    captureCtx.drawImage(source, 0, 0, capture.width, capture.height);
    const bitmap = await createImageBitmap(capture);
    const transfer = [bitmap];
    if (spare) transfer.push(spare.buffer);
    const res = await ask({ type: 'frame', bitmap, out: spare }, transfer);
    spare = res.data;
    return res;
  }

  return {
    segment,
    inputSize: ready.inputSize,
    maskSize: ready.maskSize,
    coeffSize: ready.coeffSize,
    dispose: () => worker.terminate(),
  };
}
