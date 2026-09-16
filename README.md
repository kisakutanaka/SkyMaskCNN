# SkyMaskCNN

モバイルブラウザ上で、カメラ映像から**空だけをリアルタイムに分離**する実装。
分離は 199KB の CNN（`tinyskynet_skyseg_256.onnx`）を onnxruntime-web で回します。

**デモ**: https://kisakutanaka.github.io/SkyMaskCNN/

**画面と合成は [SkyMaskCV](https://github.com/kisakutanaka/SkyMaskCV) と同じもので、
背景分離の中身だけを差し替えてあります。** 古典CV と CNN を同じ入力・同じ表示・
同じ計測で並べて比べるためです。

左に元映像、右に分離結果を並べて表示します。入力は同梱の動画・静止画のほか、
カメラ（HTTPS が必要なので GitHub Pages 上でのみ利用可）に切り替えられます。
「怪獣合成」は空のレイヤーに怪獣を挟んで建物に隠れさせるモード、
「空塗りつぶし」は空を単色で塗るモードです（マスクの粗の確認に使えます）。
「分離のみ」は分離結果だけを大きく出します。元映像は消さず分離結果の真裏に
重ねてあります（`display:none` にしても画面外へ出しても、ブラウザが video の更新を
止めてしまい、動画とカメラで ms/frame が出なくなるため）。

## 分離の流れ

```
映像 → 512×512 で取り込み（getImageData 1回）
     → 2×2 の面積平均で 256×256 へ縮小 + ImageNet 正規化
     → CNN 推論 → [1, 1, 128, 128] のロジット
     → sigmoid → 空である確率（0..1）
     → fast guided filter（係数は 256 で求め、512 の輝度ガイドに当てる）
     → 確率のままアルファへ載せ、GPU が元の縦横比へ拡大
```

モデル入力は **256×256 の正方形固定**なので、縦横比は取り込みで一度潰し、
マスクを元の縦横比へ引き伸ばして戻します。モデル出力は 128×128 と粗いため、
輪郭は映像そのものを輝度ガイドにしたガイデッドフィルタ（He et al., 2010 の
fast guided filter）で被写体の縁へ吸着させています。この後段があるおかげで
モデル本体を重くせずに済んでいます。

マスクは **0/1 に潰さず確率のままアルファに流します**。SkyMaskCV が二値なのに対し、
こちらは境界が確率のままぼけるので、合成が馴染みます。

## 速度

手元の Mac + Chrome（ヘッドレス, `--disable-gpu`）で 推論 15〜16ms /
フレーム全体 20〜25ms（512×512 のマスク）。**実機の iPhone (Safari) では
フレーム全体 44ms = 23fps 相当で、30fps の予算 33ms に収まっていません**
（カメラモード・1分半ほど回して熱ダレした状態・refineSize 512）。
SkyMaskCV 側は iPhone SE (第3世代) でフレーム全体 7.5ms なので、**約6倍**です。

実機での工程別内訳（直近60フレームの中央値）:

| 工程 | 実機 iPhone | 手元の Mac | どこで走るか |
|---|---|---|---|
| **`getImageData`（GPU からの読み戻し）** | **19 ms (45%)** | 0.2 ms | メイン |
| 推論と Worker 往復 | 13 ms (31%) | 7.3 ms | Worker |
| ガイデッドフィルタ | 8 ms (19%) | 5.7 ms | メイン |
| その他（`drawImage` / 縮小と正規化 / sigmoid） | 2 ms | 1.9 ms | メイン |
| 分離の合計 | **42 ms** | 15.4 ms | |

**最大のコストは推論ではなく、512×512 の読み戻し (`getImageData`) です。**
Mac で `--disable-gpu` だと 0.2ms しか出ないため、この工程は**実機でしか見えません**。
取り込みも読み戻しも `capture = max(256, refineSize)` の解像度で行うので、
`refineSize` を下げると画素数の2乗で効きます（iOS の `performance.now()` は
1ms 刻みに丸められるため、1ms 未満の工程は 0 と出ます）。

推論は `ort.env.wasm.proxy = true` で Web Worker 側に出しているので、
数百 ms かかる端末でもメインスレッドの描画は止まりません。

## 起動コスト

SkyMaskCV と違い、**最初の1フレームの前に onnxruntime-web の wasm とモデルを
取ってくる待ちがあります**（その間 stats に「モデルを読み込んでいます…」と出ます）。
**通信量の 9 割はモデルではなくランタイム**です。

| 起動時に取るもの | サイズ（brotli 後・実測） |
|---|---|
| `ort-wasm-simd-threaded.wasm` | **2,926 KB** |
| `tinyskynet_skyseg_256.onnx` | 199 KB |
| `ort.wasm.min.js` + `...threaded.mjs` | 16 + 9 KB |
| `index.html` + `sky-segmenter.js` | 23 KB |
| 合計 | **約 3.1 MB** |

モデルを 199KB に絞った効果は、通信量では**ほぼ見えません**（モデルを 0 にしても
2.9MB 残る）。49K パラメータ 1 本のために汎用ランタイムを積んでいる構図です。
（この表は手法に要る分だけです。デモの既定ソースは同梱動画 4.9MB なので、
ページ全体の通信量はさらにその分がかかります。）

読み込む ort は **wasm 専用ビルド (`ort.wasm.min.js`) を指定しています。**
既定の `ort.min.js` は WebGPU/WebNN 対応を含む jsep 版の wasm (5,234 KB) を
取りに行きますが、`sky-segmenter.js` は `executionProviders: ['wasm']` 固定で
その機能を使っていません。差し替えで**通信量 5.4MB → 3.1MB、起動は手元の実測で
1.6 秒 → 0.9 秒**（キャッシュ無効・CDN が温まった状態）になりました。回線が細い
環境ほど差は大きく、別環境では 5.1 秒 → 1.1 秒 という実測もあります。
**推論速度と精度は変わりません**（推論の差はフレーム間のばらつきの範囲）。
引き換えに将来 WebGPU / WebNN を試す選択肢は閉じますが、戻すのは
`index.html` の script タグ 1 行を `ort.min.js` に戻すだけです。

wasm は GitHub Pages が COOP/COEP ヘッダを付けられない = SharedArrayBuffer が
使えないため、シングルスレッドに固定しています。

## 構成

| ファイル | 役割 |
|---|---|
| `index.html` | 画面・ループ・合成。SkyMaskCV とは分離の呼び出しだけが違う |
| `sky-segmenter.js` | 取り込み・正規化・推論・ガイデッドフィルタ。依存はグローバルの `ort` だけ |
| `models/` | 同梱モデルと、その出所・ライセンス（[models/README.md](models/README.md)）|

`sky-segmenter.js` は[隣の SkySegmentation](https://github.com/kisakutanaka/SkySegmentation)
から持ってきたもので、そのままコピーすれば他プロジェクトでも動きます。
モデルの学習・評価コードもそちらにあります。

## 既知の限界

- 明るく平坦な壁面（白い建物など）を空と誤判定することがある（49K パラメータの容量不足）
- 入力が正方形固定なので、極端な縦長・横長では一度潰した分だけ細い構造が不利になる
- 時間方向の平滑化は入れていないため、動画ではフレーム間のちらつきが残る
- 実機のフレームコストが 30fps の予算に収まっていない（上記「速度」）
- `file://` では ES module と wasm の取得に失敗する。ローカル確認は HTTP サーバ経由で
  （例: `python3 -m http.server`）

## 素材

- **`kaiju.png`**: [openclipart #346163 "Monster D" by mickleness](https://openclipart.org/detail/346163/monster-d)
  — **Public Domain (CC0)**。透明な余白を落として 550×600 に縮小しています
