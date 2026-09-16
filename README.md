# SkyMaskCNN

モバイルブラウザ上で、カメラ映像から**空だけをリアルタイムに分離**する実装。
分離は 199KB の CNN（`tinyskynet_skyseg_256.onnx`）を onnxruntime-web で回します。

**デモ**: https://kisakutanaka.github.io/SkyMaskCNN/
**ステップ解説**: https://kisakutanaka.github.io/SkyMaskCNN/steps.html
（1 枚が空マスクになるまでを、実際に計算しながら 1 段ずつ画像で見せるページ）

**画面と合成は [SkyMaskCV](https://github.com/kisakutanaka/SkyMaskCV) と同じもので、
背景分離の中身だけを差し替えてあります。** 古典CV と CNN を同じ入力・同じ表示・
同じ計測で並べて比べるためです。1 フレームにつき 1 回分離し、stats も同じ形
（`ms/frame` と内訳）で出します。

## 結論

**実機 iPhone SE (第3世代) で 1 フレーム 27ms（開始直後）〜46ms（1分後）。
同じ端末で SkyMaskCV は 7.5ms。** 30fps の予算 33ms に対して、CNN 側は
冷えているうちだけ収まり、熱ダレすると外れます。

内訳を追った結果、**残っているコストはほぼ推論そのもの**です。

| | 開始直後 | 1分後 |
|---|---|---|
| 推論 | 13 ms | 23 ms |
| ガイデッドフィルタ | 3 ms | 5 ms |
| 取り込み・読み戻し・正規化 | 2 ms | 4 ms |

読み戻し (`getImageData`) は当初 19ms で最大のコストでしたが、**1〜2ms まで
落ちました**（下記「速度」）。**熱ダレで一様に 2 倍**になるのが効いていて、
ここから先はモデルを替えるしかありません。

## 分離の流れ

```
映像 → 512×512 へ drawImage（メインスレッド。GPU に積むだけ）
     → createImageBitmap で Worker へ転送（ここから下は全部 Worker）
     → getImageData
     → 2×2 の面積平均で 256×256 へ縮小 + ImageNet 正規化
     → CNN 推論 → [1, 1, 128, 128] のロジット
     → sigmoid → 空である確率（0..1）
     → fast guided filter（係数は 128 で求め、512 の輝度に当てながら 1 パスで合成）
     → 確率のままアルファへ載せ、GPU が元の縦横比へ拡大
```

モデル入力は **256×256 の正方形固定**なので、縦横比は取り込みで一度潰し、
マスクを元の縦横比へ引き伸ばして戻します。モデル出力は 128×128 と粗いため、
輪郭は映像そのものを輝度ガイドにしたガイデッドフィルタ（He et al., 2010 の
fast guided filter）で被写体の縁へ吸着させています。この後段があるおかげで
モデル本体を重くせずに済んでいます。

線形係数 a, b は **128×128 で求めます**（`coeffSize`）。a, b は元画像より滑らかなので
粗く求めて拡大しても品質がほとんど落ちない、というのが fast guided filter の肝で、
論文の推奨も入力の 1/4 です。**モデル出力がちょうど 128×128 なので、拡大せずに
そのまま係数計算へ渡せます**。係数の拡大・輝度ガイド・合成は 512×512 の 1 パスに
まとめてあり、中間配列は作りません。

マスクは **0/1 に潰さず確率のままアルファに流します**。SkyMaskCV が二値なのに対し、
こちらは境界が確率のままぼけるので、合成が馴染みます。

**分離に渡した 1 枚は控えておき、マスクと対で合成します。** 分離は数十 ms かかるので、
その間に映像は先へ進みます。最新フレームに返ってきたマスクを載せると、カメラを
振ったときに輪郭が 1〜3 フレーム分ずれます。

## 速度

分離は丸ごと Worker の中にあります。メインスレッドに残るのは控えの `drawImage` と
`createImageBitmap`、そして合成だけです。

**実機 iPhone SE (第3世代) / Safari**（カメラ・refineSize 512・直近60枚の中央値）:

| 経過 | 1 フレーム | 推論 | ガイデッドフィルタ | `getImageData` |
|---|---|---|---|---|
| 5 秒 | 27 ms (37fps相当) | 13 | 3 | 1 |
| 72 秒 | 46 ms (22fps相当) | 23 | 5 | 2 |

手元の Mac + Chrome（ヘッドレス, `--disable-gpu`）では 18ms/frame（分離 13.7 /
控え・合成 4.7）。

### 読み戻しをどうやって消したか

当初は分離もメインスレッドで動かしていて、**`getImageData` が 19ms（全体の45%）と
最大のコスト**でした。しかも **`refineSize` を 512 → 256 に落としても 18ms のまま**で、
画素数に比例しませんでした。中身が転送量ではなく「**GPU がいま描いている絵の
完了待ち**」だったからです。

分離を Worker へ移し、メインスレッドは `drawImage` で 512×512 に描いて
`createImageBitmap` で渡すだけにしたところ、**19ms → 1〜2ms になりました**。
Worker 側が読むのは既に出来上がった `ImageBitmap` で、待つ相手がいません。
メインスレッド側も GPU に積むだけで完了を待ちません。

そのぶん**解像度を落とさずに速くなった**ので、`refineSize` は 512 のままです。

### 熱ダレ

1 分回すと推論 13→23ms、ガイデッドフィルタ 3→5ms と、**計算そのものが一様に
2 倍近く遅くなります**。読み戻しを消しても、ここは残ります。

## 起動コスト

SkyMaskCV と違い、**最初の1フレームの前に onnxruntime-web の wasm とモデルを
取ってくる待ちがあります**（その間 stats に「モデルを読み込んでいます…」と出ます）。
**通信量の 9 割はモデルではなくランタイム**です。

| 起動時に取るもの | サイズ（brotli 後・実測） |
|---|---|
| `ort-wasm-simd-threaded.wasm` | **2,926 KB** |
| `tinyskynet_skyseg_256.onnx` | 199 KB |
| `ort.wasm.min.mjs` + `...threaded.mjs` | 16 + 9 KB |
| `index.html` + `sky-segmenter.js` | 23 KB |
| 合計 | **約 3.1 MB** |

モデルを 199KB に絞った効果は、通信量では**ほぼ見えません**（モデルを 0 にしても
2.9MB 残る）。49K パラメータ 1 本のために汎用ランタイムを積んでいる構図です。
（この表は手法に要る分だけです。デモの既定ソースは同梱動画 4.9MB なので、
ページ全体の通信量はさらにその分がかかります。）

読み込む ort は **wasm 専用ビルド (`ort.wasm.min.mjs`) を Worker 側が import します**
（メインスレッドは ort を読みません）。既定の `ort.min.js` は WebGPU/WebNN 対応を含む
jsep 版の wasm (5,234 KB) を取りに行きますが、`executionProviders: ['wasm']` 固定で
その機能を使っていません。差し替えで**通信量 5.4MB → 3.1MB、起動は手元の実測で
1.6 秒 → 0.9 秒**（キャッシュ無効・CDN が温まった状態）になりました。回線が細い
環境ほど差は大きく、別環境では 5.1 秒 → 1.1 秒 という実測もあります。
**推論速度と精度は変わりません**。引き換えに将来 WebGPU / WebNN を試す選択肢は
閉じますが、戻すのは `sky-segmenter.js` の `ortUrl` を `ort.min.mjs` に戻すだけです。

wasm は GitHub Pages が COOP/COEP ヘッダを付けられない = SharedArrayBuffer が
使えないため、シングルスレッドに固定しています。

## 構成

| ファイル | 役割 |
|---|---|
| `sky-mask.js` | **計算そのもの**。DOM も Worker も ONNX も知らない純粋関数だけ |
| `sky-segmenter.worker.js` | Worker の配管。ONNX セッションと取り込み。計算は上に任せる |
| `sky-segmenter.js` | 公開 API（メインスレッド側）。取り込みと Worker とのやりとりだけ |
| `index.html` | 画面・ループ・合成。SkyMaskCV とは分離の呼び出しだけが違う |
| `steps.html` | ステップ解説。同じ関数を同じ順で呼び直して 1 段ずつ表示する |
| `models/` | 同梱モデルと、その出所・ライセンス（[models/README.md](models/README.md)）|

計算と配管を分けてあるので、[steps.html](steps.html) は本番と同じ関数を呼び直して
途中経過を出せます。**説明用に式を書き直すと実物とずれていくので、最後に
`sky-segmenter.js`（本番の経路）の出力と突き合わせ、1 画素も違わないことを
ページ上で確かめます**（隣の SkyMaskCV の `tools/steps.mjs` と同じ規律）。

`sky-segmenter.js` / `sky-segmenter.worker.js` / `sky-mask.js` の 3 つで 1 組で、そのままコピーすれば
他プロジェクトでも動きます（元は[隣の SkySegmentation](https://github.com/kisakutanaka/SkySegmentation)
の 1 ファイル版）。module worker と OffscreenCanvas が要ります（Safari 16.4+ / Chrome 69+）。
モデルの学習・評価コードもそちらにあります。

## 既知の限界

- 明るく平坦な壁面（白い建物など）を空と誤判定することがある（49K パラメータの容量不足）
- 入力が正方形固定なので、極端な縦長・横長では一度潰した分だけ細い構造が不利になる
- `refineSize` は `inputSize` (256) の整数倍に、`coeffSize` は 2 の冪に丸められる
  （丸めずに使うと縮小ループが取り込み範囲の外を読む）
- 時間方向の平滑化は入れていないため、動画ではフレーム間のちらつきが残る
- 熱ダレ後は 1 フレーム 46ms まで落ちる（開始直後は 27ms）
- 右ペインは分離 1 枚分だけ古いフレームを映す。左の元映像とは 27〜46ms ずれる
- `file://` では ES module と wasm の取得に失敗する。ローカル確認は HTTP サーバ経由で
  （例: `python3 -m http.server`）

## 素材

- **`kaiju.png`**: [openclipart #346163 "Monster D" by mickleness](https://openclipart.org/detail/346163/monster-d)
  — **Public Domain (CC0)**。透明な余白を落として 550×600 に縮小しています
