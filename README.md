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

手元の Mac + Chrome（ヘッドレス, `--disable-gpu`）で **推論 15ms 前後 /
フレーム全体 20〜25ms**（512×512 のマスク）。**実機のモバイルでは未計測**です。
推論は `ort.env.wasm.proxy = true` で Web Worker 側に出しているので、
数百 ms かかる端末でもメインスレッドの描画は止まりません。

SkyMaskCV と違い、**最初の1フレームの前にモデル（199KB）と onnxruntime-web の wasm を
CDN から取ってくる待ちがあります**（その間 stats に「モデルを読み込んでいます…」と出ます）。
wasm は GitHub Pages が COOP/COEP ヘッダを付けられない = SharedArrayBuffer が使えない
ため、シングルスレッドに固定しています。

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
- `file://` では ES module と wasm の取得に失敗する。ローカル確認は HTTP サーバ経由で
  （例: `python3 -m http.server`）

## 素材

- **`kaiju.png`**: [openclipart #346163 "Monster D" by mickleness](https://openclipart.org/detail/346163/monster-d)
  — **Public Domain (CC0)**。透明な余白を落として 550×600 に縮小しています
