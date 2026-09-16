# 同梱モデルについて

このリポジトリが含むモデルは **`tinyskynet_skyseg_256.onnx` (199KB)** の1本だけで、
ページが読み込むのもこれです。教師モデルや比較用の旧モデルは、学習コードと一緒に
隣の [SkySegmentation](https://github.com/kisakutanaka/SkySegmentation) に置いてあります
（このリポジトリはデモだけを持ちます）。

## `tinyskynet_skyseg_256.onnx` (199 KB)

| 項目 | 内容 |
|---|---|
| 出所 | [SkySegmentation](https://github.com/kisakutanaka/SkySegmentation) の `tools/train.py` で学習。SkySeg を教師にした蒸留 |
| 教師 | [SkySeg](https://huggingface.co/JianyuanWang/skyseg)（U-2-Net, 168MB, **MIT**。出所は [xiongzhu666/Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing)）|
| ライセンス | **MIT**（重み・コードとも。ただし下記の学習データの注記を参照）|
| 学習データ | Open Images V7 のうち「空」と「市街地」の両方のラベルが付いた写真 3,901 枚。**画像は CC BY 2.0**（商用可）|
| パラメータ数 | 49,233 |
| 入力 / 出力 | `[1, 3, 256, 256]` → `[1, 1, 128, 128]` のロジット |

出力が 128×128 と粗いのは、境界の精緻化を `sky-segmenter.js` 側のガイデッドフィルタで
行っているためです（モデルを重くせずにエッジを立てられる）。

検証 335 枚を教師 SkySeg の出力と突き合わせた一致度は 全体 IoU 0.8888 /
境界帯 ±8px の一致率 0.9223 です。**このモデルはその SkySeg を教師にして学習している
ので、この指標はモデル側に有利**で、「SkySeg の出力にどれだけ近いか」以上の意味は
ありません。測定条件と旧モデルとの比較は SkySegmentation 側の `models/README.md` に
あります。

**既知の弱点**: 明るく平坦な壁面（白い建物など）を空と誤判定することがあります。
容量不足（49K パラメータ）が理由なので、チャンネル数を増やせば改善する見込みです。

## ライセンスに関する注意

### SkySeg（教師）

- 重みは **MIT**。HuggingFace の配布元・元リポジトリともに MIT を明記しています。
- ただし**学習データは非公開**です（作者は「高精度版は自社プロダクトで使っている」と述べています）。
  ADE20K のように「非商用限定」と明記されたデータではありませんが、**素性を検証できません**。
  既知の制限が無いかわりに未知が残る、という交換だと理解してください。

### TinySkyNet（同梱モデル）

- 重みは MIT ですが、**教師の出力を使って学習している**ため、教師の性質を引き継いでいると
  考えるのが安全です。
- 学習に使った Open Images の写真自体は CC BY 2.0 で商用利用できます。

## 差し替え

別のモデルを試すときは `sky-segmenter.js` の `SKY_SEGMENTER_DEFAULTS`
（`modelUrl` / `inputSize` / `skyClassIndex`）と、`index.html` の `SEGMENTER_OPTIONS` を
書き換えるだけです。出力チャンネルが 1 なら sigmoid、複数なら
「`skyClassIndex` 対 それ以外の最大」の2値ソフトマックスに自動で分岐します。
