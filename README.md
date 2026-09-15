# AIGen

ローカル環境（自分のPC）で完全無料で動く、以下3機能をまとめたAI画像・動画生成ツールです。

- **T2I（テキストから画像生成）**: プロンプトから画像を生成
- **画像スタイル変換**: 実写・イラストなどの画像を「アニメ風」「水彩画風」などにジャンルチェンジ
- **I2V（画像から動画生成）**: 1枚の画像から短い動画を生成

ブラウザで動くWeb UI（[Gradio](https://www.gradio.app/)）で、3つの機能をタブ切り替えで使えます。
すべて [diffusers](https://github.com/huggingface/diffusers) 上のオープンソースモデルを使用しており、
APIキーは不要・生成コストもかかりません（電気代とダウンロード帯域を除く）。

## 必要環境

- Python 3.10 以上
- **NVIDIA GPU（VRAM 8GB以上を推奨）+ CUDA**
  - VRAMが少ない場合は「省VRAMモード」をONにし、画像サイズやフレーム数を小さくすることで動作させられる場合があります。
  - GPUがない場合でも動作はしますが、CPUのみでの生成は非常に時間がかかります（画像1枚で数分〜数十分、動画はさらに長時間）。
- ディスク空き容量 **20GB以上**（モデルを複数ダウンロードするため）
- インターネット接続（初回のモデルダウンロード時のみ必要。以降はローカルキャッシュから読み込みます）

## セットアップ

```bash
# 1. リポジトリを取得（このディレクトリで作業している場合は不要）
cd AIGen

# 2. 仮想環境を作成
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate

# 3. PyTorchをCUDA対応版でインストール（環境に合わせてURLを変更してください）
#    お使いのGPU/CUDAバージョンは https://pytorch.org/get-started/locally/ で確認できます
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. 残りの依存関係をインストール
pip install -r requirements.txt
```

GPUがない、またはCPUのみで試す場合は手順3をスキップし、`pip install torch` のみでも動作します（低速）。

## 起動方法

```bash
python app.py
```

起動後、ターミナルに表示される `http://127.0.0.1:7860` のようなURLをブラウザで開いてください。

## 使い方

### 1. テキストから画像生成 (T2I)

1. モデルを選択（SDXLは高品質・高VRAM、SD1.5は軽量・高速）
2. プロンプト（生成したい内容、英語推奨）とネガティブプロンプトを入力
3. 「生成」ボタンをクリック

### 2. 画像スタイル変換

1. 元画像をアップロード
2. スタイルプリセット（アニメ風、水彩画風など）を選択、または「カスタム」で自由にプロンプト指定
3. 「変換強度」を調整（値が高いほど元画像から大きく変化します。目安: 0.4〜0.7）
4. 「変換」ボタンをクリック

### 3. 画像から動画生成 (I2V)

1. 元画像をアップロード
2. フレーム数・FPS・モーション量などを調整
3. 「動画生成」ボタンをクリック（他の機能より時間がかかります）

VRAMが不足する場合は、解像度・フレーム数・「デコードチャンクサイズ」を小さくしてください。

## モデルについて

初回生成時に、Hugging Face Hubから各モデル（数GB〜7GB程度）が自動的にダウンロードされ、
`~/.cache/huggingface` にキャッシュされます。2回目以降はダウンロード不要です。

| 機能 | 既定モデル | 備考 |
| --- | --- | --- |
| T2I | `stabilityai/stable-diffusion-xl-base-1.0` | 高品質。軽量版として `runwayml/stable-diffusion-v1-5` も選択可 |
| スタイル変換 | `Linaqruf/animagine-xl-3.1` | アニメ風特化。軽量版として `hakurei/waifu-diffusion` も選択可 |
| I2V | `stabilityai/stable-video-diffusion-img2vid-xt` | 軽量版として `stabilityai/stable-video-diffusion-img2vid` も選択可 |

いずれのモデルも各配布元のライセンス（多くはOpenRAIL系やCreativeML系）に従います。
生成物を商用利用・再配布する場合は、利用前に各モデルページのライセンスを必ずご確認ください。

## VRAM節約のヒント

- 画面上部の「省VRAMモード」をON（既定でON）にすると、使用していないモデル部分をCPUに退避し、VRAM使用量を抑えます（速度は低下）。
- 各機能を切り替えると、直前に使用していたモデルは自動的にアンロードされます（同時に複数モデルをVRAMに載せません）。
- T2I/スタイル変換は解像度を下げる（例: 768x768）、I2Vはフレーム数・解像度・デコードチャンクサイズを下げると省VRAMになります。
- それでも `CUDA out of memory` が出る場合は、軽量モデル（SD1.5系）に切り替えてください。

## トラブルシューティング

- **`torch.cuda.is_available()` が False / CPUで動いてしまう**: CUDA対応のPyTorchが入っていない可能性があります。手順3を確認してください。
- **モデルダウンロードが失敗する**: 一部モデルはHugging Faceの利用規約への同意が必要な場合があります。ブラウザでモデルページ（例: `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`）を開いて同意した上で、`huggingface-cli login` でログインしてから再実行してください。
- **動画生成でエラーになる**: `imageio-ffmpeg` が正しくインストールされているか確認してください（`pip install -r requirements.txt` で導入済みのはずです）。

## ディレクトリ構成

```
AIGen/
├── app.py                    # Gradio Web UI 本体
├── aigen/
│   ├── config.py             # モデル選択肢・既定設定
│   ├── utils.py               # デバイス判定・メモリ最適化・共通処理
│   ├── t2i.py                  # T2I (Text-to-Image) パイプライン
│   ├── style_transfer.py       # 画像スタイル変換 (img2img) パイプライン
│   └── i2v.py                    # I2V (Image-to-Video) パイプライン
├── outputs/                   # 生成された画像・動画の保存先
├── requirements.txt
└── README.md
```
