# AIGen

ローカル環境（自分のPC）で完全無料で動く、以下3機能をまとめたAI画像・動画生成ツールです。

- **T2I（テキストから画像生成）**: プロンプトから画像を生成
- **画像スタイル変換**: 実写・イラストなどの画像を「アニメ風」「水彩画風」などにジャンルチェンジ
- **I2V（画像から動画生成）**: 1枚の画像から短い動画を生成

ブラウザで動くWeb UI（[Gradio](https://www.gradio.app/)）で、3つの機能をタブ切り替えで使えます。
すべて [diffusers](https://github.com/huggingface/diffusers) 上のオープンソースモデルを使用しており、
APIキーは不要・生成コストもかかりません（電気代とダウンロード帯域を除く）。

## クイックスタート（ダブルクリックで起動）

初回セットアップ（仮想環境作成・依存関係インストール）とアプリ起動をまとめて行うランチャーを用意しています。
Python 3.10以上がインストール済みであれば、これだけで起動できます。

- **Mac**: `run_mac.command` をダブルクリック（初回のみ、Finderで右クリック→「開く」を選ぶ必要がある場合があります。Gatekeeperの警告が出た場合は「システム設定」→「プライバシーとセキュリティ」から許可してください）
- **Windows**: `run_windows.bat` をダブルクリック

数十秒〜数分後（初回はモデル/依存関係のダウンロードのためさらに時間がかかります）、自動的にブラウザで `http://127.0.0.1:7860` が開きます。終了する場合は、開いたターミナル/コマンドプロンプトのウィンドウで `Ctrl+C` を押してください。

うまく動かない場合は、下記の手動セットアップ手順を試してください。

## 必要環境

本ツールは以下のいずれかの環境で動作します。

- **Windows/Linux + NVIDIA GPU（VRAM 8GB以上を推奨）+ CUDA**（推奨・最速）
- **Mac（Apple Silicon: M1/M2/M3/M4、統合メモリ16GB以上を推奨）**
  - PyTorchのMPS（Metal）バックエンドを自動的に使用します。Intel Macは非対応です。
  - NVIDIA GPUよりは低速で、対応していない演算はCPUに自動フォールバックします。
  - SDXLやStable Video Diffusionは重いため、統合メモリ8GBのMacでは軽量モデル（SD1.5系）や低解像度設定を推奨します。
- **CPUのみ**（GPUなしでも動作しますが、画像1枚で数分〜数十分、動画はさらに長時間かかります）

共通の要件:

- Python 3.10 以上
- ディスク空き容量 **20GB以上**（モデルを複数ダウンロードするため）
- インターネット接続（初回のモデルダウンロード時のみ必要。以降はローカルキャッシュから読み込みます）

VRAM/メモリが少ない場合は「省メモリモード」をONにし、画像サイズやフレーム数を小さくすることで動作させられる場合があります。

## セットアップ

```bash
# 1. リポジトリを取得（このディレクトリで作業している場合は不要）
cd AIGen

# 2. 仮想環境を作成
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate

# 3. (Windows/Linux + NVIDIA GPUの場合のみ) CUDA対応版PyTorchを先にインストール
#    お使いのGPU/CUDAバージョンは https://pytorch.org/get-started/locally/ で確認できます
#    例 (CUDA 12.1): pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. 残りの依存関係をインストール
pip install -r requirements.txt
```

- **Mac（Apple Silicon）**: 手順3は不要です。手順4の `pip install -r requirements.txt` で入る通常版のtorchが、そのままMPSバックエンドに対応しています（macOS 12.3以降が必要）。
- **GPUなし/CPUのみ**: 同様に手順3は不要です（動作は低速です）。

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

VRAM/メモリが不足する場合は、解像度・フレーム数・「デコードチャンクサイズ」を小さくしてください。

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

## VRAM/メモリ節約のヒント

- 画面上部の「省メモリモード」をON（既定でON）にすると、NVIDIA GPU環境では使用していないモデル部分をCPUに退避してVRAM使用量を抑えます（速度は低下）。Mac(MPS)/CPU環境ではこのオフロードは行われませんが、attention/vae slicingは常時有効です。
- 各機能を切り替えると、直前に使用していたモデルは自動的にアンロードされます（同時に複数モデルをGPU/統合メモリに載せません）。
- T2I/スタイル変換は解像度を下げる（例: 768x768）、I2Vはフレーム数・解像度・デコードチャンクサイズを下げるとメモリ使用量が減ります。
- それでも `CUDA out of memory` / `MPS backend out of memory` が出る場合は、軽量モデル（SD1.5系）に切り替えてください。

## Macでの注意点

- SDXLやStable Video Diffusionは統合メモリを多く消費します。16GB以上のMacを推奨し、8GBのMacでは軽量モデル（SD1.5系のT2I・Waifu Diffusion）や低解像度（例: 512x512、I2Vは256x256〜384x384）から試してください。
- 初回起動時に自動的に `PYTORCH_ENABLE_MPS_FALLBACK=1` が設定され、MPSで未対応の演算はCPUに自動フォールバックします（該当箇所は多少遅くなりますが、エラーにはなりません）。
- NVIDIA GPUに比べて生成速度は遅く、特にI2V（動画生成）は数分以上かかることがあります。

## トラブルシューティング

- **GPU/MPSが使われずCPUで動いてしまう**: NVIDIA GPU環境ではCUDA対応のPyTorchが入っていない可能性があります（手順3を確認）。Macの場合はmacOS 12.3以降・Apple Siliconであることを確認してください（Intel Macは非対応）。
- **モデルダウンロードが失敗する**: 一部モデルはHugging Faceの利用規約への同意が必要な場合があります。ブラウザでモデルページ（例: `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt`）を開いて同意した上で、`huggingface-cli login` でログインしてから再実行してください。
- **動画生成でエラーになる**: `imageio-ffmpeg` が正しくインストールされているか確認してください（`pip install -r requirements.txt` で導入済みのはずです）。
- **Macで画像が真っ黒になる**: 稀にMPSのfloat16関連の不具合で発生することがあります。本ツールはMac/CPUでは既定でfloat32を使用しているため通常は問題ありませんが、発生する場合はPyTorchを最新版に更新してください。
- **`requirements.txt` が見つからない (`pip install -r requirements.txt` が失敗する)**: 意図せず入れ子のディレクトリに入っている可能性があります。macOSはデフォルトでファイル名の大文字・小文字を区別しないため、リポジトリ直下で `cd AIGen` すると、中の `aigen/`（Pythonパッケージ）フォルダに入ってしまうことがあります。`pwd` で現在地を確認し、`config.py` や `t2i.py` などが直接見えている場合は `cd ..` で一つ上に戻ってください。`app.py` と `requirements.txt` がある階層が正しいリポジトリ直下です。

## ディレクトリ構成

```
AIGen/
├── app.py                    # Gradio Web UI 本体
├── run_mac.command           # Mac用ワンクリック起動スクリプト
├── run_windows.bat           # Windows用ワンクリック起動スクリプト
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
