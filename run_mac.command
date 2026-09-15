#!/bin/bash
# AIGenをダブルクリックで起動するためのMac用ランチャー。
# 初回は仮想環境の作成と依存関係のインストールを自動で行うため時間がかかります。

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[AIGen] 初回セットアップ: 仮想環境を作成しています..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[AIGen] 依存関係を確認しています（初回はダウンロードに時間がかかります）..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "[AIGen] 起動しています。しばらくしたらブラウザが自動で開きます..."
( sleep 5 && open "http://127.0.0.1:7860" ) &

python app.py
