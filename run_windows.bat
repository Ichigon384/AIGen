@echo off
REM AIGenをダブルクリックで起動するためのWindows用ランチャー。
REM 初回は仮想環境の作成と依存関係のインストールを自動で行うため時間がかかります。

cd /d "%~dp0"

if not exist ".venv" (
    echo [AIGen] 初回セットアップ: 仮想環境を作成しています...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [AIGen] 依存関係を確認しています（初回はダウンロードに時間がかかります）...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo [AIGen] 起動しています。しばらくしたらブラウザが自動で開きます...
start "" cmd /c "timeout /t 5 >nul && start http://127.0.0.1:7860"

python app.py
pause
