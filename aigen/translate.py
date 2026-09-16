"""日本語プロンプトを画像生成モデルが理解しやすい英語に自動翻訳するモジュール。

Stable Diffusion系モデルのテキストエンコーダ(CLIP)はほぼ英語の画像キャプションのみで
学習されており、日本語をそのまま渡すとトークンとして正しく理解されない。
ユーザーが日本語でプロンプトを書けるように、ローカルで動作する翻訳モデル
(Helsinki-NLP/opus-mt-ja-en)を使い、生成前に自動で英語へ翻訳する。
APIキーは不要で、初回のみモデル（数百MB）をダウンロードしローカルにキャッシュする。
"""

from __future__ import annotations

import re

# ひらがな・カタカナ・漢字(CJK統合漢字)・全角記号の範囲
_JAPANESE_PATTERN = re.compile(r"[぀-ゟ゠-ヿ一-鿿＀-￯]")

_translator = None


def contains_japanese(text: str) -> bool:
    """テキストに日本語（ひらがな・カタカナ・漢字・全角記号）が含まれるか判定する。"""
    return bool(_JAPANESE_PATTERN.search(text))


def _get_translator():
    global _translator
    if _translator is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "日本語プロンプトの自動翻訳には transformers が必要です。"
                "`pip install -r requirements.txt` を実行してください。"
            ) from exc

        try:
            # 翻訳モデルは軽量(数百MB)なため、画像生成モデルのVRAM/統合メモリ管理とは
            # 独立させ、常にCPUで動作させる(画像生成用GPU/MPSのメモリを消費しない)。
            _translator = pipeline(
                "translation", model="Helsinki-NLP/opus-mt-ja-en", device=-1
            )
        except Exception as exc:
            raise RuntimeError(
                "翻訳モデル (Helsinki-NLP/opus-mt-ja-en) の読み込みに失敗しました。"
                "`pip install -r requirements.txt` で sentencepiece が導入されているか、"
                "初回ダウンロードに必要なインターネット接続があるかを確認してください。"
            ) from exc

    return _translator


def translate_to_english(text: str) -> str:
    """日本語が含まれていれば英訳して返す。日本語が含まれなければそのまま返す。

    既に英語のテキスト（既定のネガティブプロンプト等）に対しては翻訳モデルを
    呼び出さないため、余計な遅延やモデルダウンロードは発生しない。
    """
    stripped = text.strip()
    if not stripped or not contains_japanese(stripped):
        return stripped

    translator = _get_translator()
    result = translator(stripped, max_length=512)
    return result[0]["translation_text"].strip()
