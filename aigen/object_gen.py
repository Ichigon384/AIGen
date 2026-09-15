"""動画編集などで素材として使う「オブジェクト画像」を生成するモジュール。

風景など背景込みの画像ではなく、単一の物体・キャラクターだけを描かせ、
必要に応じて背景を除去して透過PNG（RGBA）として出力する。
画像生成自体はT2I (TextToImageGenerator) を流用し、プロンプトの組み立てと
背景透過処理のみをここで担当する。
"""

from __future__ import annotations

from PIL import Image

# オブジェクト生成用のスタイルプリセット: プリセット名 -> プロンプトに追加する英語の説明
OBJECT_STYLE_PRESETS: dict[str, str] = {
    "アニメ風": (
        "anime style, cel shading, clean bold line art, flat colors, "
        "single isolated object, product shot, plain white background, "
        "no shadow, no scenery, best quality, masterpiece"
    ),
    "フラットベクター風": (
        "flat vector illustration, flat design, simple geometric shapes, "
        "bold outlines, minimal shading, single isolated object, "
        "plain white background, no shadow, no scenery, best quality"
    ),
    "カスタム（追加プロンプトのみ使用）": (
        "single isolated object, plain white background, no shadow, no scenery"
    ),
}

# 背景・複数オブジェクト・写真的な要素を避けるための既定ネガティブプロンプト
OBJECT_NEGATIVE_DEFAULT = (
    "landscape, background, scenery, room, multiple objects, duplicate, "
    "text, watermark, signature, border, frame, photo, realistic, "
    "low quality, worst quality, blurry, deformed, cropped"
)


def build_object_prompt(subject_prompt: str, style_preset: str, extra_prompt: str) -> str:
    """被写体・スタイルプリセット・追加プロンプトを1本のプロンプトに組み立てる。"""
    preset_prompt = OBJECT_STYLE_PRESETS.get(style_preset, "")
    parts = [subject_prompt.strip(), preset_prompt, extra_prompt.strip()]
    prompt = ", ".join(p for p in parts if p)
    if not prompt:
        raise ValueError("生成したいオブジェクトの説明を入力してください。")
    return prompt


def remove_background(image: Image.Image) -> Image.Image:
    """rembgを使って背景を除去し、アルファチャンネル付きのRGBA画像を返す。

    rembgは初回呼び出し時に背景除去モデル（U2Net、約176MB）を自動ダウンロードし、
    ローカルにキャッシュする。ダウンロード先: ~/.u2net
    """
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "背景透過には rembg が必要です。`pip install -r requirements.txt` を実行してください。"
        ) from exc

    return remove(image)
