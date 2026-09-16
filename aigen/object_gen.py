"""動画編集などで素材として使う「オブジェクト画像」を生成するモジュール。

風景など背景込みの画像ではなく、単一の物体・キャラクターだけを描かせ、
必要に応じて背景を除去して透過PNG（RGBA）として出力する。
画像生成自体はT2I (TextToImageGenerator) を流用し、プロンプトの組み立てと
背景透過処理のみをここで担当する。
"""

from __future__ import annotations

from PIL import Image

# オブジェクト生成用のスタイルプリセット: プリセット名 -> プロンプトに追加する英語の説明
# 「leaf」「flower」等の単語は学習データ上「繰り返しパターン(壁紙・テキスタイル)」画像に
# 強く結びついていることが多く、"single object"だけでは抑えきれないことがあるため、
# 構図を明示的に固定する語句（centered, one single object in frame, no pattern等）を
# 厚めに入れている。
OBJECT_STYLE_PRESETS: dict[str, str] = {
    "アニメ風": (
        "anime style, cel shading, clean bold line art, flat colors, "
        "one single object centered in frame, isolated on background, product shot, "
        "plain white background, no shadow, no scenery, best quality, masterpiece"
    ),
    "フラットベクター風": (
        "flat vector illustration, flat design, simple geometric shapes, "
        "bold outlines, minimal shading, one single object centered in frame, "
        "isolated on background, plain white background, no shadow, no scenery, best quality"
    ),
    "カスタム（追加プロンプトのみ使用）": (
        "one single object centered in frame, isolated on background, "
        "plain white background, no shadow, no scenery"
    ),
}

# 背景・複数オブジェクト・パターン化・写真的な要素を避けるための既定ネガティブプロンプト
OBJECT_NEGATIVE_DEFAULT = (
    "pattern, seamless pattern, repeating pattern, tile, tiling, wallpaper, textile, "
    "collage, grid, array, landscape, background, scenery, room, "
    "multiple objects, many objects, several objects, group, collection, duplicate, "
    "text, watermark, signature, border, frame, photo, realistic, "
    "low quality, worst quality, blurry, deformed, cropped"
)


def build_object_prompt(subject_prompt: str, style_preset: str, extra_prompt: str) -> str:
    """被写体・スタイルプリセット・追加プロンプトを1本のプロンプトに組み立てる。"""
    subject = subject_prompt.strip()
    if not subject:
        raise ValueError("生成したいオブジェクトの説明を入力してください。")

    # 被写体を単数形として明示することで、「leaf」のような単語が
    # 「leaves(複数)のパターン柄」として生成されるのを防ぐ。
    subject_singular = f"a single {subject}"

    preset_prompt = OBJECT_STYLE_PRESETS.get(style_preset, "")
    parts = [subject_singular, preset_prompt, extra_prompt.strip()]
    return ", ".join(p for p in parts if p)


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
