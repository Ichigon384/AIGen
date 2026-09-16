"""モデル選択肢や既定設定をまとめた設定モジュール。"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AppConfig:
    low_vram: bool = True
    output_dir: str = "outputs"


DEFAULT_CONFIG = AppConfig()

# テキストから画像生成 (T2I) で選べるモデル
# 注: runwayml/stable-diffusion-v1-5 はRunway社によりHugging Face上から削除されたため、
# コミュニティが管理する後継の公式ミラー組織を使用する。
T2I_MODEL_CHOICES: dict[str, str] = {
    "SDXL Base 1.0 (高品質・標準、VRAM目安8GB+)": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 1.5 (軽量・高速、VRAM目安4GB+)": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}

# 画像スタイル変換 (img2img) で選べるモデル
# 注: Animagine XL 3.1は開発が Linaqruf 個人アカウントから cagliostrolab
# 組織アカウントに移管されており、正しいリポジトリIDは cagliostrolab/animagine-xl-3.1。
STYLE_MODEL_CHOICES: dict[str, str] = {
    "Animagine XL 3.1 (アニメ風・SDXLベース)": "cagliostrolab/animagine-xl-3.1",
    "Waifu Diffusion 1.5 (アニメ風・SD1.5ベース、軽量)": "hakurei/waifu-diffusion",
}

# 画像から動画生成 (I2V) で選べるモデル
I2V_MODEL_CHOICES: dict[str, str] = {
    "Stable Video Diffusion XT (最大25フレーム)": "stabilityai/stable-video-diffusion-img2vid-xt",
    "Stable Video Diffusion (最大14フレーム・軽量)": "stabilityai/stable-video-diffusion-img2vid",
}
