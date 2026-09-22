"""MVなどで使う「同じキャラクターを保ったまま、ポーズやカメラアングルだけを変える」
生成を行うモジュール。

キャラクターの参照画像を1枚アップロードすると、IP-Adapter (h94/IP-Adapter) が
その画像の見た目の特徴（顔・髪型・服装の色柄など）を手がかりとして生成に反映しつつ、
プロンプトで指定したポーズ・カメラアングルに従った新しい画像を作る。
学習（ファインチューニング/LoRA作成）は不要で、参照画像1枚とプロンプトだけで使える。

背景透過には aigen.object_gen の remove_background をそのまま利用する。
"""

from __future__ import annotations

from PIL import Image

from .utils import (
    free_memory,
    get_device,
    get_torch_dtype,
    make_generator,
    optimize_pipeline,
    resize_to_valid_dimensions,
)

# キャラクター生成用のスタイルプリセット: プリセット名 -> プロンプトに追加する英語の説明
CHARACTER_STYLE_PRESETS: dict[str, str] = {
    "アニメ風": (
        "anime style, cel shading, clean line art, consistent character design, "
        "single character, isolated on background, plain white background, "
        "no shadow, no scenery, best quality, masterpiece"
    ),
    "フラットベクター風": (
        "flat vector illustration, flat design, bold outlines, minimal shading, "
        "consistent character design, single character, isolated on background, "
        "plain white background, no shadow, no scenery, best quality"
    ),
    "カスタム（追加プロンプトのみ使用）": (
        "consistent character design, single character, isolated on background, "
        "plain white background, no shadow, no scenery"
    ),
}

# 別人化・複数人・背景混入を避けるための既定ネガティブプロンプト
CHARACTER_NEGATIVE_DEFAULT = (
    "multiple characters, different character, extra person, twins, "
    "landscape, background, scenery, room, "
    "text, watermark, signature, border, frame, photo, realistic, "
    "low quality, worst quality, blurry, deformed, cropped, bad anatomy"
)

_IP_ADAPTER_REPO = "h94/IP-Adapter"


class CharacterGenerator:
    def __init__(self, low_vram: bool = True):
        self.low_vram = low_vram
        self.model_id: str | None = None
        self.pipe = None

    def load(self, model_id: str) -> None:
        if self.model_id == model_id and self.pipe is not None:
            return
        self.unload()

        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        device = get_device()
        dtype = get_torch_dtype(device)
        is_xl = "xl" in model_id.lower()
        pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

        pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            # 本ツールが使うモデルは公開モデルのため認証不要。ローカルに無効/期限切れの
            # HFトークンが保存されていても影響を受けないよう明示的に未認証で取得する。
            token=False,
        )

        # 参照画像からキャラクターの特徴を反映させるためIP-Adapterを読み込む
        # (SDXL系とSD1.5系で重みファイル・サブフォルダが異なる)
        if is_xl:
            pipe.load_ip_adapter(
                _IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
            )
        else:
            pipe.load_ip_adapter(
                _IP_ADAPTER_REPO, subfolder="models", weight_name="ip-adapter_sd15.bin"
            )

        self.pipe = optimize_pipeline(pipe, low_vram=self.low_vram)
        self.model_id = model_id

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.model_id = None
            free_memory()

    def generate(
        self,
        model_id: str,
        reference_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        identity_strength: float,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[Image.Image, int]:
        self.load(model_id)
        device = get_device()
        generator, used_seed = make_generator(seed, device)

        # IP-Adapterの影響度。高いほど参照画像に似るがポーズ指定の自由度が下がる
        self.pipe.set_ip_adapter_scale(identity_strength)

        max_size = 1024 if "xl" in model_id.lower() else 768
        ref_image = resize_to_valid_dimensions(reference_image.convert("RGB"), max_size=max_size)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            ip_adapter_image=ref_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0], used_seed
