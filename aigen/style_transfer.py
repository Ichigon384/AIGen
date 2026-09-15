"""画像のジャンル・スタイル変換 (img2img) を行うパイプラインのラッパー。"""

from __future__ import annotations

from PIL import Image

from .utils import (
    free_memory,
    get_device,
    get_torch_dtype,
    make_generator,
    optimize_pipeline,
)

# スタイルプリセット: プリセット名 -> プロンプトに追加する英語の説明
STYLE_PRESETS: dict[str, str] = {
    "アニメ風": "anime style, cel shading, vibrant colors, clean line art, masterpiece, best quality",
    "水彩画風": "watercolor painting, soft brush strokes, pastel colors, paper texture",
    "油絵風": "oil painting, thick brush strokes, canvas texture, rich colors, impressionist style",
    "サイバーパンク風": "cyberpunk style, neon lights, futuristic city, high contrast, cinematic lighting",
    "ピクセルアート風": "pixel art, 8-bit style, retro game graphics, limited color palette",
    "カスタム（追加プロンプトのみ使用）": "",
}


class StyleTransferGenerator:
    def __init__(self, low_vram: bool = True):
        self.low_vram = low_vram
        self.model_id: str | None = None
        self.pipe = None

    def load(self, model_id: str) -> None:
        if self.model_id == model_id and self.pipe is not None:
            return
        self.unload()

        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
        )

        device = get_device()
        dtype = get_torch_dtype(device)
        pipeline_cls = (
            StableDiffusionXLImg2ImgPipeline
            if "xl" in model_id.lower()
            else StableDiffusionImg2ImgPipeline
        )

        pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
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
        image: Image.Image,
        style_preset: str,
        extra_prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> tuple[Image.Image, int]:
        self.load(model_id)
        device = get_device()
        generator, used_seed = make_generator(seed, device)

        preset_prompt = STYLE_PRESETS.get(style_preset, "")
        prompt = ", ".join(p for p in [preset_prompt, extra_prompt.strip()] if p)
        if not prompt:
            prompt = "high quality, masterpiece"

        source_image = image.convert("RGB")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=source_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0], used_seed
