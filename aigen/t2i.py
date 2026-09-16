"""テキストから画像生成 (T2I) を行うパイプラインのラッパー。"""

from __future__ import annotations

from PIL import Image

from .utils import (
    free_memory,
    get_device,
    get_torch_dtype,
    make_generator,
    optimize_pipeline,
)


class TextToImageGenerator:
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
        pipeline_cls = (
            StableDiffusionXLPipeline
            if "xl" in model_id.lower()
            else StableDiffusionPipeline
        )

        pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            # 本ツールが使うモデルはすべて公開モデルのため認証不要。
            # ローカルに無効/期限切れのHFトークンが保存されていると公開モデルの
            # 取得まで401エラーになることがあるため、明示的に未認証で取得する。
            token=False,
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
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[Image.Image, int]:
        self.load(model_id)
        device = get_device()
        generator, used_seed = make_generator(seed, device)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0], used_seed
