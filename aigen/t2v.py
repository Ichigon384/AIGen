"""テキストから動画生成 (T2V) を行うパイプラインのラッパー (ModelScope T2V / Zeroscope)。"""

from __future__ import annotations

from .utils import (
    ensure_output_dir,
    free_memory,
    get_device,
    get_torch_dtype,
    make_generator,
    optimize_pipeline,
    timestamped_filename,
)


class TextToVideoGenerator:
    def __init__(self, low_vram: bool = True, output_dir: str = "outputs"):
        self.low_vram = low_vram
        self.output_dir = output_dir
        self.model_id: str | None = None
        self.pipe = None

    def load(self, model_id: str) -> None:
        if self.model_id == model_id and self.pipe is not None:
            return
        self.unload()

        from diffusers import TextToVideoSDPipeline

        device = get_device()
        dtype = get_torch_dtype(device)

        pipe = TextToVideoSDPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # 本ツールが使うモデルは公開モデルのため認証不要。ローカルに無効/期限切れの
            # HFトークンが保存されていても影響を受けないよう明示的に未認証で取得する。
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
        num_frames: int,
        fps: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[str, int]:
        from diffusers.utils import export_to_video

        self.load(model_id)
        device = get_device()
        generator, used_seed = make_generator(seed, device)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        frames = result.frames[0]

        out_dir = ensure_output_dir(self.output_dir)
        out_path = out_dir / timestamped_filename("t2v", "mp4")
        export_to_video(frames, str(out_path), fps=fps)

        return str(out_path), used_seed
