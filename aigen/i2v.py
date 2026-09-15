"""画像から動画生成 (I2V) を行うパイプラインのラッパー (Stable Video Diffusion)。"""

from __future__ import annotations

from PIL import Image

from .utils import (
    ensure_output_dir,
    free_memory,
    get_device,
    get_torch_dtype,
    make_generator,
    optimize_pipeline,
    timestamped_filename,
)


class ImageToVideoGenerator:
    def __init__(self, low_vram: bool = True, output_dir: str = "outputs"):
        self.low_vram = low_vram
        self.output_dir = output_dir
        self.model_id: str | None = None
        self.pipe = None

    def load(self, model_id: str) -> None:
        if self.model_id == model_id and self.pipe is not None:
            return
        self.unload()

        import torch
        from diffusers import StableVideoDiffusionPipeline

        device = get_device()
        dtype = get_torch_dtype(device)

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
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
        num_frames: int,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        decode_chunk_size: int,
        width: int,
        height: int,
        seed: int,
    ) -> tuple[str, int]:
        from diffusers.utils import export_to_video

        self.load(model_id)
        device = get_device()
        generator, used_seed = make_generator(seed, device)

        source_image = image.convert("RGB").resize((width, height))

        result = self.pipe(
            image=source_image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator,
        )
        frames = result.frames[0]

        out_dir = ensure_output_dir(self.output_dir)
        out_path = out_dir / timestamped_filename("i2v", "mp4")
        export_to_video(frames, str(out_path), fps=fps)

        return str(out_path), used_seed
