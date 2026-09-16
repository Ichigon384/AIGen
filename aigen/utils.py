"""パイプライン共通のユーティリティ（デバイス判定、メモリ最適化、シード管理など）。"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from PIL import Image


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3/M4) Mac
        return "mps"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    # MPS はfloat16で真っ黒な画像が出るなど不安定な場合があるため、
    # CUDA以外は速度より安定性を優先してfloat32を使う。
    return torch.float16 if device == "cuda" else torch.float32


def optimize_pipeline(pipe, low_vram: bool = True):
    """VRAM/メモリ使用量を抑えるための最適化をパイプラインに適用する。"""
    device = get_device()

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass

        if low_vram:
            # GPU/CPU間でモジュールを自動的にオフロードし、VRAMを節約する
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers未インストール、または非対応環境の場合は無視する
            pass

        return pipe

    # MPS (Mac) / CPU: model_cpu_offloadとxformersはCUDA専用のため使わない。
    pipe.to(device)

    if low_vram:
        # attention/vae slicingは統合メモリの消費を抑えるが、MPSでは処理が
        # 細切れになることでかえって大幅に遅くなる場合があるため、
        # 統合メモリに余裕がある場合は省メモリモードをOFFにして無効化できるようにする。
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
            pass

    return pipe


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def make_generator(seed: int | None, device: str) -> tuple[torch.Generator, int]:
    """指定シード（負値ならランダム）でGeneratorを作成し、使用したシード値も返す。"""
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % (2**31 - 1)
    generator_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return generator, seed


def resize_to_valid_dimensions(
    image: Image.Image, max_size: int = 1024, multiple: int = 8
) -> Image.Image:
    """img2img/VAEが要求する8の倍数のサイズに、アスペクト比を保ったままリサイズする。

    スマートフォン等で撮影した画像はサイズが8の倍数でないことが多く、
    そのままパイプラインに渡すとVAEの形状不一致エラーになるため、
    アップロード画像は生成前に必ずこの関数を通す。
    """
    width, height = image.size
    scale = min(1.0, max_size / max(width, height))
    new_width = max(multiple, round(width * scale / multiple) * multiple)
    new_height = max(multiple, round(height * scale / multiple) * multiple)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), Image.LANCZOS)


def ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
