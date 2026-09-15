"""パイプライン共通のユーティリティ（デバイス判定、メモリ最適化、シード管理など）。"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def optimize_pipeline(pipe, low_vram: bool = True):
    """VRAM使用量を抑えるための最適化をパイプラインに適用する。"""
    device = get_device()
    if device != "cuda":
        pipe.to(device)
        return pipe

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


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_generator(seed: int | None, device: str) -> tuple[torch.Generator, int]:
    """指定シード（負値ならランダム）でGeneratorを作成し、使用したシード値も返す。"""
    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % (2**31 - 1)
    generator_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return generator, seed


def ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
