"""MVなどで使う「同じキャラクターを保ったまま、ポーズやカメラアングルだけを変える」
生成を行うモジュール。

キャラクターの参照画像を1枚アップロードすると、IP-Adapter FaceID (h94/IP-Adapter-FaceID) が
insightfaceで抽出した「顔の特徴量（埋め込みベクトル）」を手がかりとして生成に反映しつつ、
プロンプトで指定したポーズ・カメラアングルに従った新しい画像を作る。
通常のIP-Adapter（画像全体の雰囲気を見るだけ）より、顔そのものの同一性を強く保てる。
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

_IP_ADAPTER_FACEID_REPO = "h94/IP-Adapter-FaceID"

_face_app = None


def _get_face_app():
    """insightfaceの顔検出・埋め込み抽出アプリを遅延初期化する（常にCPUで動作）。

    顔検出は画像生成本体のGPU/MPS処理とは独立した軽い処理のため、VRAM/統合メモリの
    管理対象には含めず、常にCPU上のonnxruntimeで実行する(rembgと同様の考え方)。
    初回呼び出し時にモデル(buffalo_l、約300MB)を自動ダウンロードする。
    """
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "キャラクター一貫生成には insightface が必要です。"
                "`pip install -r requirements.txt` を実行してください。"
            ) from exc

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app

    return _face_app


def _extract_face_embedding(reference_image: Image.Image):
    """参照画像から最も大きく写っている顔の埋め込みベクトルを抽出する。"""
    import cv2
    import numpy as np

    app = _get_face_app()
    bgr_image = cv2.cvtColor(np.array(reference_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr_image)

    if not faces:
        raise RuntimeError(
            "参照画像から顔を検出できませんでした。顔がはっきり正面付近を向いて"
            "写っている画像を使用してください。"
        )

    # 複数の顔が写っている場合は、最も大きく写っている顔を採用する
    largest_face = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return largest_face.normed_embedding


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

        # 顔の埋め込みベクトルからキャラクターの同一性を反映させるため
        # IP-Adapter FaceIDを読み込む (SDXL系とSD1.5系で重みファイルが異なる)。
        # FaceIDはCLIP画像エンコーダを使わないため image_encoder_folder=None を指定する。
        weight_name = "ip-adapter-faceid_sdxl.bin" if is_xl else "ip-adapter-faceid_sd15.bin"
        pipe.load_ip_adapter(
            _IP_ADAPTER_FACEID_REPO,
            subfolder=None,
            weight_name=weight_name,
            image_encoder_folder=None,
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
        import torch

        self.load(model_id)
        device = get_device()
        dtype = get_torch_dtype(device)
        generator, used_seed = make_generator(seed, device)

        # IP-Adapterの影響度。高いほど参照画像(顔)に似るがポーズ指定の自由度が下がる
        self.pipe.set_ip_adapter_scale(identity_strength)

        face_embedding = _extract_face_embedding(reference_image)
        # diffusers公式のIP-Adapter FaceID例と同じ手順でテンソル形状を組み立てる:
        # (512,) -> (1,512) -> [tensor]をstackして(1,1,512) -> unsqueezeで(1,1,1,512)
        positive_embeds = torch.from_numpy(face_embedding).unsqueeze(0)
        positive_embeds = torch.stack([positive_embeds], dim=0).unsqueeze(0)
        negative_embeds = torch.zeros_like(positive_embeds)
        # classifier-free guidance用に [negative, positive] の順で結合する -> (2,1,1,512)
        id_embeds = torch.cat([negative_embeds, positive_embeds]).to(device=device, dtype=dtype)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            ip_adapter_image_embeds=[id_embeds],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0], used_seed
