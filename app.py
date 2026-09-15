"""AIGen: ローカルで動くAI画像・動画生成ツール
(T2I / オブジェクト生成 / 画像スタイル変換 / I2V)。

起動方法:
    python app.py
"""

from __future__ import annotations

import os

# Mac (MPS) では一部の演算が未実装のことがあるため、CPUへの自動フォールバックを
# 有効化する。torch/diffusersをインポートする前に設定する必要がある。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pillow_heif

# iPhone等で撮影されたHEIC/HEIF画像をPillow(PIL.Image.open)で直接開けるようにする。
# GradioのImageコンポーネントは内部でPIL.Image.openを使うため、これを登録しないと
# HEICファイルのアップロード時に UnidentifiedImageError になる。
pillow_heif.register_heif_opener()

import gradio as gr

from aigen.config import I2V_MODEL_CHOICES, STYLE_MODEL_CHOICES, T2I_MODEL_CHOICES
from aigen.i2v import ImageToVideoGenerator
from aigen.object_gen import (
    OBJECT_NEGATIVE_DEFAULT,
    OBJECT_STYLE_PRESETS,
    build_object_prompt,
    remove_background,
)
from aigen.style_transfer import STYLE_PRESETS, StyleTransferGenerator
from aigen.t2i import TextToImageGenerator
from aigen.utils import get_device

LOW_VRAM_DEFAULT = True
OUTPUT_DIR = "outputs"

t2i_generator = TextToImageGenerator(low_vram=LOW_VRAM_DEFAULT)
style_generator = StyleTransferGenerator(low_vram=LOW_VRAM_DEFAULT)
i2v_generator = ImageToVideoGenerator(low_vram=LOW_VRAM_DEFAULT, output_dir=OUTPUT_DIR)


def _unload_others(keep: str) -> None:
    """VRAM節約のため、使用中でないパイプラインをアンロードする。"""
    if keep != "t2i":
        t2i_generator.unload()
    if keep != "style":
        style_generator.unload()
    if keep != "i2v":
        i2v_generator.unload()


def run_t2i(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    steps: float,
    guidance_scale: float,
    width: float,
    height: float,
    seed: float,
    low_vram: bool,
):
    if not prompt or not prompt.strip():
        raise gr.Error("プロンプトを入力してください。")

    t2i_generator.low_vram = low_vram
    _unload_others(keep="t2i")

    model_id = T2I_MODEL_CHOICES[model_name]
    image, used_seed = t2i_generator.generate(
        model_id,
        prompt,
        negative_prompt,
        int(steps),
        float(guidance_scale),
        int(width),
        int(height),
        int(seed),
    )
    return image, f"使用シード値: {used_seed}"


def run_object_gen(
    model_name: str,
    subject_prompt: str,
    style_preset: str,
    extra_prompt: str,
    negative_prompt: str,
    transparent_bg: bool,
    steps: float,
    guidance_scale: float,
    width: float,
    height: float,
    seed: float,
    low_vram: bool,
):
    if not subject_prompt or not subject_prompt.strip():
        raise gr.Error("生成したいオブジェクトの説明を入力してください。")

    # オブジェクト生成はT2Iと同じ画像生成パイプラインを流用する
    t2i_generator.low_vram = low_vram
    _unload_others(keep="t2i")

    model_id = T2I_MODEL_CHOICES[model_name]
    prompt = build_object_prompt(subject_prompt, style_preset, extra_prompt)
    image, used_seed = t2i_generator.generate(
        model_id,
        prompt,
        negative_prompt,
        int(steps),
        float(guidance_scale),
        int(width),
        int(height),
        int(seed),
    )

    log = f"使用シード値: {used_seed}"
    if transparent_bg:
        try:
            image = remove_background(image)
            log += " / 背景透過: 完了"
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc

    return image, log


def run_style_transfer(
    model_name: str,
    image,
    style_preset: str,
    extra_prompt: str,
    negative_prompt: str,
    strength: float,
    steps: float,
    guidance_scale: float,
    seed: float,
    low_vram: bool,
):
    if image is None:
        raise gr.Error("元画像をアップロードしてください。")

    style_generator.low_vram = low_vram
    _unload_others(keep="style")

    model_id = STYLE_MODEL_CHOICES[model_name]
    result, used_seed = style_generator.generate(
        model_id,
        image,
        style_preset,
        extra_prompt,
        negative_prompt,
        float(strength),
        int(steps),
        float(guidance_scale),
        int(seed),
    )
    return result, f"使用シード値: {used_seed}"


def run_i2v(
    model_name: str,
    image,
    num_frames: float,
    fps: float,
    motion_bucket_id: float,
    noise_aug_strength: float,
    decode_chunk_size: float,
    width: float,
    height: float,
    seed: float,
    low_vram: bool,
):
    if image is None:
        raise gr.Error("元画像をアップロードしてください。")

    i2v_generator.low_vram = low_vram
    _unload_others(keep="i2v")

    model_id = I2V_MODEL_CHOICES[model_name]
    video_path, used_seed = i2v_generator.generate(
        model_id,
        image,
        int(num_frames),
        int(fps),
        int(motion_bucket_id),
        float(noise_aug_strength),
        int(decode_chunk_size),
        int(width),
        int(height),
        int(seed),
    )
    return video_path, f"使用シード値: {used_seed}"


_DEVICE_LABELS = {
    "cuda": "cuda (NVIDIA GPU)",
    "mps": "mps (Apple Silicon)",
    "cpu": "cpu (GPUなし・低速)",
}

with gr.Blocks(title="AIGen - ローカルAI画像・動画生成ツール") as demo:
    gr.Markdown(
        "# AIGen\n"
        "テキストから画像生成 (T2I) / 動画編集素材向けオブジェクト生成 / "
        "画像のスタイル変換 (アニメ風など) / 画像から動画生成 (I2V) を"
        "ローカル環境で実行します。\n\n"
        f"検出デバイス: **{_DEVICE_LABELS.get(get_device(), get_device())}**"
    )

    low_vram_checkbox = gr.Checkbox(
        value=LOW_VRAM_DEFAULT,
        label="省メモリモード (VRAM/統合メモリが少ない場合はONを推奨。生成速度は低下します)",
    )

    with gr.Tabs():
        with gr.Tab("テキストから画像生成 (T2I)"):
            with gr.Row():
                with gr.Column():
                    t2i_model = gr.Dropdown(
                        list(T2I_MODEL_CHOICES.keys()),
                        value=list(T2I_MODEL_CHOICES.keys())[0],
                        label="モデル",
                    )
                    t2i_prompt = gr.Textbox(
                        label="プロンプト",
                        lines=3,
                        placeholder="例: a girl standing in a cherry blossom garden, masterpiece, best quality",
                    )
                    t2i_negative = gr.Textbox(
                        label="ネガティブプロンプト",
                        lines=2,
                        value="low quality, worst quality, blurry, deformed",
                    )
                    with gr.Row():
                        t2i_width = gr.Slider(256, 1536, value=1024, step=64, label="幅")
                        t2i_height = gr.Slider(256, 1536, value=1024, step=64, label="高さ")
                    with gr.Row():
                        t2i_steps = gr.Slider(1, 100, value=30, step=1, label="ステップ数")
                        t2i_cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFGスケール")
                    t2i_seed = gr.Number(value=-1, label="シード値 (-1でランダム)", precision=0)
                    t2i_button = gr.Button("生成", variant="primary")
                with gr.Column():
                    t2i_output = gr.Image(label="生成結果")
                    t2i_log = gr.Textbox(label="ログ", interactive=False)

            t2i_button.click(
                run_t2i,
                inputs=[
                    t2i_model,
                    t2i_prompt,
                    t2i_negative,
                    t2i_steps,
                    t2i_cfg,
                    t2i_width,
                    t2i_height,
                    t2i_seed,
                    low_vram_checkbox,
                ],
                outputs=[t2i_output, t2i_log],
            )

        with gr.Tab("オブジェクト生成 (動画編集素材向け)"):
            gr.Markdown(
                "背景のない単一のオブジェクト・キャラクター画像を生成します。"
                "動画編集ソフトに素材として読み込むことを想定し、アニメ風・ベクター風のスタイルと"
                "背景透過（PNG/アルファチャンネル）に対応しています。"
            )
            with gr.Row():
                with gr.Column():
                    obj_model = gr.Dropdown(
                        list(T2I_MODEL_CHOICES.keys()),
                        value=list(T2I_MODEL_CHOICES.keys())[0],
                        label="モデル",
                    )
                    obj_subject_prompt = gr.Textbox(
                        label="生成したいオブジェクト",
                        lines=2,
                        placeholder="例: a red apple / a robot cat / a treasure chest",
                    )
                    obj_style_preset = gr.Dropdown(
                        list(OBJECT_STYLE_PRESETS.keys()), value="アニメ風", label="スタイルプリセット"
                    )
                    obj_extra_prompt = gr.Textbox(label="追加プロンプト（任意）", lines=2)
                    obj_negative = gr.Textbox(
                        label="ネガティブプロンプト",
                        lines=2,
                        value=OBJECT_NEGATIVE_DEFAULT,
                    )
                    obj_transparent = gr.Checkbox(value=True, label="背景を透過する (PNG/アルファチャンネル)")
                    with gr.Row():
                        obj_width = gr.Slider(256, 1536, value=1024, step=64, label="幅")
                        obj_height = gr.Slider(256, 1536, value=1024, step=64, label="高さ")
                    with gr.Row():
                        obj_steps = gr.Slider(1, 100, value=30, step=1, label="ステップ数")
                        obj_cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFGスケール")
                    obj_seed = gr.Number(value=-1, label="シード値 (-1でランダム)", precision=0)
                    obj_button = gr.Button("生成", variant="primary")
                with gr.Column():
                    obj_output = gr.Image(label="生成結果", image_mode="RGBA")
                    obj_log = gr.Textbox(label="ログ", interactive=False)

            obj_button.click(
                run_object_gen,
                inputs=[
                    obj_model,
                    obj_subject_prompt,
                    obj_style_preset,
                    obj_extra_prompt,
                    obj_negative,
                    obj_transparent,
                    obj_steps,
                    obj_cfg,
                    obj_width,
                    obj_height,
                    obj_seed,
                    low_vram_checkbox,
                ],
                outputs=[obj_output, obj_log],
            )

        with gr.Tab("画像スタイル変換"):
            with gr.Row():
                with gr.Column():
                    style_model = gr.Dropdown(
                        list(STYLE_MODEL_CHOICES.keys()),
                        value=list(STYLE_MODEL_CHOICES.keys())[0],
                        label="モデル",
                    )
                    style_input_image = gr.Image(label="元画像", type="pil")
                    style_preset = gr.Dropdown(
                        list(STYLE_PRESETS.keys()), value="アニメ風", label="スタイルプリセット"
                    )
                    style_extra_prompt = gr.Textbox(label="追加プロンプト（任意）", lines=2)
                    style_negative = gr.Textbox(
                        label="ネガティブプロンプト",
                        lines=2,
                        value="low quality, worst quality, blurry, deformed",
                    )
                    style_strength = gr.Slider(
                        0.1, 1.0, value=0.6, step=0.05,
                        label="変換強度 (高いほど元画像から離れたスタイルになります)",
                    )
                    with gr.Row():
                        style_steps = gr.Slider(1, 100, value=30, step=1, label="ステップ数")
                        style_cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFGスケール")
                    style_seed = gr.Number(value=-1, label="シード値 (-1でランダム)", precision=0)
                    style_button = gr.Button("変換", variant="primary")
                with gr.Column():
                    style_output = gr.Image(label="変換結果")
                    style_log = gr.Textbox(label="ログ", interactive=False)

            style_button.click(
                run_style_transfer,
                inputs=[
                    style_model,
                    style_input_image,
                    style_preset,
                    style_extra_prompt,
                    style_negative,
                    style_strength,
                    style_steps,
                    style_cfg,
                    style_seed,
                    low_vram_checkbox,
                ],
                outputs=[style_output, style_log],
            )

        with gr.Tab("画像から動画生成 (I2V)"):
            with gr.Row():
                with gr.Column():
                    i2v_model = gr.Dropdown(
                        list(I2V_MODEL_CHOICES.keys()),
                        value=list(I2V_MODEL_CHOICES.keys())[0],
                        label="モデル",
                    )
                    i2v_input_image = gr.Image(label="元画像", type="pil")
                    with gr.Row():
                        i2v_width = gr.Slider(256, 1024, value=576, step=64, label="幅")
                        i2v_height = gr.Slider(256, 1024, value=320, step=64, label="高さ")
                    with gr.Row():
                        i2v_frames = gr.Slider(8, 25, value=14, step=1, label="フレーム数")
                        i2v_fps = gr.Slider(4, 30, value=7, step=1, label="FPS")
                    with gr.Row():
                        i2v_motion = gr.Slider(1, 255, value=127, step=1, label="モーション量 (motion_bucket_id)")
                        i2v_noise = gr.Slider(0.0, 1.0, value=0.02, step=0.01, label="ノイズ付加強度")
                    i2v_chunk = gr.Slider(1, 25, value=4, step=1, label="デコードチャンクサイズ (小さいほど省VRAM)")
                    i2v_seed = gr.Number(value=-1, label="シード値 (-1でランダム)", precision=0)
                    i2v_button = gr.Button("動画生成", variant="primary")
                with gr.Column():
                    i2v_output = gr.Video(label="生成結果")
                    i2v_log = gr.Textbox(label="ログ", interactive=False)

            i2v_button.click(
                run_i2v,
                inputs=[
                    i2v_model,
                    i2v_input_image,
                    i2v_frames,
                    i2v_fps,
                    i2v_motion,
                    i2v_noise,
                    i2v_chunk,
                    i2v_width,
                    i2v_height,
                    i2v_seed,
                    low_vram_checkbox,
                ],
                outputs=[i2v_output, i2v_log],
            )

    gr.Markdown(
        "---\n"
        "初回生成時は各モデルのダウンロード（数GB）が行われるため時間がかかります。\n"
        "メモリ不足エラー (CUDA/MPS out of memory) が出る場合は「省メモリモード」をON、"
        "画像サイズ・フレーム数・デコードチャンクサイズを小さくしてください。"
    )

if __name__ == "__main__":
    demo.queue().launch()
