"""Local-only text-to-image app. Model files must already exist on disk."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

# Set before importing Hugging Face libraries: generation must never attempt a network call.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "stable-diffusion-v1-5")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_pipeline: StableDiffusionPipeline | None = None
_pipeline_lock = Lock()
_generation_lock = Lock()

STYLE_PRESETS = {
    "No extra style": "",
    "Photorealistic": "photorealistic, realistic lighting, detailed textures",
    "Digital art": "polished digital illustration, rich colors",
    "Watercolor": "watercolor painting, soft pigments, paper texture",
    "Anime": "anime illustration, expressive line art",
    "Pixel art": "pixel art, crisp pixels, limited color palette",
}


def get_pipeline() -> tuple[StableDiffusionPipeline, str]:
    global _pipeline
    if _pipeline is not None:
        return _pipeline, str(_pipeline._execution_device)

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline, str(_pipeline._execution_device)
        if not (MODEL_DIR / "model_index.json").is_file():
            raise gr.Error(
                f"Model not found at {MODEL_DIR}. First, connect to the internet and run "
                "scripts/download_model.py. Then you can disconnect and run this app offline."
            )

        device_options = []
        if torch.cuda.is_available():
            device_options.append(("cuda", torch.float16))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_options.append(("mps", torch.float16))
        device_options.append(("cpu", torch.float32))

        load_errors = []
        for device, dtype in device_options:
            candidate = None
            try:
                candidate = StableDiffusionPipeline.from_pretrained(
                    str(MODEL_DIR),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                candidate.to(device)
                if device in {"cpu", "mps"}:
                    candidate.enable_attention_slicing()
                _pipeline = candidate
                return _pipeline, device
            except Exception as exc:
                _pipeline = None
                load_errors.append(f"{device}: {exc}")
                candidate = None
                gc.collect()
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        details = "\n".join(load_errors)
        raise gr.Error(
            f"Could not load the local model from {MODEL_DIR} on the available devices. "
            f"Check that the download finished and PyTorch is compatible. Details: {details}"
        )


def _whole_number(value: int | float, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error(f"{label} must be a whole number.") from exc
    if number != value:
        raise gr.Error(f"{label} must be a whole number.")
    return number


def generate(
    prompt: str,
    negative_prompt: str,
    style: str,
    steps: int,
    guidance: float,
    seed: int,
    width: int,
    height: int,
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write what you want to see in the picture first.")

    width = _whole_number(width, "Width")
    height = _whole_number(height, "Height")
    steps = _whole_number(steps, "Steps")
    seed = _whole_number(seed, "Seed")
    try:
        guidance = float(guidance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("Prompt strength must be a number.") from exc

    if not 256 <= width <= 768 or width % 8:
        raise gr.Error("Width must be between 256 and 768 pixels, in multiples of 8.")
    if not 256 <= height <= 768 or height % 8:
        raise gr.Error("Height must be between 256 and 768 pixels, in multiples of 8.")
    if not 10 <= steps <= 50:
        raise gr.Error("Steps must be between 10 and 50.")
    if not 1 <= guidance <= 12:
        raise gr.Error("Prompt strength must be between 1 and 12.")
    if seed < -1:
        raise gr.Error("Use -1 for a random seed, or enter 0 or a positive number.")
    if seed > 2**53 - 1:
        raise gr.Error("Seed is too large for the browser. Use a value up to 9007199254740991.")

    style_description = STYLE_PRESETS.get(style or "No extra style")
    final_prompt = f"{prompt}, {style_description}" if style_description else prompt
    pipe, device = get_pipeline()
    actual_seed = seed if seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
    try:
        generator = torch.Generator(device=device).manual_seed(actual_seed)
    except Exception as exc:
        raise gr.Error(f"Could not initialize random generation on {device}: {exc}") from exc

    try:
        # Diffusers pipelines share mutable scheduler state, so serialize simultaneous requests.
        with _generation_lock:
            result = pipe(
                prompt=final_prompt,
                negative_prompt=(negative_prompt or "").strip() or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
        image = result.images[0]
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device == "cuda":
                torch.cuda.empty_cache()
            raise gr.Error(
                "Not enough memory for this image. Try a smaller size such as 512 × 512, "
                "fewer steps, or close other GPU-heavy apps."
            ) from exc
        raise gr.Error(f"Image generation failed: {exc}") from exc
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc

    path = OUTPUT_DIR / f"image-{actual_seed}-{uuid4().hex[:8]}.png"
    try:
        image.save(path)
    except OSError as exc:
        raise gr.Error(f"Could not save the image to {OUTPUT_DIR}. Check that the folder is writable.") from exc

    style_label = style if style in STYLE_PRESETS else "No extra style"
    return str(path), f"Saved: {path}\nSeed: {actual_seed} · Device: {device} · Style: {style_label}"


with gr.Blocks(title="Offline AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Offline AI Image Generator\n"
        "Write what picture you want, choose an optional art style, and generate it on this computer. "
        "After the one-time model download, it works without internet."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="What picture should AI make?",
                placeholder="A tiny cabin under a starry sky",
                lines=4,
            )
            gr.Examples(
                examples=[
                    ["A small astronaut tending flowers on the moon"],
                    ["A cozy village beside a lake at sunrise"],
                    ["A friendly orange cat wearing a tiny crown"],
                    ["A peaceful mountain cabin in falling snow"],
                ],
                inputs=[prompt],
                label="Try an example",
            )
            negative = gr.Textbox(label="Things to avoid (optional)", placeholder="blurry, low quality", lines=2)
            style = gr.Dropdown(
                choices=list(STYLE_PRESETS),
                value="No extra style",
                label="Art style",
            )
            with gr.Row():
                width = gr.Slider(256, 768, value=512, step=64, label="Width")
                height = gr.Slider(256, 768, value=512, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(10, 50, value=25, step=1, label="Quality steps")
                guidance = gr.Slider(1, 12, value=7.5, step=0.5, label="Prompt strength")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("Generate image", variant="primary")
        with gr.Column(scale=1):
            image = gr.Image(label="Your AI image", type="filepath")
            status = gr.Textbox(label="Saved image details", interactive=False)
    run.click(
        generate,
        [prompt, negative, style, steps, guidance, seed, width, height],
        [image, status],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        inbrowser=True,
    )
