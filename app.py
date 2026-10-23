"""Local-only text-to-image app. Model files must already exist on disk."""
from __future__ import annotations

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

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        try:
            _pipeline = StableDiffusionPipeline.from_pretrained(
                str(MODEL_DIR),
                torch_dtype=dtype,
                local_files_only=True,
            )
            _pipeline.to(device)
            if device == "cpu":
                _pipeline.enable_attention_slicing()
        except Exception as exc:
            _pipeline = None
            raise gr.Error(
                f"Could not load the local model from {MODEL_DIR}. Check that the download "
                f"finished and the installed PyTorch/Diffusers versions are compatible. Details: {exc}"
            ) from exc
        return _pipeline, device


def generate(prompt: str, negative_prompt: str, steps: int, guidance: float, seed: int, width: int, height: int):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    if width % 8 or height % 8:
        raise gr.Error("Width and height must be multiples of 8.")

    pipe, device = get_pipeline()
    actual_seed = int(seed) if int(seed) >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device=device).manual_seed(actual_seed)
    try:
        result = pipe(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=generator,
        )
        image = result.images[0]
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc

    path = OUTPUT_DIR / f"image-{actual_seed}-{uuid4().hex[:8]}.png"
    image.save(path)
    return str(path), f"Saved {path}\nSeed: {actual_seed} · Device: {device}"


with gr.Blocks(title="Offline AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Offline AI Image Generator\n"
        "Write what picture you want, and AI will make it on this computer. "
        "Download the AI model once while online; then you can use the app without internet."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", placeholder="A tiny cabin under a starry sky, watercolor", lines=4)
            negative = gr.Textbox(label="Negative prompt (optional)", placeholder="blurry, low quality", lines=2)
            with gr.Row():
                width = gr.Slider(256, 768, value=512, step=64, label="Width")
                height = gr.Slider(256, 768, value=512, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                guidance = gr.Slider(1, 12, value=7.5, step=0.5, label="Prompt strength")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("Generate image", variant="primary")
        with gr.Column(scale=1):
            image = gr.Image(label="Generated image", type="filepath")
            status = gr.Textbox(label="Status", interactive=False)
    run.click(generate, [prompt, negative, steps, guidance, seed, width, height], [image, status])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", "7860")), share=False, inbrowser=True)




