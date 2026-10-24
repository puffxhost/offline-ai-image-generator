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
from diffusers import DiffusionPipeline

from model_catalog import DEFAULT_MODEL, MODEL_CATALOG

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", ROOT / "models")).expanduser()
CUSTOM_DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_pipeline = None
_pipeline_model: str | None = None
_pipeline_lock = Lock()
_generation_lock = Lock()

STYLE_PRESETS = {
    "No extra style": "",
    "Photorealistic": "photorealistic, realistic lighting, detailed textures",
    "Digital art": "polished digital illustration, rich colors",
    "Watercolor": "watercolor painting, soft pigments, paper texture",
    "Anime": "anime illustration, expressive line art",
    "Pixel art": "pixel art, crisp pixels, limited color palette",
    "Cinematic": "cinematic lighting, dramatic composition, rich shadows, film still",
    "3D render": "high-quality 3D render, soft studio lighting, detailed materials",
    "Product photo": "premium commercial product photography, clean studio backdrop",
    "Oil painting": "expressive oil painting, visible brush strokes, layered pigments",
    "Concept art": "professional environment concept art, atmospheric perspective",
    "Fashion editorial": "high-fashion editorial photography, elegant pose, refined styling",
}


def model_directory(model_key: str) -> Path:
    if model_key not in MODEL_CATALOG:
        raise gr.Error("Choose one of the supported models.")
    if model_key == DEFAULT_MODEL and CUSTOM_DEFAULT_MODEL_DIR:
        return Path(CUSTOM_DEFAULT_MODEL_DIR).expanduser()
    return MODEL_ROOT / model_key


def _clear_device_cache(device: str | None = None) -> None:
    gc.collect()
    if (device and device.startswith("cuda")) or (device is None and torch.cuda.is_available()):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if device == "mps" or (device is None and hasattr(torch, "mps")):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_pipeline(model_key: str, force_cpu: bool = False):
    global _pipeline, _pipeline_model
    if model_key not in MODEL_CATALOG:
        raise gr.Error("Choose one of the supported models.")

    model_dir = model_directory(model_key)
    if not (model_dir / "model_index.json").is_file():
        raise gr.Error(
            f"{MODEL_CATALOG[model_key]['name']} is not installed at {model_dir}. "
            f"Connect to the internet and run: python scripts/download_model.py --model {model_key}"
        )

    with _pipeline_lock:
        if _pipeline is not None and _pipeline_model == model_key:
            loaded_device = str(_pipeline._execution_device)
            if not force_cpu or loaded_device == "cpu":
                return _pipeline, loaded_device

        previous_device = str(_pipeline._execution_device) if _pipeline is not None else None
        _pipeline = None
        _pipeline_model = None
        _clear_device_cache(previous_device)

        if force_cpu:
            device_options = [("cpu", torch.float32)]
        else:
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
                candidate = DiffusionPipeline.from_pretrained(
                    str(model_dir),
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                candidate.to(device)
                if device in {"cpu", "mps"}:
                    candidate.enable_attention_slicing()
                _pipeline = candidate
                _pipeline_model = model_key
                return _pipeline, device
            except Exception as exc:
                _pipeline = None
                _pipeline_model = None
                load_errors.append(f"{device}: {exc}")
                candidate = None
                _clear_device_cache(device)

        details = "\n".join(load_errors)
        raise gr.Error(
            f"Could not load {MODEL_CATALOG[model_key]['name']} from {model_dir} "
            f"on the available devices. Check the download, model compatibility, and "
            f"PyTorch installation. Details: {details}"
        )


def _whole_number(value: int | float, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error(f"{label} must be a whole number.") from exc
    if number != value:
        raise gr.Error(f"{label} must be a whole number.")
    return number


def update_model_controls(model_key: str):
    settings = MODEL_CATALOG.get(model_key, MODEL_CATALOG[DEFAULT_MODEL])
    step_min, step_max, step_default = settings["steps"]
    guidance_min, guidance_max, guidance_default = settings["guidance"]
    resolution_max = settings.get("max_resolution", 768)
    return (
        gr.update(minimum=step_min, maximum=step_max, value=step_default),
        gr.update(minimum=guidance_min, maximum=guidance_max, value=guidance_default),
        gr.update(maximum=resolution_max, value=512),
        gr.update(maximum=resolution_max, value=512),
    )


def apply_aspect_ratio(preset: str):
    dimensions = {
        "Square · 1:1": (512, 512),
        "Portrait · 2:3": (512, 768),
        "Landscape · 3:2": (768, 512),
        "Wide · 16:9": (768, 432),
        "Story · 9:16": (432, 768),
    }
    width_value, height_value = dimensions.get(preset, (512, 512))
    return gr.update(value=width_value), gr.update(value=height_value)


def prompt_suggestions(prompt: str, style: str) -> list[str]:
    prompt = (prompt or "").strip().rstrip(".,")
    style_hint = STYLE_PRESETS.get(style or "", "")
    suffix = f", {style_hint}" if style_hint else ""
    return [
        f"{prompt}, cinematic composition, soft directional lighting, rich visual detail{suffix}",
        f"{prompt}, editorial framing, atmospheric depth, balanced colors, polished finish{suffix}",
        f"{prompt}, close-up subject focus, beautiful natural light, intricate texture{suffix}",
    ]


def _generate_one(
    model_key: str,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
):
    global _pipeline, _pipeline_model
    pipe, device = get_pipeline(model_key)
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
    except Exception as exc:
        raise gr.Error(f"Could not initialize random generation on {device}: {exc}") from exc

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "generator": generator,
    }
    try:
        result = pipe(**kwargs)
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not is_oom or not (device.startswith("cuda") or device == "mps"):
            raise

        failed_device = device
        _pipeline = None
        _pipeline_model = None
        del pipe
        _clear_device_cache(failed_device)
        try:
            pipe, device = get_pipeline(model_key, force_cpu=True)
            kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
            result = pipe(**kwargs)
        except Exception as fallback_exc:
            raise gr.Error(
                f"Not enough memory on {failed_device}; CPU retry also failed: {fallback_exc}"
            ) from fallback_exc
    return result.images[0], device


def generate(
    model_key: str,
    prompt: str,
    negative_prompt: str,
    style: str,
    steps: int,
    guidance: float,
    seed: int,
    width: int,
    height: int,
    image_count: int,
):
    if model_key not in MODEL_CATALOG:
        raise gr.Error("Choose one of the supported models.")
    step_min, step_max, _ = MODEL_CATALOG[model_key]["steps"]
    guidance_min, guidance_max, _ = MODEL_CATALOG[model_key]["guidance"]

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write what you want to see in the picture first.")

    width = _whole_number(width, "Width")
    height = _whole_number(height, "Height")
    steps = _whole_number(steps, "Steps")
    seed = _whole_number(seed, "Seed")
    image_count = _whole_number(image_count, "Image count")
    try:
        guidance = float(guidance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("Prompt strength must be a number.") from exc

    max_resolution = MODEL_CATALOG[model_key].get("max_resolution", 768)
    if not 256 <= width <= max_resolution or width % 8:
        raise gr.Error(f"Width must be between 256 and {max_resolution} pixels, in multiples of 8.")
    if not 256 <= height <= max_resolution or height % 8:
        raise gr.Error(f"Height must be between 256 and {max_resolution} pixels, in multiples of 8.")
    if not step_min <= steps <= step_max:
        model_name = MODEL_CATALOG[model_key]["name"]
        raise gr.Error(f"Steps for {model_name} must be between {step_min} and {step_max}.")
    if not guidance_min <= guidance <= guidance_max:
        model_name = MODEL_CATALOG[model_key]["name"]
        raise gr.Error(f"Prompt strength for {model_name} must be between {guidance_min} and {guidance_max}.")
    if not 1 <= image_count <= 4:
        raise gr.Error("Choose between 1 and 4 images per generation.")
    if seed < -1:
        raise gr.Error("Use -1 for a random seed, or enter 0 or a positive number.")
    if seed > 2**53 - 1:
        raise gr.Error("Seed is too large for the browser. Use a value up to 9007199254740991.")

    style_description = STYLE_PRESETS.get(style or "No extra style")
    final_prompt = f"{prompt}, {style_description}" if style_description else prompt
    first_seed = seed if seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
    if first_seed + image_count - 1 > 2**53 - 1:
        raise gr.Error("This seed is too large for the selected image count.")

    negative_value = (negative_prompt or "").strip() or None
    generated = []
    devices = []
    seeds = []
    try:
        # Keep one pipeline in memory and serialize batches/model switching.
        with _generation_lock:
            for index in range(image_count):
                image_seed = first_seed + index
                image, device = _generate_one(
                    model_key,
                    final_prompt,
                    negative_value,
                    steps,
                    guidance,
                    width,
                    height,
                    image_seed,
                )
                generated.append((image, image_seed))
                devices.append(device)
                seeds.append(image_seed)
    except gr.Error:
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _clear_device_cache(devices[-1] if devices else None)
            raise gr.Error(
                "Not enough memory for this image batch. Try fewer images, a smaller size, "
                "fewer steps, or close other GPU-heavy apps."
            ) from exc
        raise gr.Error(f"Image generation failed: {exc}") from exc
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}") from exc

    gallery_items = []
    for image, image_seed in generated:
        path = OUTPUT_DIR / f"image-{image_seed}-{uuid4().hex[:8]}.png"
        try:
            image.save(path)
        except OSError as exc:
            raise gr.Error(f"Could not save the image to {OUTPUT_DIR}. Check that the folder is writable.") from exc
        gallery_items.append((str(path), f"Seed {image_seed}"))

    style_label = style if style in STYLE_PRESETS else "No extra style"
    device_label = devices[-1] if devices else "unknown"
    status = (
        f"Created {len(gallery_items)} image{'s' if len(gallery_items) != 1 else ''} · "
        f"{width} × {height} · {steps} steps · Seed{'s' if len(seeds) != 1 else ''}: "
        f"{', '.join(str(value) for value in seeds)} · {device_label} · {style_label}"
    )
    suggestions = prompt_suggestions(prompt, style)
    return gallery_items, status, *suggestions


with gr.Blocks(title="Puffx AI Image Generator · Puffx Host", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Puffx AI Image Generator\n"
        "Create images from your ideas on your own computer with Puffx AI. "
        "Choose a model, download it once, and use it offline.\n\n"
        "**By Puffx Host** · [puffxhost.com](https://puffxhost.com)"
    )
    gr.Markdown(
        "### Choose a model\n"
        "Download any model first with python scripts/download_model.py --model <id>. "
        "Models are stored separately; switching loads the selected model from disk and may take a few minutes."
    )
    model_choices = [
        (f"{details['name']} · {details['size']} · {details['tier']}", key)
        for key, details in MODEL_CATALOG.items()
    ]
    model = gr.Dropdown(
        choices=model_choices,
        value=DEFAULT_MODEL,
        label="AI model",
        info="Small models need less disk/RAM; SDXL and DreamShaper need more. See README for trade-offs.",
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
                steps = gr.Slider(1, 50, value=25, step=1, label="Quality steps")
                guidance = gr.Slider(0, 12, value=7.5, step=0.5, label="Prompt strength")
            with gr.Row():
                image_count = gr.Slider(1, 4, value=2, step=1, label="Images per batch")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("Generate 2 images", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Your AI images",
                columns=2,
                height=500,
                object_fit="contain",
                type="filepath",
            )
            status = gr.Textbox(label="Saved image details", interactive=False)
            with gr.Accordion("Prompt suggestions for your next image", open=False):
                gr.Markdown("Pick an idea to reuse it in the prompt box.")
                suggestion_boxes = [
                    gr.Textbox(label=f"Prompt idea {i + 1}", interactive=False, lines=2)
                    for i in range(4)
                ]
                suggestion_buttons = [
                    gr.Button(f"Use idea {i + 1}")
                    for i in range(4)
                ]
    model.change(
        update_model_controls,
        inputs=[model],
        outputs=[steps, guidance, width, height],
    )
    gr.Markdown(
        "---\nMade with ❤️ by **Puffx Host** · "
        "[Support](https://puffxhost.com/contact) · "
        "[support@puffxhost.in](mailto:support@puffxhost.in)"
    )
    image_count.change(
        lambda value: gr.update(
            value=f"Generate {int(value)} image{'s' if int(value) != 1 else ''}"
        ),
        inputs=[image_count],
        outputs=[run],
    )
    run.click(
        generate,
        [model, prompt, negative, style, steps, guidance, seed, width, height, image_count],
        [gallery, status, *suggestion_boxes],
        concurrency_limit=1,
    )
    for button, suggestion in zip(suggestion_buttons, suggestion_boxes):
        button.click(lambda text: text, inputs=[suggestion], outputs=[prompt])

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        inbrowser=True,
    )
