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


APP_CSS = r"""
.gradio-container { max-width: 1080px !important; background: radial-gradient(ellipse at 15% 0%,rgba(99,102,241,.15),transparent 34%),radial-gradient(ellipse at 90% 4%,rgba(34,211,238,.08),transparent 25%),#080b14; color:#eef2ff; }
body { background:#080b14 !important; }
#hero { padding:25px 28px 20px; border:1px solid rgba(148,163,184,.15); border-radius:24px; background:linear-gradient(120deg,rgba(20,27,48,.96),rgba(13,22,38,.94)); box-shadow:0 18px 55px rgba(0,0,0,.22); margin:12px 0 14px; }
#hero .eyebrow { color:#a5b4fc; text-transform:uppercase; letter-spacing:.16em; font-size:11px; font-weight:800; }
#hero h1 { color:#f8fafc; font-size:clamp(27px,4vw,42px); line-height:1.08; margin:8px 0; }
#hero p { max-width:680px; color:#aab7ce; font-size:14px; line-height:1.55; margin:0; }
#mode-row { display:flex; justify-content:center; margin:12px auto; max-width:420px; }
#mode-row label { border-radius:14px !important; }
#composer, #results-panel { border:1px solid rgba(148,163,184,.16) !important; border-radius:22px !important; background:rgba(15,20,34,.92) !important; padding:20px !important; box-shadow:0 14px 42px rgba(0,0,0,.2); }
#composer textarea { border-radius:15px !important; min-height:120px !important; font-size:16px !important; line-height:1.55 !important; }
#composer input, #composer select { border-radius:12px !important; }
#idea-row { gap:8px; margin:6px 0 10px; }
#idea-row button { border-radius:999px !important; font-size:11px !important; padding:7px 10px !important; }
#advanced-panel { margin-top:12px; padding:14px; border:1px solid rgba(148,163,184,.14); border-radius:16px; background:rgba(9,14,25,.55); }
#generate-button button { min-height:52px; border-radius:14px !important; font-size:15px !important; font-weight:800 !important; color:white !important; background:linear-gradient(100deg,#6366f1,#8b5cf6 55%,#06b6d4) !important; border:0 !important; box-shadow:0 10px 28px rgba(99,102,241,.23); }
#generate-button button:hover { filter:brightness(1.08); transform:translateY(-1px); }
#result-gallery { border-radius:16px; overflow:hidden; }
#suggestions-panel { border:1px solid rgba(129,140,248,.18); border-radius:16px; background:rgba(30,41,59,.35); padding:14px; }
#developer-card { margin-top:14px; padding:16px 20px; border-radius:18px; border:1px solid rgba(34,211,238,.16); background:linear-gradient(105deg,rgba(14,25,42,.98),rgba(21,26,49,.96)); }
#developer-card h3 { color:#f8fafc; margin:0 0 4px; font-size:15px; }
#developer-card p { color:#9eabc0; margin:4px 0 10px; font-size:12px; }
#developer-card a { color:#a5b4fc; text-decoration:none; margin-right:14px; font-size:12px; }
footer { display:none !important; }
@media (max-width: 680px) {
    .gradio-container { padding-left:10px !important; padding-right:10px !important; }
    #hero { padding:19px 17px; margin-top:6px; border-radius:19px; }
    #composer, #results-panel { padding:14px !important; border-radius:18px !important; }
    #composer textarea { min-height:105px !important; font-size:15px !important; }
    #generate-button button { min-height:50px; }
    #developer-card { padding:14px; }
    #developer-card a { display:inline-block; margin:3px 10px 3px 0; }
}
"""


with gr.Blocks(
    title="Puffx AI Image Studio",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate"),
    css=APP_CSS,
) as demo:
    gr.HTML(
        """
        <section id="hero">
          <div class="eyebrow">PUFFX HOST · PRIVATE IMAGE STUDIO</div>
          <h1>What would you like to create?</h1>
          <p>Describe your idea in your own words. Generate a set of images locally, then refine your prompt if you want.</p>
        </section>
        """
    )

    mode = gr.Radio(
        choices=["Simple", "Advanced"],
        value="Simple",
        label="Choose your workspace",
        info="Simple keeps the screen focused. Advanced reveals the full image controls.",
        elem_id="mode-row",
    )
    with gr.Column(elem_id="composer"):
        prompt = gr.Textbox(
            label="Your prompt",
            placeholder="A tiny cabin beside a quiet lake at sunrise, soft mist, warm cinematic light…",
            lines=4,
            max_lines=8,
            autofocus=True,
        )
        with gr.Row(elem_id="idea-row"):
            gr.Button("Cozy landscape", size="sm").click(
                lambda: "A cozy cabin beside a quiet lake at sunrise, soft mist, warm light",
                outputs=[prompt],
            )
            gr.Button("Dreamy portrait", size="sm").click(
                lambda: "An expressive portrait surrounded by wildflowers, dreamy natural light",
                outputs=[prompt],
            )
            gr.Button("Product photo", size="sm").click(
                lambda: "A premium product on a clean studio backdrop, soft commercial lighting",
                outputs=[prompt],
            )
        with gr.Row():
            image_count = gr.Slider(
                1, 4, value=2, step=1, label="Images to create",
                info="Generate up to four variations together.",
            )
        model_choices = [
            (f"{details['name']} · {details['size']}", key)
            for key, details in MODEL_CATALOG.items()
        ]
        with gr.Column(visible=False, elem_id="advanced-panel") as advanced_panel:
            gr.Markdown("### Advanced controls")
            gr.Markdown("Optional settings for users who want more control.")
            model = gr.Dropdown(
                choices=model_choices,
                value=DEFAULT_MODEL,
                label="AI model",
                info="Choose a model already downloaded on this computer.",
            )
            with gr.Row():
                style = gr.Dropdown(
                    choices=list(STYLE_PRESETS),
                    value="No extra style",
                    label="Art style",
                )
                aspect = gr.Dropdown(
                    choices=["Square · 1:1", "Portrait · 2:3", "Landscape · 3:2", "Wide · 16:9", "Story · 9:16"],
                    value="Square · 1:1",
                    label="Image shape",
                )
            negative = gr.Textbox(
                label="Things to avoid (optional)",
                placeholder="blurry, text, watermark, extra fingers",
                lines=2,
            )
            with gr.Row():
                width = gr.Slider(256, 768, value=512, step=64, label="Width")
                height = gr.Slider(256, 768, value=512, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(1, 50, value=25, step=1, label="Detail steps")
                guidance = gr.Slider(0, 12, value=7.5, step=0.5, label="Prompt strength")
            seed = gr.Number(value=-1, precision=0, label="Seed · -1 for random")

        run = gr.Button("✦ Generate 2 images", variant="primary", elem_id="generate-button")
        gr.Markdown(
            "<div style='text-align:center;color:#8190a8;font-size:11px'>"
            "Runs on your computer · Images save to outputs/ · Choose Advanced for more controls"
            "</div>"
        )

    with gr.Column(elem_id="results-panel"):
        gr.Markdown("## Your images")
        status = gr.Markdown("Your images will appear here.")
        gallery = gr.Gallery(
            label="Generated images",
            columns=2,
            height=560,
            object_fit="contain",
            type="filepath",
            show_label=False,
            elem_id="result-gallery",
        )
        with gr.Group(elem_id="suggestions-panel"):
            gr.Markdown("### Try another direction")
            gr.Markdown("After you create images, choose an idea to update your prompt.")
            suggestion_boxes = [
                gr.Textbox(label=f"Prompt idea {i + 1}", interactive=False, lines=2)
                for i in range(3)
            ]
            suggestion_buttons = [gr.Button(f"Use idea {i + 1}") for i in range(3)]

    gr.HTML(
        """
        <section id="developer-card">
          <h3>Made by Nitin Mehta · Puffx Host</h3>
          <p>Questions or feedback? Contact the developer.</p>
          <a href="mailto:support@puffxhost.in">✉ support@puffxhost.in</a>
          <a href="https://puffxhost.com/contact" target="_blank" rel="noopener">Puffx Host · Contact</a>
          <a href="https://wa.me/917354336624" target="_blank" rel="noopener">WhatsApp support</a>
        </section>
        """
    )

    mode.change(
        lambda value: gr.update(visible=value == "Advanced"),
        inputs=[mode],
        outputs=[advanced_panel],
    )
    model.change(
        update_model_controls,
        inputs=[model],
        outputs=[steps, guidance, width, height],
    )
    aspect.change(
        apply_aspect_ratio,
        inputs=[aspect],
        outputs=[width, height],
    )
    def run_generation(model_key, text, avoid, art_style, detail_steps, prompt_strength, chosen_seed, image_width, image_height, image_total):
        return generate(
            model_key, text, avoid, art_style, detail_steps, prompt_strength,
            chosen_seed, image_width, image_height, image_total,
        )

    def run_simple_generation(text, image_total):
        return generate(
            DEFAULT_MODEL, text, "", "No extra style", 25, 7.5, -1,
            512, 512, image_total,
        )

    def update_count_label(value):
        return gr.update(value=f"✦ Generate {int(value)} image{'s' if int(value) != 1 else ''}")

    image_count.change(update_count_label, inputs=[image_count], outputs=[run])
    def dispatch_generation(current_mode, text, image_total, model_key, avoid, art_style, detail_steps, prompt_strength, chosen_seed, image_width, image_height):
        if current_mode == "Advanced":
            return run_generation(
                model_key, text, avoid, art_style, detail_steps, prompt_strength,
                chosen_seed, image_width, image_height, image_total,
            )
        return run_simple_generation(text, image_total)

    run.click(
        dispatch_generation,
        inputs=[mode, prompt, image_count, model, negative, style, steps, guidance, seed, width, height],
        outputs=[gallery, status, *suggestion_boxes],
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
