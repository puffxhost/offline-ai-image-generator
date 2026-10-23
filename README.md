# Offline Image Studio

A small, local text-to-image app. Download the model once while online; after that, generation runs on your own computer with Hugging Face offline mode enabled. Prompts and generated images are not sent to a hosted service.

## Quick start

1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment:

   ```powershell
   # Windows PowerShell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   # macOS / Linux
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install PyTorch for your hardware using the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). This project supports CPU, NVIDIA CUDA, and Apple Silicon/MPS where supported by your PyTorch build.
4. Install the app packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. While connected to the internet, download the model files:

   ```bash
   python scripts/download_model.py
   ```

   The default model is `stable-diffusion-v1-5/stable-diffusion-v1-5`. The download is several GB; allow it to finish. If the model requires accepting terms or signing in, follow its model page instructions first. You may also set `MODEL_REPO` to another compatible Diffusers model repository before downloading.
6. Disconnect from the internet and launch the app:

   ```bash
   python app.py
   ```

7. Open `http://127.0.0.1:7860` if the browser does not open automatically. Enter a prompt and click **Generate image**. Images are saved under `outputs/`.

## Offline behavior

The app sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and disables Hugging Face telemetry before importing those libraries. The pipeline loads only from the local `models/stable-diffusion-v1-5/` folder with `local_files_only=True`; it does not fetch weights on startup. The UI binds to `127.0.0.1` and Gradio sharing is disabled by default.

For a model stored elsewhere, set `MODEL_DIR` to its local Diffusers directory. The model needs a `model_index.json` and all component/config/weight files. You can also set `OUTPUT_DIR` to choose where generated PNGs are saved.

## Hardware notes

- A supported GPU is strongly recommended for practical generation speed. CPU mode works with lower memory use but may be very slow.
- The default 512×512 output is a sensible starting point for Stable Diffusion 1.5. Higher sizes need more memory and can produce artifacts.
- Set a seed to reproduce a generation approximately; `-1` chooses a random seed.

## Model and usage terms

This repository's code is MIT licensed. The default model has its own [CreativeML OpenRAIL-M license and model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5); those model terms are separate from this code's license. Read and follow the model license and acceptable-use restrictions before downloading or using it. If you choose another model, check that model's terms too.

Stable Diffusion v1.5 is primarily trained on English prompts and has known limitations and biases. The model card describes its recommended safety checker. This app keeps Diffusers' default safety checker enabled.

## Troubleshooting

- **Model not found:** run the download step while online and check for `models/stable-diffusion-v1-5/model_index.json`.
- **PyTorch has no CUDA/MPS:** reinstall the PyTorch build that matches your hardware; the app falls back to CPU.
- **Out of memory:** use a GPU with more memory or CPU mode, and reduce width/height.
- **Download needs authentication:** sign in with Hugging Face CLI or use a token as described by Hugging Face, then rerun the download script. Do not put tokens in this repository.

## License

The app code is distributed under the MIT License in `LICENSE`. Model weights are downloaded separately and are not included in this repository.
