# Puffx AI Image Generator

**Puffx AI Image Generator** is a local text-to-image app from **Puffx Host**. Turn written ideas into images on your own computer. Download a model once while online; after setup, generate locally, including offline.

> First time setup needs internet and several GB of free disk space. Image generation itself can run offline. A GPU is optional, but CPU generation can be very slow.

## What you need

- Python 3.10 or 3.11 (recommended for this project).
- Git.
- Internet for installing packages and downloading the model the first time.
- Free disk space for the Python packages, model weights (several GB), and your generated images.
- Windows, macOS, or Linux.
- A supported GPU is recommended. If the model will not load on the GPU, the app retries on CPU; CPU generation is much slower and needs enough system memory.

## 1. Clone the repository

Run these commands in the terminal you normally use.

### Windows CMD

~~~cmd
git clone https://github.com/puffxhost/offline-ai-image-generator.git
cd offline-ai-image-generator
py -3.11 --version
~~~

### Windows PowerShell

~~~powershell
git clone https://github.com/puffxhost/offline-ai-image-generator.git
cd offline-ai-image-generator
py -3.11 --version
~~~

### macOS or Linux Terminal

~~~bash
git clone https://github.com/puffxhost/offline-ai-image-generator.git
cd offline-ai-image-generator
python3.11 --version
~~~

If the version command says Python 3.11 is unavailable, install Python 3.11 and reopen the terminal. On Windows, you can also check py --list; on macOS/Linux, check python3 --version.

## 2. Create and activate a virtual environment

A virtual environment keeps this app's Python packages separate from other projects.

### Windows CMD

~~~cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
~~~

When it is active, CMD usually shows (.venv) at the start of the prompt.

### Windows PowerShell

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

If PowerShell blocks activation, you can leave activation off and invoke the environment's Python directly. For example:

~~~powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
~~~

Use that same .\.venv\Scripts\python.exe prefix for the install, download, and launch commands in the next sections.

### macOS or Linux Terminal

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

If activation worked, your prompt normally starts with (.venv).

## 3. Install PyTorch for your computer

Install PyTorch before the packages in requirements.txt. The correct command depends on your operating system and hardware.

1. Open the official [PyTorch install selector](https://pytorch.org/get-started/locally/).
2. Select your operating system, Pip, and Python.
3. Select CPU if you do not have a supported GPU. For NVIDIA, select a CUDA option supported by your driver. On Apple Silicon, use the macOS/Pip instructions.
4. Copy the command shown by the selector and run it in the same terminal while the virtual environment is active.

Do not guess a CUDA command from another computer. If you are unsure, choose CPU first; it works but is much slower for image generation. See the selector for current commands and supported combinations.

You can check that PyTorch installed in the active environment:

~~~bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
~~~

A value of False for CUDA is normal on a CPU-only computer or when the wrong PyTorch build was selected. On a Mac, check the MPS value.

## 4. Install the app packages

Keep the virtual environment active, then run:

~~~bash
python -m pip install -r requirements.txt
~~~

This installs Diffusers, Transformers, Gradio, and the model-loading libraries. PyTorch is installed separately because its package command varies by hardware.

## 5. Choose and download an AI model (internet required once)

The app supports the models below through the standard Diffusers text-to-image pipeline. There is no single best model for every computer: pick by the balance you want between speed, detail, and disk/RAM/VRAM. The download sizes are estimates. The downloader fetches Diffusers component folders and skips duplicate standalone checkpoint files where the model repository includes them, so actual transfer size may differ.

| Model | Download estimate | Good starting choice for | Notes |
| --- | ---: | --- | --- |
| [Segmind Tiny SD](https://huggingface.co/segmind/tiny-sd) (`tiny-sd`) | about 1.1 GB | Low disk space and modest computers | Small and quick; image detail is simpler. |
| [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (`stable-diffusion-v1-5`) | about 4.3 GB | General-purpose starting point | Default model; wide support and a good speed/quality balance. |
| [Stable Diffusion Turbo](https://huggingface.co/stabilityai/sd-turbo) (`sd-turbo`) | about 5.2 GB | Fast drafts | Designed for 1–4 steps; uses low prompt guidance. Check its model terms for your use. |
| [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (`sdxl-base-1.0`) | about 7 GB | More detail and complex scenes | Needs more memory than SD 1.5. Start at 512 × 512 or 640 × 640. |
| [DreamShaper 8](https://huggingface.co/Lykon/dreamshaper-8) (`dreamshaper-8`) | about 8 GB | Illustrative and realistic styles | SD 1.5-based fine-tune with a flexible style range. |
| [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) (`ssd-1b`) | about 13–14 GB of Diffusers files | Detailed SDXL-style images with faster generation | Distilled SDXL family model; its card reports faster inference than SDXL Base. Apache 2.0 license. |
| [DreamShaper XL](https://huggingface.co/Lykon/dreamshaper-xl-1-0) (`dreamshaper-xl`) | about 21 GB | High-detail creative, anime, or realistic images | Large SDXL fine-tune; requires plenty of free disk and memory. Review the model card license. |

Sizes and performance vary by revision, file format, and computer. A large download does not mean it will fit in GPU memory. If unsure, start with Tiny SD or the default Stable Diffusion 1.5. For better detail on a capable computer, try SSD-1B or DreamShaper XL. Every model has its own license and usage terms; read its model card before downloading or using it.

See the supported IDs and descriptions:

~~~bash
python scripts/download_model.py --list
~~~

Download only the models you want, while online. For example:

~~~bash
python scripts/download_model.py --model ssd-1b
python scripts/download_model.py --model dreamshaper-xl
~~~

Use any ID in the table after `--model`. The files are stored separately under `models/<model-id>/`, so you can keep multiple models installed. For example, SSD-1B goes in `models/ssd-1b/`. Run one download command at a time. Leave the terminal open and connected until “Download complete” appears. If interrupted, run the same command again to resume.

If a model host asks you to accept terms or sign in, complete that on the model page first. Do not put access tokens in this repository. Downloads can use a lot of disk space; check the model page for current files and terms before starting.

### Switch between downloaded models

1. Start the app with `python app.py` (or `docker compose up` for Docker).
2. In the **AI model** menu, choose the model you want.
3. Enter a prompt and press **Generate image**. The selected model loads from its local folder the first time it is used. The first generation can take several minutes.
4. To switch, choose another model and generate again. The app unloads the previous model before loading the next one to reduce memory use.
5. Keep the internet disconnected for offline use. A model must already be downloaded on this computer; choosing an uninstalled model shows its ID and download command.

Downloading a model does not change the app code or replace the other model folders. You can switch back any time by choosing that model again. If the model download is incomplete, rerun its download command while connected to the internet.

### Download a model for Docker

The native Python downloader and Docker share the same `models/` folder. For the new model choices in this checkout, first build the updated app image, then download a model before disconnecting:

~~~bash
docker compose build
docker compose run --rm puffx-ai-image-generator python scripts/download_model.py --model ssd-1b
~~~

Replace `ssd-1b` with another model ID from the table. Repeat the command to install more models. Then start the app with `docker compose up` and select the downloaded model in the menu. Docker is CPU-only in this project, so large SDXL models may be very slow or may need more system memory than your computer has.

## 6. Generate images and use the controls

Start the app from the repository folder, with the virtual environment active:

~~~bash
python app.py
~~~

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). If the browser does not open automatically, enter that address yourself.

1. The app opens in **Simple** mode. Write a prompt in your own words, choose how many images to create (1–4), and press **Generate**. A model is selected for you automatically.
2. Use a quick idea chip if you want a prompt starter. The catalog models generally respond best to English prompts.
3. Switch to **Advanced** only when you want to choose a downloaded model, style, composition, avoid prompt, dimensions, detail steps, prompt strength, or seed.
4. In Advanced mode, leave seed at `-1` for a random result, or reuse a whole-number seed for a similar result later. Standard models support dimensions from 256 to 768 pixels; SDXL models support up to 1024 pixels. Larger images need substantially more memory.
5. Generate 1–4 variations in either mode. Each image is saved as a PNG in `outputs/`; the gallery and status show the results and seeds.
6. After generation, choose one of the prompt ideas to update your prompt and make another set.

### Studio features

- A focused, chat-inspired prompt screen in Simple mode, with one prompt box and multi-image generation.
- Advanced mode reveals optional controls instead of crowding the main screen.
- A responsive layout for phones, tablets, and desktop screens.
- Twelve art directions and five composition presets in Advanced mode.
- Model-aware detail controls, reproducible seeds, negative prompts, and prompt sparks.
- A results gallery, local PNG saving, and developer support links.

### Switch models

Start the app and select **Advanced**. Open the **AI model** menu and choose a downloaded model. If you have not downloaded it yet, run `python scripts/download_model.py --model <model-id>`, then reopen the app and select it. The selected model controls its available steps, prompt strength, and maximum image size. XL models use more memory, so start at 512 × 512 if unsure.

The app listens on your own computer at `127.0.0.1`, keeps Gradio sharing disabled, and disables Hugging Face and Gradio telemetry. Prompts and generated images are not sent to a hosted inference service.

## 7. Use it again later

You only need to download the model once. On later sessions:

1. Open CMD, PowerShell, or Terminal.
2. Change to the cloned project folder, for example:

~~~cmd
cd path\to\offline-ai-image-generator
~~~

3. Activate the environment using the command for your terminal:

~~~cmd
.venv\Scripts\activate.bat
~~~

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
source .venv/bin/activate
~~~

4. Start the app:

~~~bash
python app.py
~~~

5. To stop it, return to the terminal and press Ctrl+C.

Keep the model directory and the virtual environment on the computer. You can use the same installation offline.

## 8. Other places you can run it

### Windows Subsystem for Linux (WSL2)

WSL2 is useful if you prefer Linux commands while using Windows. First install Ubuntu from an elevated Windows PowerShell or CMD:

~~~powershell
wsl --install -d Ubuntu
wsl --status
~~~

Restart Windows if requested, then open Ubuntu from the Start menu and create your Linux username. In the Ubuntu terminal:

~~~bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/puffxhost/offline-ai-image-generator.git
cd offline-ai-image-generator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

Install PyTorch using the official selector. Choose Linux/Pip/Python and CPU, or choose a CUDA build only if your Windows NVIDIA driver and WSL2 GPU support are configured. Then continue with the requirements install, model download, and app launch steps above.

For better file performance, keep the project inside the WSL Linux filesystem (for example, under your Linux home directory) instead of /mnt/c. After downloading the model, the app can generate offline inside WSL2.

### Docker (CPU mode)

**Puffx Host** provides a CPU-only Docker image for computers with Docker installed. CPU image generation can be slow. Apple Silicon/MPS and NVIDIA CUDA acceleration are not configured in this container; use a native installation for GPU acceleration.

The image is published to [GitHub Container Registry](https://github.com/puffxhost/offline-ai-image-generator/pkgs/container/offline-ai-image-generator) when a stable GitHub Release is published. It contains the app and CPU PyTorch, but **not the Stable Diffusion model weights**. The model is downloaded separately into a folder on your computer, so you keep control of the model files and can use them offline. The published image may not yet include model choices added after its release. To use the current repository's full model catalog, build the image locally with `docker compose build` before downloading models.

Install Docker Desktop on Windows/macOS, or Docker Engine and the Compose plugin on Linux. Clone this repository as described above, open a terminal in the repository folder, and pull the prebuilt image while online:

~~~bash
docker compose pull
~~~

If you prefer to build the image locally instead, use:

~~~bash
docker compose build
~~~

Download the model into the mounted models folder while online:

~~~bash
docker compose run --rm puffx-ai-image-generator python scripts/download_model.py --model tiny-sd
~~~

When the script prints “Download complete,” disconnect from the internet and start the app:

~~~bash
docker compose up
~~~

Open http://127.0.0.1:7860 in your browser. Images are saved in the repository's outputs folder. Press Ctrl+C to stop; use `docker compose down` to stop the container. The model and output folders remain on your computer. On later runs, use `docker compose up`; the container image, model, and generated images are stored separately.

The Puffx Host container package publishes `latest`, `1`, `1.1`, and `1.1.0` tags for release `v1.1.0`. Each stable release publishes version-matched tags automatically. To pin a specific version, set `image: ghcr.io/puffxhost/offline-ai-image-generator:1.1.0` in `compose.yaml`.

Building the image or downloading the model requires internet once. The prebuilt CPU image supports `linux/amd64`; the model weights are several GB and must be downloaded separately. For completely offline use, pull/build the image and finish the model download before disconnecting.
### Cloud notebooks and remote machines

Google Colab, Kaggle notebooks, and hosted virtual machines can run Python apps, but they are not a reliable way to meet the offline goal: they need internet to set up, may reset their storage, and run on someone else's computer. Use a cloud notebook only when you are okay with those limits.

A remote Linux computer can run this app if it has Python and enough disk/RAM/GPU, but it is only offline after all packages and the model are already present there. Connecting to that remote machine still uses a network. For fully local offline generation, use Windows, macOS, Linux, WSL2, or Docker on your own computer.


## Optional settings

The app reads these environment variables when it starts:

- MODEL_ROOT: parent folder for catalog models. Default: models. Each model is stored in its own subfolder, such as models/tiny-sd.
- MODEL_DIR: optional custom local Diffusers folder used for the default Stable Diffusion 1.5 choice. Prefer MODEL_ROOT for catalog models.
- OUTPUT_DIR: path where generated PNG files are saved. Default: outputs.
- HOST: local web address to bind. Default: 127.0.0.1.
- PORT: web port. Default: 7860.

Examples for using a different local model directory:

~~~cmd
set MODEL_DIR=D:\AI\Models\my-model
python app.py
~~~

~~~powershell
$env:MODEL_DIR = "D:\AI\Models\my-model"
python app.py
~~~

~~~bash
MODEL_DIR="$HOME/AI/Models/my-model" python app.py
~~~

The selected catalog model must already be downloaded. MODEL_DIR is a legacy custom-folder override for the default choice. A custom model must be a local Diffusers pipeline with model_index.json and compatible component files. MODEL_REPO applies only to the default downloader choice; third-party custom models may require different pipeline classes and are not guaranteed to work.

If port 7860 is already in use, pick another port for that run:

~~~cmd
set PORT=7861
python app.py
~~~

~~~powershell
$env:PORT = "7861"
python app.py
~~~

~~~bash
PORT=7861 python app.py
~~~

## Troubleshooting

### py or python is not recognized

Install Python 3.11 and select the installer option that adds Python to PATH. Close and reopen CMD/PowerShell after installation. On Windows, try py -3.11; on macOS/Linux, try python3.11.

### PowerShell says scripts are disabled

Use the no-activation method from the virtual environment section. Run commands with .\.venv\Scripts\python.exe instead of python; this does not require changing the machine's execution policy.

### No module named torch, diffusers, or gradio

Make sure the virtual environment is active in this terminal. Then install PyTorch using the official selector and run:

~~~bash
python -m pip install -r requirements.txt
~~~

### The model is missing or offline loading fails

Reconnect to the internet and run python scripts/download_model.py --model <model-id>. Confirm that models/<model-id>/model_index.json exists and that the download completed. Select that same model in the app.

### CUDA is unavailable

The app falls back to CPU. To use an NVIDIA GPU, install the PyTorch command matching your system from the official selector, then check torch.cuda.is_available() again.

### Out of memory or generation is too slow

Use 512 × 512 or smaller dimensions, close other GPU-heavy apps, and try fewer steps. CPU mode can be very slow. Higher-resolution generation requires more memory and can produce artifacts.

### Port 7860 is already in use

Set PORT to 7861 using the matching shell example in Optional settings, then start the app again.

## Model license and responsible use

This repository's code is MIT licensed. Every model has separate terms. Review the model cards linked in the model table before downloading or using a model, and follow its license and acceptable-use restrictions. Model weights are not included in this repository.

Catalog models have different training data, prompt strengths, safety behavior, and limitations. Read each model card. The app uses each pipeline's published Diffusers configuration; not every model provides a safety checker.

## 🛟 Support & Contact

Need help or want to report an issue?

- 🌐 Website: [puffxhost.com/contact](https://puffxhost.com/contact)
- 📧 Email: [support@puffxhost.in](mailto:support@puffxhost.in)
- 📞 Call: [+91 8602967573](https://wa.me/918602967573)
- 💬 WhatsApp: [+91 7354336624](https://wa.me/917354336624)

## 📜 License

This project’s app code is licensed under the MIT License.
See the LICENSE file for full details. Model weights have separate license terms; see the model license section above.

## 🙌 Contributing

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

## ⭐️ Give a Star

If you found this project useful, please consider giving it a star on GitHub to support its development.

## 🔗 About

Type what picture you want, and this app creates it on your computer with AI. Set it up and download the AI model once while online; after that, you can make pictures without internet.

## 🧠 Puffx Host

**Puffx AI Image Generator** is developed by **Nitin Mehta** and maintained under the Puffx Host brand.

Official site: [puffxhost.com](https://puffxhost.com) · Support: [puffxhost.com/contact](https://puffxhost.com/contact)

---

Made with ❤️ by **Puffx Host**.
