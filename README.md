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

The app includes a model menu. Download whichever models you want while online; each one stays in its own folder, and you can switch models later from the app. You do not have to install all of them. Download size is disk space used by model files, not the RAM/VRAM requirement. Actual totals can vary with repository updates and file formats.

| Model | Approximate model download | Best for | Notes |
| --- | ---: | --- | --- |
| [Segmind Tiny SD](https://huggingface.co/segmind/tiny-sd) (tiny-sd) | about 1.1 GB | Low disk space and lighter computers | Smallest option; expect simpler results. |
| [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (stable-diffusion-v1-5) | about 4.3 GB of weights | General-purpose images | Balanced default. |
| [Stable Diffusion Turbo](https://huggingface.co/stabilityai/sd-turbo) (sd-turbo) | about 5.2 GB of weights | Faster drafts | Use 1–4 steps. Read Stability AI model terms. |
| [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (sdxl-base-1.0) | about 7 GB of weights | Detailed, larger images | Requires more memory; start at 512 × 512 if needed. |
| [DreamShaper 8](https://huggingface.co/Lykon/dreamshaper-8) (dreamshaper-8) | about 8 GB | Illustrative and realistic styles | Larger download; review its model card and license. |

Sizes are approximate and can change. The downloader requests Diffusers component folders rather than unrelated checkpoint files that some model repositories keep alongside the pipeline. The exact transferred size can differ from these estimates.

First, see the model IDs and descriptions:

~~~bash
python scripts/download_model.py --list
~~~

Download only the model you want. For example, the smallest model:

~~~bash
python scripts/download_model.py --model tiny-sd
~~~

Or download the default model:

~~~bash
python scripts/download_model.py
~~~

To download another choice, replace the ID with sd-turbo, sdxl-base-1.0, or dreamshaper-8. Downloads go into models/<model-id>/; for example, Tiny SD is stored in models/tiny-sd/. You need enough free disk space for every model you choose and your generated images. Keep the terminal open and internet connected until it prints “Download complete.” If interrupted, rerun the same command to resume.

The model chooser appears in the app after launch. You can install several models, choose one from the AI model menu, and change the selection later. Switching unloads the previous pipeline before loading the next model to reduce memory use. Initial loading may take a few minutes. The selected model must already be downloaded to work offline.

If a model host asks you to accept terms or sign in, follow the model card instructions. Never put access tokens in this repository. Check each model license and acceptable-use terms before use.

## 6. Start generating images

You can now disconnect from the internet. From the repository folder, with the virtual environment active, run:

~~~bash
python app.py
~~~

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. If the browser does not open automatically, enter that address yourself.

1. Choose a downloaded model in the AI model menu. The app can only use models already downloaded to the models folder.
2. Type what you want to see, or click one of the example ideas. Most catalog models follow English prompts best.
3. Optionally list things to avoid, such as "blurry" or "extra fingers."
4. Choose an art style, such as watercolor, anime, or photorealistic.
5. Start with 512 × 512; larger images need more memory.
6. Leave seed at -1 for a different result each time, or reuse a whole number from 0 to 9007199254740991 to make a similar result again.
7. Select Generate image. The first generation can take longer while the model loads. The app shows the seed, device, and style, and saves a PNG in the outputs folder.

The app listens on your own computer at 127.0.0.1, keeps Gradio sharing disabled, and disables Hugging Face and Gradio telemetry. It does not send prompts or generated images to a hosted inference service.

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

The image is published to [GitHub Container Registry](https://github.com/puffxhost/offline-ai-image-generator/pkgs/container/offline-ai-image-generator) when a stable GitHub Release is published. It contains the app and CPU PyTorch, but **not the Stable Diffusion model weights**. The model is downloaded separately into a folder on your computer, so you keep control of the model files and can use them offline.

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
