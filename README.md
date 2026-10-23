# Offline Image Studio

Generate text-to-image artwork on your own computer. The model is downloaded once while you are online. After setup, the app loads the model from disk and generates images locally.

> First time setup needs internet and several GB of free disk space. Image generation itself can run offline. A GPU is optional, but CPU generation can be very slow.

## What you need

- Python 3.10 or 3.11 (recommended for this project).
- Git.
- Internet for installing packages and downloading the model the first time.
- Free disk space for the Python packages, model weights (several GB), and your generated images.
- Windows, macOS, or Linux.
- A supported GPU is recommended. The app can fall back to CPU.

## 1. Clone the repository

Run these commands in the terminal you normally use.

### Windows CMD

~~~cmd
git clone https://github.com/puffxhost/offline-image-studio.git
cd offline-image-studio
py -3.11 --version
~~~

### Windows PowerShell

~~~powershell
git clone https://github.com/puffxhost/offline-image-studio.git
cd offline-image-studio
py -3.11 --version
~~~

### macOS or Linux Terminal

~~~bash
git clone https://github.com/puffxhost/offline-image-studio.git
cd offline-image-studio
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

## 5. Download the model (internet required)

With the internet connected, run:

~~~bash
python scripts/download_model.py
~~~

The default model is stable-diffusion-v1-5/stable-diffusion-v1-5. The download is several GB and may take a while. Wait until the script prints “Download complete.” The model files will be placed in:

~~~text
models/stable-diffusion-v1-5/
~~~

The folder should contain model_index.json. Do not close the terminal or disconnect from the internet while the download is still running. If a download is interrupted, reconnect and run the same command again.

If the model host asks you to accept its terms or sign in, follow the instructions on the model page. Do not put access tokens in this repository.

## 6. Start generating images

You can now disconnect from the internet. From the repository folder, with the virtual environment active, run:

~~~bash
python app.py
~~~

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. If the browser does not open automatically, enter that address yourself.

1. Write a prompt in English (the default model follows English prompts best).
2. Optionally add a negative prompt for things you want the image to avoid.
3. Start with 512 × 512. Higher dimensions need more memory.
4. Leave seed at -1 for a random result, or enter a number to reproduce a result approximately.
5. Select Generate image. The first generation can take longer while the model loads into memory.
6. Find the saved PNG in the repository's outputs folder.

The app listens on your own computer at 127.0.0.1, keeps Gradio sharing disabled, and disables Hugging Face and Gradio telemetry. It does not send prompts or generated images to a hosted inference service.

## 7. Use it again later

You only need to download the model once. On later sessions:

1. Open CMD, PowerShell, or Terminal.
2. Change to the cloned project folder, for example:

~~~cmd
cd path\to\offline-image-studio
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

## Optional settings

The app reads these environment variables when it starts:

- MODEL_DIR: path to a local Diffusers model folder. Default: models/stable-diffusion-v1-5.
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

The selected model must be compatible with Diffusers StableDiffusionPipeline and must already be fully present on disk. The model download script supports changing the Hugging Face repository with MODEL_REPO before downloading; check the model's license and compatibility first.

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

Reconnect to the internet and run python scripts/download_model.py. Confirm that models/stable-diffusion-v1-5/model_index.json exists and that the download completed. If you chose a different model, set MODEL_DIR to its local model folder.

### CUDA is unavailable

The app falls back to CPU. To use an NVIDIA GPU, install the PyTorch command matching your system from the official selector, then check torch.cuda.is_available() again.

### Out of memory or generation is too slow

Use 512 × 512 or smaller dimensions, close other GPU-heavy apps, and try fewer steps. CPU mode can be very slow. Higher-resolution generation requires more memory and can produce artifacts.

### Port 7860 is already in use

Set PORT to 7861 using the matching shell example in Optional settings, then start the app again.

## Model license and responsible use

This repository's code is MIT licensed. The default model has a separate [CreativeML OpenRAIL-M license and model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Read and follow the model's terms and acceptable-use restrictions. If you download another model, check its terms too. The model weights are not included in this repository.

Stable Diffusion v1.5 is primarily trained on English prompts and has known limitations and biases. The app keeps Diffusers' default safety checker enabled.

## License

The app code is distributed under the MIT License in LICENSE. Model weights are downloaded separately and are not covered by this repository's code license.

---

Made with ❤️ by Nitin Mehta.
