"""Download Stable Diffusion v1.5 once so the app can run without internet later."""
from pathlib import Path
import os
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
repo_id = os.environ.get("MODEL_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")
local_dir = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "stable-diffusion-v1-5")).expanduser()
local_dir.mkdir(parents=True, exist_ok=True)
print(f"Downloading {repo_id} to {local_dir}. This can take several GB.")
snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
print("Download complete. You can now disconnect from the internet and launch app.py.")
