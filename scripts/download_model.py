"""Download one curated Diffusers model so the app can use it offline."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from model_catalog import DEFAULT_MODEL, MODEL_ALLOW_PATTERNS, MODEL_CATALOG

parser = argparse.ArgumentParser(description="Download a supported image-generation model.")
parser.add_argument(
    "--model",
    choices=sorted(MODEL_CATALOG),
    default=os.environ.get("MODEL_KEY", DEFAULT_MODEL),
    help="Model ID from the app's model dropdown (default: stable-diffusion-v1-5).",
)
parser.add_argument(
    "--list",
    action="store_true",
    help="List supported models and exit.",
)
args = parser.parse_args()

if args.list:
    for key, item in MODEL_CATALOG.items():
        print(f"{key:24} {item['size']:26} {item['name']} — {item['description']}")
    raise SystemExit(0)

selected = MODEL_CATALOG[args.model]
model_root = Path(os.environ.get("MODEL_ROOT", ROOT / "models")).expanduser()
local_dir = model_root / args.model
if args.model == DEFAULT_MODEL and os.environ.get("MODEL_DIR"):
    local_dir = Path(os.environ["MODEL_DIR"]).expanduser()

if (local_dir / "model_index.json").is_file():
    print(f"{selected['name']} already appears installed at {local_dir}.")
    print("If loading fails, remove that model folder and download it again.")
    raise SystemExit(0)

local_dir.mkdir(parents=True, exist_ok=True)
repo_id = os.environ.get("MODEL_REPO") if args.model == DEFAULT_MODEL else None
repo_id = repo_id or selected["repo_id"]
print(f"Downloading {selected['name']} ({selected['size']}) from {repo_id} to {local_dir}.")
print("Only standard Diffusers pipeline components are requested; progress is shown below.")

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    allow_patterns=MODEL_ALLOW_PATTERNS,
)

if not (local_dir / "model_index.json").is_file():
    raise SystemExit(
        "The repository did not provide a Diffusers model_index.json. "
        "Choose a catalog model or check MODEL_REPO compatibility."
    )

print(f"Download complete: {local_dir}")
print("Launch app.py and choose this model from the AI model menu. It can now run offline.")
