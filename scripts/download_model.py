"""Download SigLIP 2 model weights and processor to local models directory."""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_name: str = "google/siglip2-base-patch16-224", target_dir: str = "./models"):
    target_path = Path(target_dir) / model_name.replace("/", "--")
    print(f"Downloading {model_name} to {target_path}...")
    target_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
    )
    print(f"✓ Model successfully downloaded to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224", help="HuggingFace model ID")
    parser.add_argument("--dir", default="./models", help="Output directory")
    args = parser.parse_args()
    download_model(args.model, args.dir)
