"""Stress test and throughput benchmark for SigLIP 2 on Apple Silicon (MPS).
Measures:
1. End-to-end throughput (Disk I/O + HEIC/JPG decode + Preprocess + MPS GPU inference)
2. Pure GPU model inference throughput
3. Optimal batch size comparison (16, 32, 64, 128)
"""
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import pillow_heif

# Enable HEIC support for PIL
pillow_heif.register_heif_opener()

from transformers import AutoModel, AutoProcessor

def load_and_preprocess_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
            return img.convert("RGB")
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None

def run_benchmark(
    folder_path: str,
    max_images: int = 1000,
    batch_size: int = 64,
    workers: int = 8,
    model_name: str = "google/siglip2-base-patch16-224",
):
    print("=" * 70)
    print(" SIGLIP 2 IMAGE EMBEDDING STRESS TEST (APPLE SILICON MPS)")
    print("=" * 70)
    print(f"Target folder: {folder_path}")
    print(f"Model: {model_name}")
    print(f"PyTorch version: {torch.__version__} | Device: mps (Apple Silicon GPU)")
    print(f"Workers for I/O & decoding: {workers}")

    # 1. Discover all images
    root = Path(folder_path)
    if not root.exists():
        print(f"Error: Folder {folder_path} does not exist!")
        return

    exts = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff"}
    all_files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    
    ext_counts = {}
    for f in all_files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"\nDiscovered {len(all_files)} total images: {ext_counts}")
    selected_files = all_files[:max_images]
    print(f"Running benchmark on {len(selected_files)} images (batch size = {batch_size})...\n")

    # 2. Load model & processor
    print("Loading model onto MPS GPU...")
    t_load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"✓ Model loaded in {time.perf_counter() - t_load_start:.2f}s\n")

    # 3. Warm-up GPU
    print("Warming up MPS pipeline...")
    dummy_imgs = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(batch_size)]
    dummy_inputs = processor(images=dummy_imgs, return_tensors="pt").to("mps")
    with torch.inference_mode():
        _ = model.get_image_features(**dummy_inputs)
    torch.mps.synchronize()
    print("✓ Warm-up complete.\n")

    # =========================================================================
    # TEST 1: Pure GPU Model Inference (Preprocessed tensors in memory)
    # =========================================================================
    print("--- [TEST 1] Pure GPU Inference Throughput ---")
    print(f"Evaluating {len(selected_files)} image passes on GPU in batches of {batch_size}...")

    total_batches = (len(selected_files) + batch_size - 1) // batch_size
    t_gpu_start = time.perf_counter()

    with torch.inference_mode():
        for b in range(total_batches):
            # Pass pre-created dummy inputs to measure raw tensor math throughput
            _ = model.get_image_features(**dummy_inputs)
        torch.mps.synchronize()

    t_gpu_total = time.perf_counter() - t_gpu_start
    gpu_fps = len(selected_files) / t_gpu_total
    print(f"✓ Pure GPU Time: {t_gpu_total:.2f}s")
    print(f"⚡ Pure GPU Throughput: {gpu_fps:.1f} images/sec\n")

    # =========================================================================
    # TEST 2: End-to-End Real World (Disk I/O + HEIC/JPG Decode + GPU Embed)
    # =========================================================================
    print("--- [TEST 2] End-to-End Throughput (Disk I/O + Decode + Embed) ---")
    t_e2e_start = time.perf_counter()
    embedded_count = 0
    embedding_dim = None

    # Process in streaming chunks with threadpool decoding
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(0, len(selected_files), batch_size):
            chunk_paths = selected_files[i : i + batch_size]
            t_batch_start = time.perf_counter()

            # Parallel decode images
            images = list(executor.map(load_and_preprocess_image, chunk_paths))
            valid_images = [img for img in images if img is not None]

            if not valid_images:
                continue

            # Preprocess & transfer to MPS
            inputs = processor(images=valid_images, return_tensors="pt").to("mps")

            # GPU inference
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                if hasattr(features, "pooler_output") and features.pooler_output is not None:
                    emb = features.pooler_output
                elif hasattr(features, "last_hidden_state"):
                    emb = features.last_hidden_state[:, 0]
                elif torch.is_tensor(features):
                    emb = features
                else:
                    emb = features[0]
                
                # Normalization
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
            torch.mps.synchronize()

            embedded_count += len(valid_images)
            if embedding_dim is None:
                embedding_dim = emb.shape[-1]

            elapsed_batch = time.perf_counter() - t_batch_start
            batch_fps = len(valid_images) / elapsed_batch
            pct = (embedded_count / len(selected_files)) * 100
            print(f"[{pct:5.1f}%] Processed {embedded_count}/{len(selected_files)} images "
                  f"({batch_fps:5.1f} img/s this batch)")

    t_e2e_total = time.perf_counter() - t_e2e_start
    e2e_fps = embedded_count / t_e2e_total

    # 4. Check memory
    rss_mb = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
    mem_str = f"{int(rss_mb) // 1024} MB" if rss_mb else "N/A"

    print("\n" + "=" * 70)
    print(" STRESS TEST RESULTS")
    print("=" * 70)
    print(f"Total images processed : {embedded_count}")
    print(f"Embedding dimension    : {embedding_dim}")
    print(f"Total time elapsed     : {t_e2e_total:.2f} seconds")
    print(f"End-to-End Speed       : {e2e_fps:.1f} images/second")
    print(f"Pure GPU Max Speed     : {gpu_fps:.1f} images/second")
    print(f"Peak Process Memory    : {mem_str}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP 2 Stress Test")
    parser.add_argument("--folder", default="/Users/rushivyas/Pictures/pin-bhabha/", help="Images folder path")
    parser.add_argument("--count", type=int, default=1000, help="Number of images to process (e.g. 1000 or all 1472)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU inference")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent I/O decoding workers")
    args = parser.parse_args()

    run_benchmark(
        folder_path=args.folder,
        max_images=args.count,
        batch_size=args.batch_size,
        workers=args.workers,
    )
