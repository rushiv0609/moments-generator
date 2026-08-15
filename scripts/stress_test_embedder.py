"""Stress test and throughput benchmark for SigLIP 2 on Apple Silicon (MPS).
Outputs formatted results and saves report file to data/benchmarks/.
"""
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

from transformers import AutoModel, AutoProcessor
from telemetry_utils import get_current_ram_mb, get_mps_ram_mb, save_benchmark_report

def load_and_preprocess_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
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
    output_path: str | None = None,
):
    print("=" * 70)
    print(" SIGLIP 2 IMAGE EMBEDDING STRESS TEST (APPLE SILICON MPS)")
    print("=" * 70)
    print(f"Target folder: {folder_path}")
    print(f"Model: {model_name}")
    print(f"PyTorch version: {torch.__version__} | Device: mps (Apple Silicon GPU)")
    print(f"Workers for I/O & decoding: {workers}")

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

    print("Loading model onto MPS GPU...")
    t_load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    load_sec = time.perf_counter() - t_load_start
    print(f"✓ Model loaded in {load_sec:.2f}s\n")

    print("Warming up MPS pipeline...")
    dummy_imgs = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(batch_size)]
    dummy_inputs = processor(images=dummy_imgs, return_tensors="pt").to("mps")
    with torch.inference_mode():
        _ = model.get_image_features(**dummy_inputs)
    torch.mps.synchronize()
    print("✓ Warm-up complete.\n")

    # TEST 1: Pure GPU Model Inference
    print("--- [TEST 1] Pure GPU Inference Throughput ---")
    total_batches = (len(selected_files) + batch_size - 1) // batch_size
    t_gpu_start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(total_batches):
            _ = model.get_image_features(**dummy_inputs)
        torch.mps.synchronize()

    t_gpu_total = time.perf_counter() - t_gpu_start
    gpu_fps = len(selected_files) / t_gpu_total
    print(f"✓ Pure GPU Time: {t_gpu_total:.2f}s")
    print(f"⚡ Pure GPU Throughput: {gpu_fps:.1f} images/sec\n")

    # TEST 2: End-to-End Real World
    print("--- [TEST 2] End-to-End Throughput (Disk I/O + Decode + Embed) ---")
    t_e2e_start = time.perf_counter()
    embedded_count = 0
    embedding_dim = None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(0, len(selected_files), batch_size):
            chunk_paths = selected_files[i : i + batch_size]
            t_batch_start = time.perf_counter()

            images = list(executor.map(load_and_preprocess_image, chunk_paths))
            valid_images = [img for img in images if img is not None]

            if not valid_images:
                continue

            inputs = processor(images=valid_images, return_tensors="pt").to("mps")

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
                _ = emb / emb.norm(dim=-1, keepdim=True)
                
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
    ram_mb = get_current_ram_mb()
    mps_mb = get_mps_ram_mb()

    lines = [
        "=" * 70,
        " STRESS TEST RESULTS",
        "=" * 70,
        f"Total images processed : {embedded_count}",
        f"Embedding dimension    : {embedding_dim}",
        f"Total time elapsed     : {t_e2e_total:.2f} seconds",
        f"End-to-End Speed       : {e2e_fps:.1f} images/second",
        f"Pure GPU Max Speed     : {gpu_fps:.1f} images/second",
        f"Peak Process RAM       : {ram_mb:.0f} MB",
        f"MPS Allocated Memory   : {mps_mb:.0f} MB",
        "=" * 70,
    ]
    table_str = "\n".join(lines)
    print("\n" + table_str)

    metrics = {
        "dataset": {"folder": folder_path, "total_files": len(all_files), "tested_count": len(selected_files)},
        "load_time_sec": round(load_sec, 2),
        "pure_gpu_throughput_fps": round(gpu_fps, 1),
        "e2e_time_sec": round(t_e2e_total, 2),
        "e2e_throughput_fps": round(e2e_fps, 1),
        "peak_ram_mb": round(ram_mb, 1),
        "mps_allocated_mb": round(mps_mb, 1),
    }

    save_benchmark_report(
        title="SigLIP 2 Image Embedding Stress Test",
        table_str=table_str,
        metrics_dict=metrics,
        output_path=output_path,
        default_filename_prefix="stress_test_results",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP 2 Stress Test")
    parser.add_argument("--folder", default="/Users/rushivyas/Pictures/pin-bhabha/", help="Images folder path")
    parser.add_argument("--count", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU inference")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent I/O decoding workers")
    parser.add_argument("--output", default=None, help="Output markdown report file")
    args = parser.parse_args()

    run_benchmark(
        folder_path=args.folder,
        max_images=args.count,
        batch_size=args.batch_size,
        workers=args.workers,
        output_path=args.output,
    )
