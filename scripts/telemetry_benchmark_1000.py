"""High-precision telemetry breakdown for SigLIP 2 pipeline on 1,000 real photos.
Measures exact time spent in:
1. Decompression (Apple ImageIO / Disk I/O)
2. Transformation on CPU vs Transformation on GPU (MPS)
3. GPU Neural Network Forward Pass (SigLIP 2 Vision Tower)
4. Vector L2 Normalization

Outputs formatted table and saves report file to data/benchmarks/.
"""
import os
import sys
import time
import queue
import threading
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
import numpy as np
import torch
import Quartz
from Foundation import NSURL
from transformers import AutoModel, AutoProcessor

from telemetry_utils import get_current_ram_mb, get_mps_ram_mb, save_benchmark_report

def load_files(folder_path: str, count: int = 1000):
    root = Path(folder_path)
    exts = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff"}
    all_files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    return all_files[:count]

# =========================================================================
# APPROACH 1: CPU Transformation (HuggingFace AutoProcessor on CPU)
# =========================================================================
def run_approach_1_cpu_transform(files, model, processor, batch_size=64, num_workers=12):
    print("\n" + "=" * 75)
    print(" [APPROACH 1] CPU TRANSFORMATION (ImageIO Decode -> CPU Processor -> GPU NN)")
    print("=" * 75)
    
    frame_queue = queue.Queue(maxsize=128)
    options = {
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceThumbnailMaxPixelSize: 224,
        Quartz.kCGImageSourceShouldCache: False,
    }

    t_decompress_total = [0.0]
    t_cpu_transform_total = [0.0]
    t_gpu_nn_total = [0.0]
    t_gpu_norm_total = [0.0]
    lock = threading.Lock()

    def decode_worker(chunk):
        local_decomp = 0.0
        for f in chunk:
            t0 = time.perf_counter()
            try:
                url = NSURL.fileURLWithPath_(str(f))
                src = Quartz.CGImageSourceCreateWithURL(url, None)
                if src:
                    cg_img = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, options)
                    if cg_img:
                        w = Quartz.CGImageGetWidth(cg_img)
                        h = Quartz.CGImageGetHeight(cg_img)
                        dp = Quartz.CGImageGetDataProvider(cg_img)
                        data = Quartz.CGDataProviderCopyData(dp)
                        img = Image.frombuffer("RGBA", (w, h), bytes(data), "raw", "RGBA", 0, 1).convert("RGB")
                        local_decomp += (time.perf_counter() - t0)
                        frame_queue.put(img)
                        continue
                with Image.open(f) as img:
                    img = img.convert("RGB")
                    img.thumbnail((224, 224))
                    local_decomp += (time.perf_counter() - t0)
                    frame_queue.put(img)
            except Exception:
                pass
        with lock:
            t_decompress_total[0] += local_decomp

    t_wall_start = time.perf_counter()
    threads = []
    chunk_size = (len(files) + num_workers - 1) // num_workers
    for i in range(0, len(files), chunk_size):
        t = threading.Thread(target=decode_worker, args=(files[i : i + chunk_size],))
        t.start()
        threads.append(t)

    processed_count = [0]

    def gpu_consumer():
        batch = []
        while True:
            try:
                img = frame_queue.get(timeout=0.2)
                batch.append(img)
                if len(batch) >= batch_size:
                    t_tx0 = time.perf_counter()
                    inputs = processor(images=batch, return_tensors="pt").to("mps")
                    t_cpu_transform_total[0] += (time.perf_counter() - t_tx0)

                    t_nn0 = time.perf_counter()
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
                    torch.mps.synchronize()
                    t_gpu_nn_total[0] += (time.perf_counter() - t_nn0)

                    t_norm0 = time.perf_counter()
                    with torch.inference_mode():
                        _ = emb / emb.norm(dim=-1, keepdim=True)
                    torch.mps.synchronize()
                    t_gpu_norm_total[0] += (time.perf_counter() - t_norm0)

                    processed_count[0] += len(batch)
                    batch = []
            except queue.Empty:
                if all(not t.is_alive() for t in threads) and frame_queue.empty():
                    break

        if batch:
            t_tx0 = time.perf_counter()
            inputs = processor(images=batch, return_tensors="pt").to("mps")
            t_cpu_transform_total[0] += (time.perf_counter() - t_tx0)

            t_nn0 = time.perf_counter()
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                emb = features if torch.is_tensor(features) else features[0]
            torch.mps.synchronize()
            t_gpu_nn_total[0] += (time.perf_counter() - t_nn0)

            t_norm0 = time.perf_counter()
            with torch.inference_mode():
                _ = emb / emb.norm(dim=-1, keepdim=True)
            torch.mps.synchronize()
            t_gpu_norm_total[0] += (time.perf_counter() - t_norm0)
            processed_count[0] += len(batch)

    consumer = threading.Thread(target=gpu_consumer)
    consumer.start()

    for t in threads:
        t.join()
    consumer.join()

    t_wall_total = time.perf_counter() - t_wall_start
    ram_mb = get_current_ram_mb()

    return {
        "approach": "CPU Transformation (AutoProcessor)",
        "wall_time_sec": round(t_wall_total, 2),
        "processed_count": processed_count[0],
        "throughput_fps": round(processed_count[0] / t_wall_total, 1),
        "cpu_decompress_sum_sec": round(t_decompress_total[0], 2),
        "cpu_decompress_effective_wall_sec": round(t_decompress_total[0] / num_workers, 2),
        "transform_time_sec": round(t_cpu_transform_total[0], 2),
        "gpu_nn_forward_time_sec": round(t_gpu_nn_total[0], 2),
        "gpu_norm_time_sec": round(t_gpu_norm_total[0], 3),
        "peak_ram_mb": round(ram_mb, 1),
    }

# =========================================================================
# APPROACH 2: GPU Transformation (Raw bytes -> Direct GPU MPS Transform)
# =========================================================================
def run_approach_2_gpu_transform(files, model, batch_size=64, num_workers=12):
    print("\n" + "=" * 75)
    print(" [APPROACH 2] GPU TRANSFORMATION (ImageIO Decode -> GPU MPS Shader Transform -> GPU NN)")
    print("=" * 75)
    
    frame_queue = queue.Queue(maxsize=128)
    options = {
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceThumbnailMaxPixelSize: 224,
        Quartz.kCGImageSourceShouldCache: False,
    }

    MEAN = torch.tensor([0.5, 0.5, 0.5], device="mps", dtype=torch.float16).view(1, 3, 1, 1)
    STD = torch.tensor([0.5, 0.5, 0.5], device="mps", dtype=torch.float16).view(1, 3, 1, 1)

    t_decompress_total = [0.0]
    t_gpu_transform_total = [0.0]
    t_gpu_nn_total = [0.0]
    t_gpu_norm_total = [0.0]
    lock = threading.Lock()

    def decode_worker(chunk):
        local_decomp = 0.0
        for f in chunk:
            t0 = time.perf_counter()
            try:
                url = NSURL.fileURLWithPath_(str(f))
                src = Quartz.CGImageSourceCreateWithURL(url, None)
                if src:
                    cg_img = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, options)
                    if cg_img:
                        w = Quartz.CGImageGetWidth(cg_img)
                        h = Quartz.CGImageGetHeight(cg_img)
                        dp = Quartz.CGImageGetDataProvider(cg_img)
                        data = Quartz.CGDataProviderCopyData(dp)
                        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
                        local_decomp += (time.perf_counter() - t0)
                        frame_queue.put(arr)
                        continue
                with Image.open(f) as img:
                    img = img.convert("RGB").resize((224, 224))
                    arr = np.array(img, dtype=np.uint8)
                    local_decomp += (time.perf_counter() - t0)
                    frame_queue.put(arr)
            except Exception:
                pass
        with lock:
            t_decompress_total[0] += local_decomp

    t_wall_start = time.perf_counter()
    threads = []
    chunk_size = (len(files) + num_workers - 1) // num_workers
    for i in range(0, len(files), chunk_size):
        t = threading.Thread(target=decode_worker, args=(files[i : i + chunk_size],))
        t.start()
        threads.append(t)

    processed_count = [0]

    def gpu_consumer():
        batch_arrays = []
        while True:
            try:
                arr = frame_queue.get(timeout=0.2)
                if arr.shape[0] != 224 or arr.shape[1] != 224:
                    pad_h = max(0, 224 - arr.shape[0])
                    pad_w = max(0, 224 - arr.shape[1])
                    if pad_h > 0 or pad_w > 0:
                        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                    arr = arr[:224, :224]
                batch_arrays.append(arr)

                if len(batch_arrays) >= batch_size:
                    t_tx0 = time.perf_counter()
                    batch_np = np.stack(batch_arrays, axis=0)
                    tensor_gpu = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to("mps", non_blocking=True)
                    pixel_values = (tensor_gpu.to(torch.float16) / 255.0 - MEAN) / STD
                    torch.mps.synchronize()
                    t_gpu_transform_total[0] += (time.perf_counter() - t_tx0)

                    t_nn0 = time.perf_counter()
                    with torch.inference_mode():
                        features = model.get_image_features(pixel_values=pixel_values)
                        if hasattr(features, "pooler_output") and features.pooler_output is not None:
                            emb = features.pooler_output
                        elif hasattr(features, "last_hidden_state"):
                            emb = features.last_hidden_state[:, 0]
                        elif torch.is_tensor(features):
                            emb = features
                        else:
                            emb = features[0]
                    torch.mps.synchronize()
                    t_gpu_nn_total[0] += (time.perf_counter() - t_nn0)

                    t_norm0 = time.perf_counter()
                    with torch.inference_mode():
                        _ = emb / emb.norm(dim=-1, keepdim=True)
                    torch.mps.synchronize()
                    t_gpu_norm_total[0] += (time.perf_counter() - t_norm0)

                    processed_count[0] += len(batch_arrays)
                    batch_arrays = []
            except queue.Empty:
                if all(not t.is_alive() for t in threads) and frame_queue.empty():
                    break

        if batch_arrays:
            t_tx0 = time.perf_counter()
            batch_np = np.stack(batch_arrays, axis=0)
            tensor_gpu = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to("mps", non_blocking=True)
            pixel_values = (tensor_gpu.to(torch.float16) / 255.0 - MEAN) / STD
            torch.mps.synchronize()
            t_gpu_transform_total[0] += (time.perf_counter() - t_tx0)

            t_nn0 = time.perf_counter()
            with torch.inference_mode():
                features = model.get_image_features(pixel_values=pixel_values)
                emb = features if torch.is_tensor(features) else features[0]
            torch.mps.synchronize()
            t_gpu_nn_total[0] += (time.perf_counter() - t_nn0)

            t_norm0 = time.perf_counter()
            with torch.inference_mode():
                _ = emb / emb.norm(dim=-1, keepdim=True)
            torch.mps.synchronize()
            t_gpu_norm_total[0] += (time.perf_counter() - t_norm0)
            processed_count[0] += len(batch_arrays)

    consumer = threading.Thread(target=gpu_consumer)
    consumer.start()

    for t in threads:
        t.join()
    consumer.join()

    t_wall_total = time.perf_counter() - t_wall_start
    ram_mb = get_current_ram_mb()

    return {
        "approach": "GPU Transformation (Metal Shaders)",
        "wall_time_sec": round(t_wall_total, 2),
        "processed_count": processed_count[0],
        "throughput_fps": round(processed_count[0] / t_wall_total, 1),
        "cpu_decompress_sum_sec": round(t_decompress_total[0], 2),
        "cpu_decompress_effective_wall_sec": round(t_decompress_total[0] / num_workers, 2),
        "transform_time_sec": round(t_gpu_transform_total[0], 2),
        "gpu_nn_forward_time_sec": round(t_gpu_nn_total[0], 2),
        "gpu_norm_time_sec": round(t_gpu_norm_total[0], 3),
        "peak_ram_mb": round(ram_mb, 1),
    }

def main():
    parser = argparse.ArgumentParser(description="Detailed Telemetry Benchmark on 1000 Photos")
    parser.add_argument("--folder", default="/Users/rushivyas/Pictures/pin-bhabha/", help="Folder of test images")
    parser.add_argument("--count", type=int, default=1000, help="Number of files to benchmark")
    parser.add_argument("--output", default=None, help="Output file path for markdown report")
    args = parser.parse_args()

    files = load_files(args.folder, args.count)
    print(f"Loaded {len(files)} test files from {args.folder}")

    model_name = "google/siglip2-base-patch16-224"
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # Run Benchmark 1
    m1 = run_approach_1_cpu_transform(files, model, processor, batch_size=64, num_workers=12)

    # Run Benchmark 2
    m2 = run_approach_2_gpu_transform(files, model, batch_size=64, num_workers=12)

    # Build Comparison Table
    lines = [
        "=" * 80,
        " 📊 DETAILED TELEMETRY & STAGE BREAKDOWN (1,000 PHOTOS)",
        "=" * 80,
        f"{'Pipeline Stage / Metric':<35} | {'Approach 1 (CPU Trans)':<20} | {'Approach 2 (GPU Trans)':<20}",
        "-" * 80,
        f"{'Images Processed':<35} | {m1['processed_count']:<20} | {m2['processed_count']:<20}",
        f"{'1. Decompression Sum (CPU Work)':<35} | {m1['cpu_decompress_sum_sec']:.2f} seconds       | {m2['cpu_decompress_sum_sec']:.2f} seconds",
        f"{'   Decompression Effective Wall':<35} | {m1['cpu_decompress_effective_wall_sec']:.2f} seconds       | {m2['cpu_decompress_effective_wall_sec']:.2f} seconds",
        f"{'2. Transformation Time':<35} | {m1['transform_time_sec']:.2f}s (CPU Python)  | {m2['transform_time_sec']:.2f}s (GPU Metal) ⚡",
        f"{'3. GPU Neural Net Forward Pass':<35} | {m1['gpu_nn_forward_time_sec']:.2f} seconds        | {m2['gpu_nn_forward_time_sec']:.2f} seconds",
        f"{'4. GPU L2 Vector Normalization':<35} | {m1['gpu_norm_time_sec']:.3f} seconds       | {m2['gpu_norm_time_sec']:.3f} seconds",
        "-" * 80,
        f"{'TOTAL WALL CLOCK TIME':<35} | {m1['wall_time_sec']:.2f} seconds       | {m2['wall_time_sec']:.2f} seconds",
        f"{'END-TO-END THROUGHPUT':<35} | {m1['throughput_fps']:.1f} img/sec         | {m2['throughput_fps']:.1f} img/sec 🚀",
        f"{'PEAK PROCESS RAM':<35} | {m1['peak_ram_mb']:.0f} MB                 | {m2['peak_ram_mb']:.0f} MB",
        "=" * 80,
    ]
    table_str = "\n".join(lines)
    print("\n" + table_str)

    # Save to file
    save_benchmark_report(
        title="SigLIP 2 Telemetry Breakdown: CPU vs GPU Transformation (1,000 Photos)",
        table_str=table_str,
        metrics_dict={"approach_1_cpu_transform": m1, "approach_2_gpu_transform": m2},
        output_path=args.output,
        default_filename_prefix="telemetry_breakdown_1000",
    )

if __name__ == "__main__":
    main()
