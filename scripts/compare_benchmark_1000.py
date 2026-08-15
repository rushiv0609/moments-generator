"""Side-by-side benchmark comparison on 1,000 real photos:
Old Baseline (Sequential/Standard PIL) vs Combined B+C (Native Apple ImageIO + Async Producer-Consumer Queue).
"""
import os
import sys
import time
import queue
import threading
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import Quartz
from Foundation import NSURL
from transformers import AutoModel, AutoProcessor

def run_1000_photo_benchmark(folder_path: str = "/Users/rushivyas/Pictures/pin-bhabha/", count: int = 1000):
    print("=" * 75)
    print(" 1,000 PHOTOS EMBEDDING BENCHMARK: OLD vs COMBINED B+C PIPELINE")
    print("=" * 75)
    print(f"Target folder : {folder_path}")
    print(f"Sample count  : {count} photos")

    # 1. Discover all files
    root = Path(folder_path)
    exts = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff"}
    all_files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    files = all_files[:count]
    
    ext_counts = {}
    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print(f"Selected {len(files)} files: {ext_counts}\n")

    # 2. Load model & processor
    model_name = "google/siglip2-base-patch16-224"
    print("Loading SigLIP 2 onto Apple Silicon GPU (MPS)...")
    t_model = time.perf_counter()
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"✓ Model loaded in {time.perf_counter() - t_model:.2f}s\n")

    # 3. Warm-up GPU & Measure Pure GPU Inference
    print("--- [1] PURE GPU INFERENCE BENCHMARK ---")
    batch_size = 64
    dummy_batch = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(batch_size)]
    dummy_inputs = processor(images=dummy_batch, return_tensors="pt").to("mps")
    
    # Warmup
    with torch.inference_mode():
        _ = model.get_image_features(**dummy_inputs)
    torch.mps.synchronize()

    # Measure 1000 passes on GPU
    total_batches = (len(files) + batch_size - 1) // batch_size
    t_gpu_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(total_batches):
            _ = model.get_image_features(**dummy_inputs)
        torch.mps.synchronize()
    t_gpu_total = time.perf_counter() - t_gpu_start
    pure_gpu_fps = len(files) / t_gpu_total
    print(f"✓ Pure GPU Forward-Pass Time : {t_gpu_total:.2f} seconds (for {len(files)} images)")
    print(f"⚡ Pure GPU Inference Speed   : {pure_gpu_fps:.1f} images/second\n")

    # 4. Measure Combined B+C End-to-End Pipeline
    print("--- [2] COMBINED B+C END-TO-END PIPELINE (ImageIO + Async Queue) ---")
    frame_queue = queue.Queue(maxsize=128)
    options = {
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceThumbnailMaxPixelSize: 224,
        Quartz.kCGImageSourceShouldCache: False,
    }

    def decode_worker(chunk):
        for f in chunk:
            try:
                # Fast path for HEIC/JPG via Apple ImageIO hardware thumbnail
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
                        frame_queue.put(img)
                        continue
                # Fallback to PIL if ImageIO returns None
                with Image.open(f) as img:
                    img = img.convert("RGB")
                    img.thumbnail((224, 224))
                    frame_queue.put(img)
            except Exception:
                pass

    # Launch 12 Producer Threads
    num_workers = 12
    threads = []
    chunk_size = (len(files) + num_workers - 1) // num_workers
    t_e2e_start = time.perf_counter()

    for i in range(0, len(files), chunk_size):
        chunk = files[i : i + chunk_size]
        t = threading.Thread(target=decode_worker, args=(chunk,))
        t.start()
        threads.append(t)

    # Launch GPU Consumer Thread
    processed_count = [0]
    gpu_active_time = [0.0]

    def gpu_consumer():
        batch = []
        while True:
            try:
                img = frame_queue.get(timeout=0.2)
                batch.append(img)
                if len(batch) >= batch_size:
                    t_b0 = time.perf_counter()
                    inputs = processor(images=batch, return_tensors="pt").to("mps")
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
                    gpu_active_time[0] += (time.perf_counter() - t_b0)
                    processed_count[0] += len(batch)
                    
                    pct = (processed_count[0] / len(files)) * 100
                    print(f"  [{pct:5.1f}%] Embedded {processed_count[0]}/{len(files)} images...")
                    batch = []
            except queue.Empty:
                if all(not t.is_alive() for t in threads) and frame_queue.empty():
                    break

        if batch:
            t_b0 = time.perf_counter()
            inputs = processor(images=batch, return_tensors="pt").to("mps")
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                _ = features / features.norm(dim=-1, keepdim=True) if torch.is_tensor(features) else features
            torch.mps.synchronize()
            gpu_active_time[0] += (time.perf_counter() - t_b0)
            processed_count[0] += len(batch)
            pct = (processed_count[0] / len(files)) * 100
            print(f"  [{pct:5.1f}%] Embedded {processed_count[0]}/{len(files)} images...")

    consumer = threading.Thread(target=gpu_consumer)
    consumer.start()

    for t in threads:
        t.join()
    consumer.join()

    t_e2e_total = time.perf_counter() - t_e2e_start
    total_imgs = processed_count[0]
    e2e_fps = total_imgs / t_e2e_total

    # Peak memory
    rss_mb = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
    mem_str = f"{int(rss_mb) // 1024} MB" if rss_mb else "N/A"

    # Previous baseline numbers for comparison
    prev_runtime = 74.23
    prev_fps = 13.4
    prev_gpu_fps = 350.0

    print("\n" + "=" * 75)
    print(" FINAL COMPARISON RESULTS (1,000 REAL PHOTOS)")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Previous Baseline':<20} | {'Combined B+C':<20}")
    print("-" * 75)
    print(f"{'Overall Runtime':<30} | {prev_runtime:.2f} seconds          | {t_e2e_total:.2f} seconds")
    print(f"{'End-to-End Speed':<30} | {prev_fps:.1f} images/sec         | {e2e_fps:.1f} images/sec")
    print(f"{'Pure GPU Forward Speed':<30} | {prev_gpu_fps:.1f} images/sec       | {pure_gpu_fps:.1f} images/sec")
    print(f"{'Active GPU Compute Time':<30} | {'N/A':<20} | {gpu_active_time[0]:.2f} seconds")
    print(f"{'Peak Process RAM':<30} | {'5,208 MB':<20} | {mem_str:<20}")
    print(f"{'Overall Performance Gain':<30} | {'1.0x (baseline)':<20} | {prev_runtime / t_e2e_total:.2f}x Faster! 🚀")
    print("=" * 75)

if __name__ == "__main__":
    run_1000_photo_benchmark()
