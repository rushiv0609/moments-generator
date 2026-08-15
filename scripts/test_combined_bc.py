"""Benchmark combined B+C: Apple Silicon Native ImageIO + Asynchronous Producer-Consumer Pipeline with TQDM."""
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
from tqdm import tqdm

def test_combined_pipeline(folder: str, count: int = 200, num_workers: int = 12):
    print("=" * 60)
    print(" COMBINED B + C BENCHMARK (NATIVE IMAGEIO + PIPELINED GPU)")
    print("=" * 60)
    
    root = Path(folder)
    heic_files = list(root.rglob("*.HEIC")) + list(root.rglob("*.heic"))
    files = heic_files[:count]
    print(f"Dataset: {len(files)} raw iPhone HEIC photos")
    print(f"Decoding workers: {num_workers} threads (Apple Native ImageIO)")

    model_name = "google/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    frame_queue = queue.Queue(maxsize=64)
    options = {
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceThumbnailMaxPixelSize: 224,
        Quartz.kCGImageSourceShouldCache: False,
    }

    pbar_decode = tqdm(total=len(files), desc="📸 ImageIO Decoding", unit="img")

    def decode_worker(chunk):
        for f in chunk:
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
                        frame_queue.put(img)
            except Exception:
                pass
            finally:
                pbar_decode.update(1)

    t0 = time.perf_counter()

    threads = []
    chunk_size = (len(files) + num_workers - 1) // num_workers
    for i in range(0, len(files), chunk_size):
        c = files[i : i + chunk_size]
        t = threading.Thread(target=decode_worker, args=(c,))
        t.start()
        threads.append(t)

    processed_count = 0
    pbar_gpu = tqdm(total=len(files), desc="⚡ GPU Forward (MPS)", unit="img")
    batch = []

    while True:
        try:
            img = frame_queue.get(timeout=0.1)
            batch.append(img)
            if len(batch) >= 32:
                inputs = processor(images=batch, return_tensors="pt").to("mps")
                with torch.inference_mode():
                    _ = model.get_image_features(**inputs)
                torch.mps.synchronize()
                processed_count += len(batch)
                pbar_gpu.update(len(batch))
                batch = []
        except queue.Empty:
            if all(not t.is_alive() for t in threads) and frame_queue.empty():
                break

    if batch:
        inputs = processor(images=batch, return_tensors="pt").to("mps")
        with torch.inference_mode():
            _ = model.get_image_features(**inputs)
        torch.mps.synchronize()
        processed_count += len(batch)
        pbar_gpu.update(len(batch))

    for t in threads:
        t.join()
    pbar_decode.close()
    pbar_gpu.close()

    total_time = time.perf_counter() - t0
    total_imgs = processed_count
    fps = total_imgs / total_time if total_time > 0 else 0

    print(f"\n✓ Successfully processed {total_imgs} HEIC images in {total_time:.2f} seconds")
    print(f"⚡ Throughput: {fps:.1f} images/second (vs 10.0-13.4 img/s previously)")
    print(f"🚀 Speedup   : ~{fps / 13.4:.2f}x faster end-to-end!")
    print("=" * 60)

if __name__ == "__main__":
    test_combined_pipeline("/Users/rushivyas/Pictures/pin-bhabha/", count=200, num_workers=12)
