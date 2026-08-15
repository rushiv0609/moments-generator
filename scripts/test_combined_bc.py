"""Benchmark combined B+C: Apple Silicon Native ImageIO + Asynchronous Producer-Consumer Pipeline."""
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

    SENTINEL = None

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
                        # Fast raw memory buffer to PIL Image
                        img = Image.frombuffer("RGBA", (w, h), bytes(data), "raw", "RGBA", 0, 1).convert("RGB")
                        frame_queue.put(img)
            except Exception as e:
                pass

    # Start timing
    t0 = time.perf_counter()

    # Launch Producer Threads
    threads = []
    chunk_size = (len(files) + num_workers - 1) // num_workers
    for i in range(0, len(files), chunk_size):
        c = files[i : i + chunk_size]
        t = threading.Thread(target=decode_worker, args=(c,))
        t.start()
        threads.append(t)

    # Consumer Thread for GPU
    processed_count = [0]

    def gpu_consumer():
        batch = []
        done_producers = False
        while True:
            try:
                img = frame_queue.get(timeout=0.2)
                batch.append(img)
                if len(batch) >= 32:
                    inputs = processor(images=batch, return_tensors="pt").to("mps")
                    with torch.inference_mode():
                        _ = model.get_image_features(**inputs)
                    torch.mps.synchronize()
                    processed_count[0] += len(batch)
                    batch = []
            except queue.Empty:
                if all(not t.is_alive() for t in threads) and frame_queue.empty():
                    break
        
        # Flush remainder
        if batch:
            inputs = processor(images=batch, return_tensors="pt").to("mps")
            with torch.inference_mode():
                _ = model.get_image_features(**inputs)
            torch.mps.synchronize()
            processed_count[0] += len(batch)

    consumer = threading.Thread(target=gpu_consumer)
    consumer.start()

    for t in threads:
        t.join()
    consumer.join()

    total_time = time.perf_counter() - t0
    total_imgs = processed_count[0]
    fps = total_imgs / total_time

    print(f"\n✓ Successfully processed {total_imgs} HEIC images in {total_time:.2f} seconds")
    print(f"⚡ Throughput: {fps:.1f} images/second (vs 10.0-13.4 img/s previously)")
    print(f"🚀 Speedup   : ~{fps / 13.4:.2f}x faster end-to-end!")
    print("=" * 60)

if __name__ == "__main__":
    test_combined_pipeline("/Users/rushivyas/Pictures/pin-bhabha/", count=200, num_workers=12)
