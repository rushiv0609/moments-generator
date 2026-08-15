"""Direct performance comparison: MPS (Apple Silicon GPU) vs CPU on SigLIP 2.
Outputs formatted results to stdout and saves report file to data/benchmarks/.
"""
import time
import argparse
import sys
from pathlib import Path

# Allow importing from scripts or workspace root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

from telemetry_utils import get_current_ram_mb, get_mps_ram_mb, save_benchmark_report

def benchmark_device(device_name: str, model_name: str, num_samples: int = 256, batch_sizes=[1, 16, 32, 64]):
    print(f"\n{'=' * 30} Testing on: {device_name.upper()} {'=' * 30}")
    
    dtype = torch.float16 if device_name == "mps" else torch.float32
    
    t_load0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device_name)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    load_time = time.perf_counter() - t_load0
    print(f"✓ Model loaded in {load_time:.2f}s (dtype: {dtype})")

    # 1. Text Embedding Latency (Single search query)
    prompt = "a serene mountain lake with reflection of trees"
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device_name)
    
    # Warmup
    with torch.inference_mode():
        _ = model.get_text_features(**text_inputs)
    if device_name == "mps":
        torch.mps.synchronize()

    # Benchmark text latency (100 runs)
    t_text0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(100):
            _ = model.get_text_features(**text_inputs)
        if device_name == "mps":
            torch.mps.synchronize()
    text_latency_ms = ((time.perf_counter() - t_text0) / 100) * 1000
    print(f"✓ Text search query latency: {text_latency_ms:.2f} ms ({1000 / text_latency_ms:.1f} queries/sec)")

    # 2. Image Forward Pass across different batch sizes
    results = {}
    for bs in batch_sizes:
        dummy_images = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(bs)]
        inputs = processor(images=dummy_images, return_tensors="pt").to(device_name)
        if device_name != "mps":
            inputs = {k: v.to(torch.float32) if v.dtype == torch.float16 else v for k, v in inputs.items()}

        # Warmup
        with torch.inference_mode():
            _ = model.get_image_features(**inputs)
        if device_name == "mps":
            torch.mps.synchronize()

        passes = max(4, num_samples // bs)
        total_images = passes * bs
        
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(passes):
                _ = model.get_image_features(**inputs)
            if device_name == "mps":
                torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        
        fps = total_images / elapsed
        latency_per_img_ms = (elapsed / total_images) * 1000
        results[bs] = {"fps": round(fps, 1), "latency_ms": round(latency_per_img_ms, 2)}
        print(f"  • Batch size {bs:2d} : {fps:6.1f} images/sec ({latency_per_img_ms:5.2f} ms/image)")

    ram_mb = get_current_ram_mb()
    mps_mb = get_mps_ram_mb()

    return {
        "device": device_name,
        "dtype": str(dtype),
        "load_time_sec": round(load_time, 2),
        "text_latency_ms": round(text_latency_ms, 2),
        "text_throughput_qps": round(1000 / text_latency_ms, 1),
        "batch_results": results,
        "peak_ram_mb": round(ram_mb, 1),
        "mps_allocated_mb": round(mps_mb, 1),
    }

def main():
    parser = argparse.ArgumentParser(description="MPS vs CPU Benchmark")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224", help="HuggingFace model ID")
    parser.add_argument("--output", default=None, help="Path to save markdown report file")
    args = parser.parse_args()

    print("=" * 75)
    print(" SIGLIP 2 HARDWARE BENCHMARK: APPLE SILICON MPS (GPU) vs CPU")
    print("=" * 75)
    print(f"Model: {args.model}")

    # 1. Run CPU
    cpu_metrics = benchmark_device("cpu", args.model, num_samples=128, batch_sizes=[1, 16, 32, 64])

    # 2. Run MPS (GPU)
    mps_metrics = benchmark_device("mps", args.model, num_samples=256, batch_sizes=[1, 16, 32, 64])

    # 3. Build Table
    lines = [
        "=" * 80,
        " 📊 SIDE-BY-SIDE HARDWARE COMPARISON TABLE",
        "=" * 80,
        f"{'Metric / Workload':<32} | {'CPU (Apple Silicon)':<20} | {'MPS (Apple GPU)':<20} | {'GPU Gain':<10}",
        "-" * 80,
        f"{'Model Load Time':<32} | {cpu_metrics['load_time_sec']:.2f} seconds          | {mps_metrics['load_time_sec']:.2f} seconds          | -",
        f"{'Text Search Latency (per query)':<32} | {cpu_metrics['text_latency_ms']:.2f} ms             | {mps_metrics['text_latency_ms']:.2f} ms             | {cpu_metrics['text_latency_ms']/mps_metrics['text_latency_ms']:.1f}x Faster",
        f"{'Text Search Throughput':<32} | {cpu_metrics['text_throughput_qps']:.1f} queries/s        | {mps_metrics['text_throughput_qps']:.1f} queries/s        | {mps_metrics['text_throughput_qps']/cpu_metrics['text_throughput_qps']:.1f}x Faster",
        "-" * 80,
    ]
    
    for bs in [1, 16, 32, 64]:
        cpu_fps = cpu_metrics['batch_results'][bs]['fps']
        mps_fps = mps_metrics['batch_results'][bs]['fps']
        speedup = mps_fps / cpu_fps
        lines.append(f"{'Image Throughput (Batch ' + str(bs) + ')':<32} | {cpu_fps:5.1f} img/s          | {mps_fps:5.1f} img/s          | {speedup:.1f}x Faster 🚀")
    
    lines.extend([
        "-" * 80,
        f"{'Peak Process RAM':<32} | {cpu_metrics['peak_ram_mb']:.0f} MB                 | {mps_metrics['peak_ram_mb']:.0f} MB                 | -",
        "=" * 80,
    ])
    
    table_str = "\n".join(lines)
    print("\n" + table_str)

    # Save to file
    save_benchmark_report(
        title="SigLIP 2 Hardware Benchmark: Apple Silicon MPS vs CPU",
        table_str=table_str,
        metrics_dict={"cpu": cpu_metrics, "mps": mps_metrics},
        output_path=args.output,
        default_filename_prefix="mps_vs_cpu_comparison",
    )

if __name__ == "__main__":
    main()
