# Local AI Moments Generator

An offline, local AI-powered moments video generator for Apple Silicon (M5 / M5 Pro / M-series). Ingests photos and videos (up to 20 GB), understands visual semantics using SigLIP 2 embeddings, and stitches the best highlights matching a natural-language prompt without generative video synthesis.

## Key Features

- **100% Local & Private** — All processing happens on-device. No cloud uploads, no API keys.
- **Apple Silicon Optimised** — Uses Apple MLX with JIT-compiled kernel fusion for 30% faster GPU inference vs PyTorch MPS.
- **Multi-Modal Understanding** — Text, photos (HEIC/JPEG/PNG), and videos processed through SigLIP 2 vision-language model.
- **Hardware-Accelerated Decoding** — Apple ImageIO (photos) and AVFoundation/OpenCV (videos) for native hardware decompression.
- **Smart Curation** — Temporal time-bucketing prevents one event from dominating. Burst deduplication removes near-identical shots.
- **Checkpoint & Resume** — SQLite-backed manifest tracks processing state per file. Crash-safe with automatic resume.

## Performance

Benchmarked on Apple Silicon M5 Pro with 1,000 real iPhone photos (48MP HEIC) and 25 video clips:

| Metric | Value |
|---|---|
| Photo ingestion throughput | **19.7 photos/sec** (50s for 1,000 photos) |
| Video frame extraction | **16.3 frames/sec** (17s for 283 frames) |
| Text search latency | **1.02 ms/query** (968 queries/sec) |
| GPU throughput (batched) | **470 img/sec** (MLX FP16, batch 32) |
| Model RAM (FP16) | ~860 MB |
| Model RAM (8-bit quantized) | ~440 MB (50% less, zero accuracy loss) |

> The pipeline is I/O-bound (92.5% of time spent decoding HEIC files). GPU utilisation is <5%, keeping thermals cool and fans silent.

## Quick Start

```bash
# 1. Start the system (installs dependencies, starts Qdrant, runs FastAPI)
./start.sh

# 2. Open your browser
open http://localhost:8000/ui/
```

## Architecture

```
Media Files (HEIC/JPEG/MOV) → ImageIO/AVFoundation Decode (12 threads)
    → Bounded Queue → MLX GPU Consumer (Transform + SigLIP 2 Embed + L2 Norm)
    → Qdrant Vector DB → Semantic Search → FFmpeg Render → MP4
```

**Dual-Engine Strategy:** Apple MLX (primary, +30% throughput) with PyTorch MPS (fallback). Strategy pattern allows hot-swapping at startup.

## Benchmarking

A comprehensive benchmark suite is included for performance validation:

```bash
# Run the master benchmark (all modalities, all engines)
python scripts/benchmark_all_modalities.py

# Individual benchmarks
python scripts/compare_mps_vs_cpu.py        # MPS vs CPU baseline
python scripts/compare_mlx_vs_pytorch.py    # MLX vs PyTorch (1,000 real photos)
python scripts/compare_mlx_8bit.py          # 8-bit quantization accuracy + perf
```

Reports are saved to `data/benchmarks/` in Markdown + JSON format. See [Benchmarking Guide](docs/benchmarking.md) for details.

## Project Structure

```
moments-generator/
├── app/                    # FastAPI application (Milestone 2+)
├── docs/                   # Architecture, requirements, benchmarking docs
│   ├── technical-architecture.md
│   ├── product-requirements.md
│   ├── benchmarking.md
│   └── appendix-learnings.md
├── scripts/                # Benchmark scripts, model utilities
│   ├── benchmark_all_modalities.py
│   ├── mlx_siglip2.py      # Native MLX SigLIP 2 implementation
│   └── telemetry_utils.py  # Shared benchmark reporting
├── data/
│   ├── benchmarks/         # Generated benchmark reports
│   └── manifests/          # SQLite manifest databases
├── models/                 # Cached model weights
└── exports/                # Rendered MP4 outputs
```

## Documentation

- [Technical Architecture](docs/technical-architecture.md) — Full system design with empirical data
- [Product Requirements](docs/product-requirements.md) — Feature specifications
- [Project Setup & Structure](docs/project-setup.md) — Development environment
- [Benchmarking Guide](docs/benchmarking.md) — Benchmark suite documentation
- [Appendix: Learnings](docs/appendix-learnings.md) — Project retrospective & technical learnings

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Framework** | Apple MLX (primary) / PyTorch MPS (fallback) |
| **Vision-Language Model** | SigLIP 2 (`google/siglip2-base-patch16-224`) |
| **Photo Decode** | Apple ImageIO via PyObjC (hardware-accelerated) |
| **Video Decode** | OpenCV + AVFoundation (hardware-accelerated) |
| **Vector Database** | Qdrant (Docker) |
| **API Server** | FastAPI + Uvicorn |
| **Video Rendering** | FFmpeg with VideoToolbox hardware encoder |
| **Package Manager** | uv (Astral) |
