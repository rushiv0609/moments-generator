# Benchmarking Guide

**Project:** Local AI Moments Generator  
**Last Updated:** 2026-08-15

This document describes the benchmark suite used to validate performance characteristics and make data-driven architecture decisions. All benchmark scripts are in [`scripts/`](file:///Users/rushivyas/development/projects/moments-generator/scripts/) and output reports to [`data/benchmarks/`](file:///Users/rushivyas/development/projects/moments-generator/data/benchmarks/).

---

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the master benchmark (covers all modalities + all engines)
python scripts/benchmark_all_modalities.py

# Run individual benchmarks
python scripts/compare_mps_vs_cpu.py        # PyTorch MPS vs CPU baseline
python scripts/compare_mlx_vs_pytorch.py    # MLX vs PyTorch on 1,000 real photos
python scripts/compare_mlx_8bit.py          # FP16 vs 8-bit quantized accuracy + perf
python scripts/stress_test_embedder.py      # GPU stress test (sustained load)
```

---

## Benchmark Suite

### 1. `compare_mps_vs_cpu.py` — Hardware Baseline

**Purpose:** Establishes the baseline GPU acceleration gain by comparing Apple Silicon MPS (GPU) against CPU-only inference.

**What it measures:**
- Model load time (CPU vs MPS)
- Text search latency and throughput
- Image embedding throughput at batch sizes 1, 16, 32, 64
- Peak process RAM

**Key finding:** MPS is **10-11x faster** than CPU for image embedding, **5.5x faster** for text search.

**Output:** `data/benchmarks/mps_vs_cpu_comparison_<timestamp>.md`

---

### 2. `compare_mlx_vs_pytorch.py` — MLX vs PyTorch on Real Data

**Purpose:** Head-to-head comparison of Apple MLX (native) vs PyTorch MPS (via HuggingFace transformers) on 1,000 real photos from the user's corpus.

**What it measures:**
- Accuracy: Cosine similarity between MLX and PyTorch embeddings (should be 1.000000)
- Pure GPU forward pass throughput at multiple batch sizes
- Real-world 1,000-photo ingestion wall time
- Active GPU compute time (excluding I/O)
- Peak process RAM

**Key finding:** MLX is **25-32% faster** at batch≥16 thanks to `@mx.compile` kernel fusion. Wall-clock time is identical (~50s) because the pipeline is I/O-bound.

**Output:** `data/benchmarks/mlx_vs_pytorch_comparison_<timestamp>.md`

---

### 3. `compare_mlx_8bit.py` — 8-Bit Quantization Analysis

**Purpose:** Validates that 8-bit quantized weights (`mlx-community/siglip2-base-patch16-224-8bit`) preserve embedding quality while reducing memory.

**What it measures:**
- Cosine similarity between FP16 and 8-bit embeddings (accuracy loss)
- Mean Squared Error between embedding vectors
- GPU throughput at multiple batch sizes (FP16 vs 8-bit)
- 1,000-photo ingestion comparison
- Model RAM usage

**Key finding:** **Zero accuracy loss** (cosine similarity 1.000000), **50% RAM reduction** (860 MB → 440 MB), negligible throughput impact.

**Output:** `data/benchmarks/compare_mlx_8bit_results_<timestamp>.md`

---

### 4. `benchmark_all_modalities.py` — Master Multi-Modal Benchmark

**Purpose:** Comprehensive benchmark covering all three modalities (text, photos, videos) across all engines (PyTorch MPS, MLX FP16, MLX 8-bit) with kernel fusion enabled.

**What it measures:**
- Text search: Latency and throughput (queries/sec)
- Photo ingestion: GPU forward pass at batch 1/16/32/64, real-world 1,000-photo wall time
- Video ingestion: Frame extraction + embedding for 25 real video clips (283 frames)
- Peak process RAM for each engine

**Key finding:** MLX FP16 with `@mx.compile` is the optimal engine. Text search at **968 queries/sec**, photo ingestion at **19.7 photos/sec** (I/O-bound), video at **16.3 frames/sec**.

**Output:** `data/benchmarks/master_multimodal_benchmark_<timestamp>.md`

---

### 5. `stress_test_embedder.py` — GPU Sustained Load Test

**Purpose:** Tests GPU stability under sustained load to verify no thermal throttling, memory leaks, or OOM crashes.

**What it measures:**
- Throughput over 5,000+ consecutive embeddings
- Throughput stability (variance over time)
- Memory growth (leak detection)
- GPU error rates

---

## Shared Infrastructure

### `telemetry_utils.py` — Report Generation

All benchmark scripts use the shared [`telemetry_utils.py`](file:///Users/rushivyas/development/projects/moments-generator/scripts/telemetry_utils.py) module for standardized output:

- **Markdown reports** with formatted comparison tables (human-readable)
- **JSON telemetry blocks** embedded in the Markdown (machine-parseable)
- **Timestamped filenames** for historical tracking
- **Automatic output** to `data/benchmarks/`

### `mlx_siglip2.py` — Native MLX Implementation

The [`mlx_siglip2.py`](file:///Users/rushivyas/development/projects/moments-generator/scripts/mlx_siglip2.py) module provides a native Apple MLX implementation of SigLIP 2 with:

- `@mx.compile` kernel fusion for the forward pass
- 8-bit quantized model loading from `mlx-community/siglip2-base-patch16-224-8bit`
- Vision and text encoding with L2 normalisation

---

## Output Format

All benchmarks produce Markdown files with embedded JSON telemetry:

```markdown
# Benchmark Title

**Generated at**: `<timestamp>`

## Benchmark Results

(Formatted comparison tables)

## Raw Telemetry Data (JSON)

```json
{
  "metric_1": value,
  "metric_2": value
}
```
```

Reports are saved to `data/benchmarks/` with timestamped filenames for historical tracking.

---

## Hardware Reference

All benchmarks in this project were collected on:

| Property | Value |
|---|---|
| **Machine** | Apple Silicon M5 Pro |
| **OS** | macOS (latest) |
| **Python** | 3.12 |
| **PyTorch** | 2.13.0 |
| **MLX** | Latest (via pip) |
| **Storage** | Internal SSD |
| **Test corpus** | ~1,000 iPhone photos (48MP HEIC + JPEG), 25 video clips |

---

## Adding New Benchmarks

When adding a new benchmark script:

1. Import `telemetry_utils` for consistent report formatting
2. Output to `data/benchmarks/` with a timestamped filename
3. Include both formatted tables AND raw JSON telemetry
4. Add a section to this document describing the benchmark
5. Run on real data (not just synthetic tensors) when possible
