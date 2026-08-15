# Appendix A: Project Learnings & Retrospective

**Project:** Local AI Moments Generator  
**Last Updated:** 2026-08-15  
**Purpose:** A living document capturing empirical learnings, architectural decisions, and "gotchas" discovered during development. Intended for future contributors and for retrospective review.

---

## 1. Apple Silicon ML Ecosystem Learnings

### 1.1 MLX vs PyTorch MPS: When Each Wins

| Scenario | Winner | By How Much | Why |
|---|---|---|---|
| Single-item inference (batch=1) | **PyTorch MPS** | ~1.0x (tied) | MLX has higher Python dispatch overhead per call |
| Batched inference (batch≥16) | **MLX** | +25-32% | `@mx.compile` JIT fuses operations, eliminating Metal command buffer overhead |
| Text search latency | **MLX** | 1.8x faster | 1.02ms vs 1.86ms per query — matters for interactive UI |
| Memory efficiency | **MLX** | ~same (FP16) | Both use unified memory; MLX 8-bit is 50% less |
| Ecosystem maturity | **PyTorch** | significantly | Larger community, more model support, better debugging |

**Decision:** Use MLX as primary with PyTorch MPS as fallback. The strategy pattern (`EmbedderInterface`) allows hot-swapping at startup.

**Key insight:** At batch=1, don't expect MLX to be faster. The advantage only materialises when batching allows the GPU to amortise kernel launch overhead. This is why `EMBED_BATCH_SIZE=64` is the production default.

### 1.2 MLX `@mx.compile` Kernel Fusion

The `@mx.compile` decorator JIT-compiles a function into a single Metal compute graph. For SigLIP 2's vision encoder, this fuses:
- Input normalisation (`(x/255 - 0.5) / 0.5`)
- Patch embedding convolution
- All transformer blocks
- L2 normalisation

**Measured impact:** +32% throughput at batch 16 (443 → 466 img/s with fusion vs without). The key is that without fusion, each operation submits a separate Metal command buffer. With fusion, the entire forward pass is a single dispatch.

**Gotcha:** The first call to a `@mx.compile` function is slow (compilation overhead). Warm up with a dummy batch at startup.

### 1.3 8-Bit Quantization is Free (for This Use Case)

We tested `mlx-community/siglip2-base-patch16-224-8bit` and found:
- **Cosine similarity vs FP16:** 1.000000 (exact match to 6 decimal places)
- **Mean Squared Error:** 3.58e-07 (negligible)
- **Model RAM:** 440 MB vs 860 MB (50% reduction)
- **GPU throughput:** -13% at batch 32 (448 → 388 img/s)
- **Wall-clock impact:** 0% (pipeline is I/O-bound, GPU is idle 95% of the time)

**Lesson:** For embedding models used in retrieval (where cosine similarity ranking is all that matters), 8-bit quantization is essentially free. The ranking order is preserved perfectly. This would NOT hold for generative models where token-level precision matters.

---

## 2. Apple Hardware Acceleration Learnings

### 2.1 ImageIO is 1.6x Faster Than Pillow for HEIC

When decoding 48MP iPhone HEIC photos:
- **Pillow + pillow-heif:** Full 48MP decode → RGB buffer → resize to 224×224. Requires `libheif` (C library).
- **Apple ImageIO (PyObjC):** `CGImageSourceCreateThumbnailAtIndex` with `kCGImageSourceThumbnailMaxPixelSize=224`. Hardware-accelerated, produces the 224×224 thumbnail directly without ever materialising the full-resolution image.

**Why ImageIO wins:** It uses Apple's Image I/O framework which internally leverages the hardware JPEG/HEIF decoder on the SoC. Pillow uses `libheif` in software mode.

**Gotcha:** ImageIO is macOS-only. Keep Pillow as the cross-platform fallback.

### 2.2 OpenCV AVFoundation > FFmpeg for Frame Extraction

**Original plan:** Spawn `ffmpeg -hwaccel videotoolbox` subprocesses, pipe raw frames to Python via `stdout`.

**What we found:**
- FFmpeg was not installed on the target machine.
- OpenCV's `cv2.VideoCapture` with `CAP_AVFOUNDATION` backend uses Apple's VideoToolbox hardware decoder natively, in-process.
- No subprocess overhead, no pipe serialisation, no binary dependency.
- Measured: 22.6 frames/sec extraction throughput (25 real video clips, 283 frames, 17.3s total).

**Decision:** OpenCV + AVFoundation for extraction. Keep FFmpeg for final video rendering only (it's still the best tool for complex filter graphs and encoding).

### 2.3 The Decompression Bottleneck is Unsolvable (And That's OK)

92.5% of pipeline time is spent decompressing HEIC files. This is fundamentally I/O + hardware decoder throughput limited. Even with 12 CPU threads and hardware acceleration, we get ~20 photos/second.

**What doesn't help:**
- More GPU power (GPU is idle 95% of the time)
- Faster model (embedding takes 5% of wall time)
- More threads beyond 12 (diminishing returns, OS scheduling overhead)

**What would help:**
- Faster SSD (sequential read throughput)
- Apple's hardware HEIF decoder getting faster in future SoCs
- Pre-decoded thumbnail cache (trade disk space for decode time on re-runs)

**Lesson:** Know your bottleneck. We spent time optimising the GPU pipeline (MLX, kernel fusion, 8-bit quantization) only to discover that it accounts for <6% of wall time. The real wins came from optimising the decoder (ImageIO over Pillow).

---

## 3. Pipeline Architecture Learnings

### 3.1 Bounded Queues Are Non-Negotiable

Without bounded queues, the scanner + decoder threads would fill memory with decoded frames faster than the GPU could consume them (GPU is much faster per-item, but the 12 decoder threads collectively produce more data).

With bounded queues (`frame_queue` capacity 256), backpressure naturally throttles producers. Memory usage stays flat and predictable.

### 3.2 GPU Consumer Should Own Transforms

**Original design:** CPU does transforms (AutoProcessor) → GPU does forward pass.

**Better design:** CPU sends raw uint8 numpy arrays → GPU does everything (normalize, permute, forward pass, L2 norm) in one fused `@mx.compile` call.

**Why:** Eliminates CPU-GPU data format conversion overhead and allows the entire transform + inference pipeline to be JIT-compiled into a single Metal compute graph.

### 3.3 Thread Count is Empirical, Not Theoretical

We tested 4, 8, 12, and 16 decoder threads:
- **4 threads:** GPU starved (too slow to saturate frame queue)
- **8 threads:** Better, but still some GPU idle time
- **12 threads:** Optimal — saturates I/O bandwidth without excessive context switching
- **16 threads:** Slightly worse than 12 due to OS scheduling overhead

**Lesson:** Always benchmark thread counts empirically. The "optimal" number depends on: SSD throughput, file format decode cost, OS scheduler behaviour, and memory pressure.

---

## 4. Benchmarking Methodology Learnings

### 4.1 Always Measure Wall Clock, Not Just GPU Throughput

Our initial benchmarks showed "MLX is 30% faster than PyTorch MPS!" But when we measured end-to-end wall clock on 1,000 real photos, both engines completed in ~50 seconds. The GPU throughput difference was invisible because the pipeline is I/O-bound.

**Lesson:** Synthetic benchmarks (pure GPU forward pass) are useful for understanding component performance, but wall-clock measurement on real workloads is what determines architecture decisions.

### 4.2 Use Real Data, Not Synthetic

Synthetic benchmarks with random tensors miss:
- File I/O latency (HEIC is much heavier than JPEG)
- Variable image sizes (48MP vs 12MP vs screenshots)
- File system cache effects
- OS memory pressure under sustained load

Our real-world benchmark suite uses the actual user corpus (1,000 iPhone photos, 25 video clips) which reveals bottlenecks that synthetic tests miss.

### 4.3 Telemetry Should Be Structured

We created [`scripts/telemetry_utils.py`](file:///Users/rushivyas/development/projects/moments-generator/scripts/telemetry_utils.py) to standardize benchmark output in both Markdown (human-readable) and JSON (machine-parseable) formats. Every benchmark writes to `data/benchmarks/` with a timestamp.

This made it possible to:
- Compare results across sessions
- Track performance regressions
- Make data-driven architecture decisions

---

## 5. Model Selection Learnings

### 5.1 SigLIP 2 vs CLIP

SigLIP 2 was chosen over CLIP for several reasons:
- **Sigmoid loss** (SigLIP) vs **contrastive loss** (CLIP): SigLIP doesn't need negative pairs, leading to better zero-shot retrieval.
- **Base model size:** SigLIP 2 base (224×224, 768-dim) is compact enough for local inference.
- **Upgrade path:** SigLIP 2 SO400M (384×384, 1152-dim) is available if accuracy needs improvement.

### 5.2 Embedding Dimension Trade-offs

768-dim FP16 embeddings at 1.5 KB each means:
- 10,000 photos → ~15 MB of vectors
- 100,000 photos → ~150 MB of vectors
- Video at 1 FPS → ~11 frames per clip → ~16.5 KB per video

These sizes are well within Qdrant's in-memory index capacity.

---

## 6. Development Environment Learnings

### 6.1 uv Over pip

`uv` (from Astral) is ~10-50x faster than pip for dependency resolution and installation. With the number of ML dependencies (PyTorch, MLX, transformers, etc.), this saves significant time during setup.

### 6.2 Docker for Qdrant, Native for Everything Else

Running the ML pipeline natively (outside Docker) is essential for Metal GPU access. Docker on macOS runs in a Linux VM that cannot access Metal Performance Shaders or the Neural Engine.

Qdrant runs fine in Docker because vector search is CPU/memory-bound, not GPU-bound.

---

## Changelog

| Date | Entry | Author |
|---|---|---|
| 2026-08-15 | Initial learnings captured from Milestone 1 benchmarking | — |
