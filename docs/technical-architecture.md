# Technical Architecture: Local AI Moments Generator

**Version:** 1.1  
**Status:** Finalised (updated with benchmark-proven architecture)  
**Last Updated:** 2026-08-15  
**Reference:** [Product Requirements Document](file:///Users/rushivyas/development/projects/moments-generator/docs/product-requirements.md)

---

## 1. System Overview & Deployment Topology

The application follows a **Decoupled Hybrid Architecture** designed to extract maximum performance from Apple Silicon (M5/M5 Pro unified memory and Neural Engine/GPU) while maintaining strict resource boundaries and data isolation.

```
+──────────────────────────────────────────────────────────────────────────────+
│  macOS Host (Apple Silicon — M5 / M5 Pro)                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Python Virtual Environment (.venv via uv)                            │  │
│  │                                                                        │  │
│  │  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │  │
│  │  │ FastAPI Server │  │ Embedding Engine │  │ Media Decode Engine    │  │  │
│  │  │  • REST API    │  │  • SigLIP 2      │  │  • ImageIO (photos)    │  │  │
│  │  │  • SSE Stream  │  │  • MLX / PyTorch │  │  • AVFoundation (vid)  │  │  │
│  │  │  • Static UI   │  │  • GPU batched   │  │  • FFmpeg (render)     │  │  │
│  │  └───────┬────────┘  └────────┬─────────┘  └──────────┬─────────────┘  │  │
│  │          │                    │                        │                │  │
│  │  ┌───────┴────────────────────┴────────────────────────┴─────────────┐  │  │
│  │  │                    Pipeline Orchestrator                          │  │  │
│  │  │  • Multi-threaded extraction    • Bounded queues                  │  │  │
│  │  │  • GPU-batched embedding        • Backpressure control           │  │  │
│  │  │  • Per-file error isolation     • Manifest checkpointing         │  │  │
│  │  └───────┬──────────────────────────────────────────────┬────────────┘  │  │
│  │          │                                              │              │  │
│  └──────────│──────────────────────────────────────────────│──────────────┘  │
│             │                                              │                 │
│             │  HTTP :6333                                  │                 │
│             ▼                                              ▼                 │
│  ┌─────────────────────┐                    ┌──────────────────────────┐     │
│  │ Docker Container    │                    │ SQLite Manifest DB       │     │
│  │  ┌───────────────┐  │                    │  • File tracking         │     │
│  │  │  Qdrant       │  │                    │  • Processing state      │     │
│  │  │  Vector DB    │  │                    │  • Model versioning      │     │
│  │  │  • mmap index │  │                    │  • Timeline persistence  │     │
│  │  │  • On-disk    │  │                    │  • Checkpoint / resume   │     │
│  │  └───────────────┘  │                    └──────────────────────────┘     │
│  └──────────┬──────────┘                                                     │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │ Host File System                                                        ││
│  │  • Raw Media Corpus (≤ 20 GB)                                           ││
│  │  • Qdrant Mapped Volume (./data/qdrant_storage/)                        ││
│  │  • Manifest DBs (./data/manifests/)                                     ││
│  │  • Rendered Exports (./exports/*.mp4)                                   ││
│  └──────────────────────────────────────────────────────────────────────────┘│
+──────────────────────────────────────────────────────────────────────────────+
```

### 1.1 Process Boundaries & Execution Contexts

| Context | Runs Where | Why |
|---|---|---|
| **Core Service** (FastAPI, Embedding Engine, FFmpeg, Pipeline) | macOS native, in `.venv` | Direct access to Metal GPU, Neural Engine, VideoToolbox. No hypervisor overhead. |
| **Qdrant Vector DB** | Docker container, `localhost:6333` | Isolation — a corrupt DB can't crash the core service. Easy to reset or upgrade. |
| **Manifest DB** | SQLite file on host filesystem | No external dependency. Atomic writes. Crash-safe checkpointing. |

### 1.2 Why This Split

- **MLX / PyTorch must run natively** to access Apple's Metal Performance Shaders and the Neural Engine. Docker on macOS runs in a Linux VM that cannot access these hardware accelerators.
- **FFmpeg must run natively** to use the `videotoolbox` hardware decoder/encoder.
- **Qdrant in Docker** is acceptable because vector search is CPU/memory-bound, not GPU-bound. The Docker overhead is negligible for this workload.

---

## 2. AI Inference & Hardware Acceleration

### 2.1 Model Selection

| Property | Value |
|---|---|
| **Model** | `google/siglip2-base-patch16-224` |
| **Upgrade path** | `google/siglip2-so400m-patch14-384` (if memory permits) |
| **Type** | Vision-Text encoder (shared embedding space) |
| **Model input resolution** | 224×224 (base) / 384×384 (so400m) |
| **Embedding dimension** | 768 (base) / 1152 (so400m) |
| **Precision** | Float16 (`fp16`) — default; 8-bit quantized available (see §2.6) |

> **⚠️ Critical: Two Different Resolutions in the System**
>
> | Stage | Resolution | Purpose |
> |---|---|---|
> | **Embedding** (Milestones 5-6) | 224×224 | Tiny thumbnail fed to SigLIP 2's vision encoder to produce a 768-dim semantic vector. Used ONLY for AI understanding ("what is in this image?"). |
> | **Rendering** (Milestone 10) | Original (e.g., 4000×3000) → scaled to output canvas (1080×1080) | The final video uses the **original high-resolution source files**. FFmpeg scales/pads them to the output canvas. No quality loss. |
>
> The 224×224 thumbnail **never appears in the final video**. It exists only transiently in memory during the embedding step. Original files are never modified.

### 2.2 Execution Framework: Strategy Pattern

The embedder uses a **strategy pattern** with automatic fallback:

```
┌──────────────────────────────┐
│       EmbedderInterface      │
│  embed_images(batch) → vecs  │
│  embed_text(prompt) → vec    │
└──────────┬───────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐  ┌────────────────┐
│ MLX      │  │ PyTorch + MPS  │
│ Backend  │  │ Backend        │
│ (primary)│  │ (fallback)     │
└──────────┘  └────────────────┘
```

**Startup logic:**
1. Attempt to load SigLIP 2 weights via MLX.
2. If MLX loading fails (incompatible model format, missing ops), fall back to `transformers` + `torch` with MPS device.
3. Log which backend is active. The manifest stores the backend used, so a backend change triggers re-embedding.

### 2.3 Text Encoding Pipeline

1. Normalise the user's text prompt (lowercase, strip excess whitespace).
2. Tokenise via the SigLIP 2 text tokenizer.
3. Encode through the text tower → 768-dim float16 vector.
4. L2-normalise the output vector.

### 2.4 Vision Encoding Pipeline

1. Preprocess frames/images: resize to 224×224, normalise pixel values per SigLIP 2 transforms.
2. Batch dynamically (batch size configurable, default 32).
3. Encode through the vision tower → 768-dim float16 vector per frame.
4. L2-normalise each output vector.

### 2.5 Model Initialisation

- The model is loaded **once** at FastAPI application startup via a singleton manager.
- Weights are held in unified memory for the lifetime of the process.
- **Measured memory** for `siglip2-base-patch16-224`:
  - FP16: **~860 MB** (MPS allocated: 771 MB + Python overhead).
  - 8-bit Quantized (MLX): **~440 MB** — 50% reduction with zero accuracy loss.

### 2.6 8-Bit Quantization Strategy (MLX Only)

The MLX backend supports loading 8-bit quantized weights from `mlx-community/siglip2-base-patch16-224-8bit`.

| Metric | FP16 | 8-Bit Quantized | Impact |
|---|---|---|---|
| Model RAM | ~860 MB | ~440 MB | **50% reduction** |
| Cosine similarity vs FP16 reference | 1.000000 | 1.000000 | **Zero loss** |
| Mean Squared Error vs FP16 | 0.0 | 3.58e-07 | Negligible |
| Batch 32 throughput | 448 img/s | 388 img/s | -13% (still I/O-bound) |
| 1,000 photo ingestion wall time | 50.52s | 50.02s | Identical (I/O-bound) |

**When to use 8-bit:** When processing very large corpora (50K+ files) where model RAM competes with OS file cache. Since the pipeline is I/O-bound, the slight GPU throughput reduction has zero impact on wall-clock time.

**Startup logic:** Controlled by `MODEL_PRECISION` config (`"fp16"` or `"8bit"`). When `"8bit"` is selected and the MLX backend is active, weights are loaded from the quantized checkpoint. Falls back to FP16 if quantized weights are unavailable.

---

## 3. Manifest System

The manifest is the backbone of caching, checkpointing, and incremental processing.

### 3.1 Storage

- **Format:** SQLite database.
- **Location:** `<project_root>/data/manifests/<corpus_hash>.db` — one DB per unique corpus path.
- `<corpus_hash>` = `xxhash64` of the canonical absolute corpus path.

### 3.2 Schema

```sql
-- Global metadata for this manifest
CREATE TABLE manifest_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Keys stored:
--   corpus_path       — absolute path to the corpus root
--   model_name        — e.g., "google/siglip2-base-patch16-224"
--   model_version     — e.g., model checkpoint hash or library version
--   model_backend     — "mlx" or "pytorch_mps"
--   embedding_dim     — e.g., "768"
--   created_at        — ISO 8601 timestamp
--   last_run_at       — ISO 8601 timestamp

-- One row per media file discovered in the corpus
CREATE TABLE files (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path           TEXT    UNIQUE NOT NULL,    -- Absolute path
    relative_path       TEXT    NOT NULL,            -- Relative to corpus root
    file_hash           TEXT    NOT NULL,            -- xxhash64(size + mtime) — fast change detection
    content_hash        TEXT,                        -- xxhash64(full file content) — content dedup
    file_size           INTEGER NOT NULL,            -- Bytes
    file_type           TEXT    NOT NULL,            -- 'image' | 'video'
    mime_type           TEXT,

    -- Temporal metadata
    creation_timestamp  REAL,                        -- Unix timestamp, best-effort
    timestamp_source    TEXT,                        -- 'exif' | 'container' | 'filesystem'
    duration_seconds    REAL,                        -- Video duration; NULL for images

    -- Processing state
    status              TEXT    NOT NULL DEFAULT 'PENDING',
    error_message       TEXT,

    -- Extraction data
    frame_count         INTEGER,                     -- Frames extracted (videos only)

    -- Embedding data
    model_name          TEXT,                        -- Model that produced the embeddings
    model_version       TEXT,
    qdrant_point_ids    TEXT,                        -- JSON array of UUID strings
    embedded_at         REAL,                        -- Unix timestamp

    -- Housekeeping
    scanned_at          REAL,
    updated_at          REAL
);

CREATE INDEX idx_files_status       ON files(status);
CREATE INDEX idx_files_hash         ON files(file_hash);
CREATE INDEX idx_files_content_hash ON files(content_hash);
CREATE INDEX idx_files_type         ON files(file_type);
CREATE INDEX idx_files_ts           ON files(creation_timestamp);

-- Persisted curated timeline for crash-safe rendering
CREATE TABLE timeline_segments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT    NOT NULL,               -- UUID of the generation job
    position         INTEGER NOT NULL,               -- Playback order (0-indexed)
    file_id          INTEGER NOT NULL REFERENCES files(id),
    segment_type     TEXT    NOT NULL,               -- 'image' | 'video_clip'
    start_offset     REAL,                           -- Seconds into the source video
    duration         REAL    NOT NULL,               -- Seconds for this segment
    similarity_score REAL,                           -- Cosine similarity to prompt
    time_bucket      INTEGER,                        -- Which bucket this came from
    UNIQUE(job_id, position)
);
```

### 3.3 File State Machine

```
                          ┌─────────┐
                     ┌───▶│  ERROR  │
                     │    └─────────┘
                     │ (any stage can fail)
                     │
┌─────────┐   ┌─────┴───┐   ┌───────────┐   ┌──────────┐   ┌─────────┐
│ PENDING │──▶│ SCANNED │──▶│ EXTRACTED │──▶│ EMBEDDED │──▶│ INDEXED │
└─────────┘   └─────────┘   └───────────┘   └──────────┘   └─────────┘
```

| State | Meaning | Checkpoint guarantee |
|---|---|---|
| `PENDING` | File discovered in scan, no work done | — |
| `SCANNED` | Metadata extracted (EXIF, timestamps, duration) | Metadata is persisted |
| `EXTRACTED` | Frames extracted to memory (videos) or image decoded | Ready for embedding |
| `EMBEDDED` | Embedding vectors computed | Vectors ready for Qdrant |
| `INDEXED` | Vectors upserted into Qdrant; point IDs stored | Fully processed |
| `ERROR` | Failed at any stage; `error_message` records why | Previous stages preserved |

### 3.4 Skip & Resume Logic

On each pipeline run:

```
For each file in corpus:
  existing = manifest.lookup(file.path)

  IF existing AND existing.hash == current_hash:
      IF existing.status == "INDEXED":
          → SKIP (fully processed, nothing to do)
      ELSE:
          → RESUME from the next stage after existing.status
  ELIF existing AND existing.hash != current_hash:
      → File was modified. Delete old Qdrant points. Reset to PENDING.
  ELSE:
      → New file. Insert as PENDING.

For files in manifest NOT found in corpus:
  → File was deleted. Remove Qdrant points. Delete manifest row.
```

### 3.5 Model Version Change Detection

```
On startup:
  stored = manifest.get_meta("model_name") + manifest.get_meta("model_version")
  current = config.MODEL_NAME + config.MODEL_VERSION

  IF stored != current:
      → All files reset to SCANNED (keep metadata, re-embed everything)
      → All Qdrant points deleted (collection recreated)
      → Update manifest_meta with new model info
```

This ensures metadata extraction (which is expensive for videos) is never wasted, but embeddings are always consistent with the active model.

---

## 4. Parallel Ingestion Pipeline

### 4.1 Architecture

```
                                                             raw_frame_queue                                              vector_queue
 file_queue        +-----------------------------+   (224x224 uint8 numpy)    +--------------------------------+    (768-dim fp16)     +-----------+
+---------+  ====> | Media Decoder               | ========================> | GPU Consumer                   | ===================> | Indexer   |
| Scanner |        | 12 threads (ThreadPool)      |                           | 1 thread, MLX JIT-compiled     |                      | 1 thread  |
| 1 thread|        | ImageIO (photos)             |                           | Transform -> Embed -> L2 Norm  |                      | batch     |
|         |        | AVFoundation/OpenCV (videos) |                           | @mx.compile kernel fusion      |                      | upsert    |
+---------+        +-----------------------------+                           +--------------------------------+                      +-----------+
     |                        |                                                        |                                                  |
     +------------------------+--------------------------------------------------------+--------------------------------------------------+
                                                              |
                                                    Manifest DB (SQLite)
                                               (status updated after each stage)
```

### 4.2 Queue Design

All inter-stage communication uses **bounded queues** to enforce backpressure. If the downstream consumer is slower than the upstream producer, the producer blocks until space is available — preventing unbounded memory growth.

| Queue | Capacity | Item Type | Rationale |
|---|---|---|---|
| `file_queue` | 64 | `MediaFile` objects | Scanner is very fast; small buffer is enough |
| `raw_frame_queue` | 256 | `(file_id, frame_index, uint8_numpy_224x224)` | At 224×224×3 uint8 ≈ 150 KB/frame → ~38 MB max |
| `vector_queue` | 512 | `(file_id, frame_index, embedding_vector)` | At 768 × fp16 ≈ 1.5 KB/vector → <1 MB max |

### 4.3 Stage Details

#### Stage 1: Scanner (1 thread)
- Walks the corpus directory recursively.
- For each supported file: compute `xxhash64(path + size + mtime)`.
- Check manifest for skip/resume decision.
- Extract metadata (EXIF, exifread, fs stat).
- Update manifest to `SCANNED`.
- Push to `file_queue`.

#### Stage 2: Media Decoder (12 threads, ThreadPool)
- `ThreadPoolExecutor` with 12 workers pulls from `file_queue`.
- **For images (primary — Apple ImageIO via PyObjC):**
  - Uses `CGImageSourceCreateThumbnailAtIndex` with `kCGImageSourceThumbnailMaxPixelSize=224`.
  - Hardware-accelerated HEIC/JPEG/PNG decompression directly to a 224×224 thumbnail in a single native call.
  - Avoids full 48MP decompression → resize overhead that Pillow/pillow-heif would incur.
  - Push uint8 numpy array (224×224×3) to `raw_frame_queue`.
- **For images (fallback — Pillow):**
  - If ImageIO unavailable (non-macOS), fall back to `Pillow` + `pillow-heif` for HEIC.
- **For videos (primary — OpenCV + AVFoundation):**
  - Uses `cv2.VideoCapture` with `CAP_AVFOUNDATION` backend for native Apple VideoToolbox hardware decoding.
  - Extracts frames at 1 FPS, resizes to 224×224 in-process.
  - No subprocess, no pipe buffering, no FFmpeg binary dependency for extraction.
  - Measured: 22.6 frames/second extraction throughput.
- **For videos (fallback — FFmpeg subprocess):**
  - If OpenCV/AVFoundation unavailable, fall back to FFmpeg subprocess with `videotoolbox` hwaccel.
- Update manifest to `EXTRACTED` after all frames for a file are pushed.
- **Error handling:** If decoding fails on a file, mark as `ERROR` in manifest, log warning, continue.

#### Stage 3: GPU Consumer (1 thread, MLX JIT-compiled)
- Pulls raw frames from `raw_frame_queue`, accumulates a batch of `EMBED_BATCH_SIZE` (default 64).
- **Fused GPU pipeline** (all steps run on Apple Silicon GPU in one `@mx.compile` pass):
  1. Convert uint8 numpy → MLX float16 tensor.
  2. Normalize: `(x / 255.0 - 0.5) / 0.5`.
  3. Forward pass through SigLIP 2 vision encoder → N × 768-dim vectors.
  4. L2-normalise each vector.
- Pushes `(file_id, frame_index, vector)` tuples to `vector_queue`.
- When all frames for a file have been embedded, update manifest to `EMBEDDED`.

#### Stage 4: Indexer (1 thread, batch upsert)
- Pulls vectors from `vector_queue`, accumulates a batch of `INDEX_BATCH_SIZE` (default 100).
- Upserts batch into Qdrant with payloads:
  ```json
  {
    "file_path": "/path/to/media",
    "file_type": "video",
    "frame_index": 42,
    "creation_timestamp": 1710000000.0,
    "duration_seconds": 120.5
  }
  ```
- After all vectors for a file are indexed, update manifest to `INDEXED` and store `qdrant_point_ids`.

### 4.4 Empirical Performance & Hardware Acceleration

Based on comprehensive benchmark telemetry across all three modalities — text, photos (1,000 real 48MP iPhone HEIC/JPEG), and videos (25 real clips, 283 frames) — the ingestion pipeline bottlenecks and optimal architecture are well-defined.

> All benchmark data was collected on Apple Silicon M5 Pro with PyTorch 2.13.0 and MLX. Full reports are in [`data/benchmarks/`](file:///Users/rushivyas/development/projects/moments-generator/data/benchmarks). Benchmark suite documentation is in [`docs/benchmarking.md`](file:///Users/rushivyas/development/projects/moments-generator/docs/benchmarking.md).

**Stage-by-Stage Telemetry Breakdown (1,000 Photos):**
| Pipeline Stage | Time Spent | % of Total Time | Notes |
| :--- | :--- | :--- | :--- |
| **1. File Decompression (I/O + Decode)** | ~47.0s (Wall) | **92.5%** | Across 12 CPU worker threads via Apple ImageIO. Decompressing HEIC is the heaviest step. |
| **2. GPU Transform + Embed + L2 Norm** | ~2.6s | **5.2%** | MLX JIT-compiled fused pass. SigLIP 2 at ~470 img/s (batch 32+). |
| **3. Qdrant Upsert** | ~0.4s | **<1%** | Batch upsert is negligible. |
| **4. Overhead (scheduling, queues)** | ~0.5s | **~1%** | Python threading overhead. |

**Multi-Modal Throughput Summary:**
| Modality | Engine | Throughput | Wall Time (sample) |
|---|---|---|---|
| **Text Search** | MLX FP16 (fused) | **968 queries/s** (1.02 ms/query) | — |
| **Text Search** | PyTorch MPS | 538 queries/s (1.86 ms/query) | — |
| **Photo Ingestion** | MLX FP16 (12 decode threads) | **19.7 photos/s** | 50.0s / 1,000 photos |
| **Photo Ingestion** | MLX 8-bit (12 decode threads) | **19.8 photos/s** | 50.0s / 1,000 photos |
| **Video Ingestion** | MLX FP16 + AVFoundation | **16.3 frames/s** | 17.3s / 283 frames (25 clips) |

**Pure GPU Forward Pass Throughput (no I/O — synthetic benchmark):**
| Batch Size | PyTorch MPS (FP16) | MLX FP16 (Fused) | MLX 8-Bit (Fused) |
|---|---|---|---|
| 1 | 282 img/s | 280 img/s | 107 img/s |
| 16 | 353 img/s | 466 img/s (+32%) | 405 img/s |
| 32 | 356 img/s | 470 img/s (+32%) | 420 img/s |
| 64 | 359 img/s | 466 img/s (+30%) | 421 img/s |

**Key Takeaways:**
1. The pipeline is **entirely I/O-bound** (92.5% decompression). GPU upgrade or model optimisation has near-zero impact on wall-clock time.
2. MLX `@mx.compile` kernel fusion provides **~30% GPU throughput gain** over PyTorch MPS at batch 16+.
3. At batch=1, PyTorch MPS matches MLX due to CPU dispatch overhead dominating. Batching is essential.
4. 8-bit quantization has **zero accuracy loss** (cosine similarity 1.000000) with 50% model RAM reduction.
5. The GPU at ~470 fps is starved by 12 CPU decode threads producing ~20 fps. GPU utilisation is <5%, keeping thermals cool.
6. Video extraction at 1 FPS produces ~11 frames per clip on average — useful for estimating Qdrant collection sizes.

### 4.5 Pipeline Lifecycle

```python
# Pseudocode for orchestrator
def run_pipeline(corpus_path, force_reindex):
    manifest = ManifestDB.open_or_create(corpus_path)
    
    if force_reindex:
        manifest.reset_all()
        qdrant.delete_collection()
    
    check_model_version_change(manifest)
    
    files = scanner.scan(corpus_path, manifest)  # Returns files needing work
    
    if not files:
        log.info("All files already indexed. Nothing to do.")
        return
    
    pipeline = IngestionPipeline(
        extract_workers=config.EXTRACT_WORKERS,   # Default: 12 (benchmark-proven optimal)
        embed_batch_size=config.EMBED_BATCH_SIZE,  # Default: 64 (GPU saturation sweet spot)
        index_batch_size=config.INDEX_BATCH_SIZE,  # Default: 100
    )
    
    pipeline.run(files, manifest, embedder, qdrant_client)
```

### 4.6 Memory Budget (Measured)

| Component | FP16 Mode | 8-Bit Mode | Notes |
|---|---|---|---|
| SigLIP 2 model weights | **~860 MB** | **~440 MB** | Loaded once at startup into unified memory |
| Raw frame queue (256 frames @ 150 KB) | ~38 MB | ~38 MB | Bounded uint8 numpy arrays |
| Embedding batch (64 frames on GPU) | ~50 MB | ~50 MB | Transient, freed after forward pass |
| Vector queue (512 vectors @ 1.5 KB) | <1 MB | <1 MB | Bounded |
| Qdrant client buffers | ~50 MB | ~50 MB | Batch upserts |
| Python runtime + FastAPI | ~200 MB | ~200 MB | Baseline |
| **Total peak (measured)** | **~1,200 MB** | **~780 MB** | Comfortable on 18+ GB unified memory |

> Note: Measured via `psutil.Process().memory_info().rss` during 1,000-photo ingestion. Qdrant runs separately in Docker with its own memory cap (recommend 2 GB `mem_limit`). No FFmpeg subprocesses are needed for extraction (OpenCV + AVFoundation is in-process).

---

## 5. Curation, Pacing & Culling Algorithms

### 5.1 Temporal Time-Bucketing

Prevents temporal clustering (e.g., 1,000 photos from a single day dominating a 5-year corpus).

**Step 1 — Calculate global time range:**

$$\Delta T_{corpus} = T_{max} - T_{min}$$

**Step 2 — Determine bucket count (K):**

Given target output duration $D_{out} \in [1, 300]$ seconds:

$$K = \min\left(10, \max\left(5, \left\lfloor \frac{D_{out}}{15} \right\rfloor\right)\right)$$

| Duration | K |
|---|---|
| 60 s | 5 |
| 120 s | 8 |
| 180 s | 10 |
| 300 s | 10 |

**Step 3 — Partition into intervals:**

$$B_i = \left[ T_{min} + i \cdot \frac{\Delta T_{corpus}}{K}, \; T_{min} + (i+1) \cdot \frac{\Delta T_{corpus}}{K} \right) \quad \text{for } i \in [0, K-1]$$

**Step 4 — Time quota per bucket:**

$$t_{quota} = \frac{D_{out}}{K}$$

**Step 5 — Filtered vector query per bucket:**

For each bucket $B_i$, execute a Qdrant filtered search:

```json
{
  "vector": "<prompt_embedding>",
  "filter": {
    "must": [
      {
        "key": "creation_timestamp",
        "range": { "gte": "<B_i.start>", "lt": "<B_i.end>" }
      }
    ]
  },
  "limit": 50,
  "with_payload": true
}
```

### 5.2 Semantic Thresholding & Fallback

- **Cosine similarity:** $S(p, v) = \frac{p \cdot v}{\|p\|_2 \|v\|_2}$
- **Relevance floor:** $\theta_{min} = 0.22$ (configurable via `config.MIN_SIMILARITY_THRESHOLD`).
- Any candidate scoring below $\theta_{min}$ is pruned.
- **Zero-match trigger:** If no candidates across all buckets meet the threshold, abort with `422 Unprocessable Entity` and reason `ZERO_SEMANTIC_MATCHES`.

### 5.3 Burst & Proximity Deduplication

1. **Visual similarity pruning:** If two selected items have cosine distance < 0.05 or perceptual hash Hamming distance < 5, retain only the higher-scoring item.
2. **Video temporal windowing:** If frames $f_a$ (offset $t_a$) and $f_b$ (offset $t_b$) from the same video file have $|t_a - t_b| < 4.0$ seconds, merge into a single segment: start = $\max(0, \min(t_a, t_b) - 0.5)$, duration = 3.0 seconds.
3. **Static image duration:** Each selected image is assigned a display duration of 3.0 seconds.

---

## 6. Rendering & Stitching Pipeline (FFmpeg)

### 6.1 Output Specifications

| Property | Value |
|---|---|
| Container | MP4 |
| Video codec | `h264_videotoolbox` (macOS hardware) or `libx264` (fallback) |
| Pixel format | `yuv420p` |
| Frame rate | 30 FPS |
| Bitrate | 6000 kbps |
| Audio | None (`-an`) for Phase 0 |
| Canvas resolution | Configurable (see §6.2) |

### 6.2 Canvas Resolutions

| Aspect ratio | Resolution | Use case |
|---|---|---|
| `1:1` (default) | 1080 × 1080 | Instagram, universal |
| `16:9` | 1920 × 1080 | YouTube, desktop |
| `9:16` | 1080 × 1920 | TikTok, Reels, Stories |

### 6.3 Aspect Ratio Normalisation Filter

Every source asset is scaled and padded to fit the canvas without stretching or cropping:

```
[in] scale=W:H:force_original_aspect_ratio=decrease,
     pad=W:H:(W-iw)/2:(H-ih)/2:color=black,
     setsar=1,
     fps=30 [out]
```

Where `W` and `H` are the canvas dimensions from §6.2.

### 6.4 Timeline Assembly (Complex Filter Graph)

For an assembled timeline with $N$ clips:

```bash
ffmpeg -y \
  -loop 1 -t 3.0 -i image_1.jpg \
  -ss 14.5 -t 3.5 -i video_1.mp4 \
  -loop 1 -t 3.0 -i image_2.heic \
  -filter_complex "
    [0:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v0];
    [1:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v1];
    [2:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v2];
    [v0][v1][v2]concat=n=3:v=1:a=0[v_out]
  " \
  -map "[v_out]" \
  -c:v h264_videotoolbox \
  -b:v 6000k \
  -pix_fmt yuv420p \
  ./exports/moments_output.mp4
```

### 6.5 Timeline Persistence

Before rendering begins, the curated timeline is written to the `timeline_segments` table in the manifest. This enables:
- **Crash recovery:** If FFmpeg crashes, re-run rendering from the persisted timeline without re-curating.
- **Debugging:** Inspect exactly which segments were selected and why (similarity scores, bucket assignments).

---

## 7. API Architecture & Event Streaming

### 7.1 REST Endpoints

#### `POST /api/v1/jobs/generate`

Initiates the full pipeline: scan → index → curate → render.

**Request body:**
```json
{
  "corpus_path": "/Users/developer/Pictures/Trek2025",
  "prompt": "Moments from the trek with epic mountain views",
  "target_duration_seconds": 120,
  "aspect_ratio": "1:1",
  "force_reindex": false
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "8f3b6c2a-9e12-4d5a-b9c1-ef4a37b12d54",
  "status": "QUEUED",
  "created_at": "2026-08-15T13:00:00Z"
}
```

#### `GET /api/v1/jobs/{job_id}/events`

Server-Sent Events (SSE) stream for real-time progress.

**Event stream:**
```
event: progress
data: {"stage": "SCANNING", "progress_pct": 100, "details": "Found 412 files (14.2 GB)"}

event: progress
data: {"stage": "INDEXING", "progress_pct": 45.2, "details": "Embedded 186/412 files", "skipped": 112}

event: progress
data: {"stage": "CURATING", "progress_pct": 100, "details": "Selected 34 segments across 8 time buckets"}

event: progress
data: {"stage": "RENDERING", "progress_pct": 72.0, "details": "FFmpeg rendering frame 1800/2500"}

event: complete
data: {"job_id": "8f3b6c2a-...", "output_path": "/exports/output.mp4", "duration": 119.5}
```

**Error event:**
```
event: error
data: {"job_id": "8f3b6c2a-...", "stage": "INDEXING", "error": "FFmpeg not found", "details": "..."}
```

#### `GET /api/v1/jobs/{job_id}/download`

Serves the rendered MP4 for browser playback / download.

#### `GET /api/v1/health`

Health check. Returns model status, Qdrant connectivity, FFmpeg availability.

```json
{
  "status": "healthy",
  "model": { "name": "siglip2-base-patch16-224", "backend": "mlx", "loaded": true },
  "qdrant": { "connected": true, "collections": 1 },
  "ffmpeg": { "available": true, "version": "7.1" }
}
```

---

## 8. Configuration

All configuration is managed via environment variables with sensible defaults, loaded by Pydantic `BaseSettings`.

```python
class Settings(BaseSettings):
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Model
    MODEL_NAME: str = "google/siglip2-base-patch16-224"
    MODEL_BACKEND: str = "auto"  # "auto" | "mlx" | "pytorch_mps"
    MODEL_PRECISION: str = "fp16"  # "fp16" | "8bit" (MLX only)
    EMBED_BATCH_SIZE: int = 64  # Benchmark-proven GPU saturation sweet spot

    # Pipeline
    EXTRACT_WORKERS: int = 12  # Benchmark-proven optimal for I/O-bound HEIC decode
    INDEX_BATCH_SIZE: int = 100
    FILE_QUEUE_SIZE: int = 64
    FRAME_QUEUE_SIZE: int = 256
    VECTOR_QUEUE_SIZE: int = 512

    # Curation
    MIN_SIMILARITY_THRESHOLD: float = 0.22
    MAX_OUTPUT_DURATION: int = 300
    DEFAULT_ASPECT_RATIO: str = "1:1"
    IMAGE_DISPLAY_DURATION: float = 3.0
    VIDEO_SEGMENT_DURATION: float = 3.0
    DEDUP_COSINE_THRESHOLD: float = 0.05
    DEDUP_PHASH_THRESHOLD: int = 5

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "media_embeddings"

    # Paths
    DATA_DIR: str = "./data"
    EXPORTS_DIR: str = "./exports"
    MODELS_DIR: str = "./models"

    # Rendering
    VIDEO_CODEC: str = "h264_videotoolbox"
    VIDEO_BITRATE: str = "6000k"
    VIDEO_FPS: int = 30

    class Config:
        env_file = ".env"
        env_prefix = "MOMENTS_"
```

---

## 9. Docker Compose (Qdrant)

```yaml
# docker-compose.yml
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:v1.12.1
    container_name: moments-qdrant
    ports:
      - "127.0.0.1:6333:6333"    # REST API
      - "127.0.0.1:6334:6334"    # gRPC
    volumes:
      - ./data/qdrant_storage:/qdrant/storage:z
    environment:
      QDRANT__SERVICE__GRPC_PORT: "6334"
    deploy:
      resources:
        limits:
          memory: 2G
    restart: unless-stopped
```

---

## 10. Bootstrap Script (`start.sh`)

```bash
#!/usr/bin/env bash
set -eo pipefail

echo "=== Initializing Local AI Moments Generator ==="

# ── 1. Dependency Validation ──
command -v python3 >/dev/null 2>&1 || { echo "Error: Python3 is required."; exit 1; }
command -v docker  >/dev/null 2>&1 || { echo "Error: Docker is required."; exit 1; }
# FFmpeg is optional for extraction (OpenCV + AVFoundation is used), but required for rendering
command -v ffmpeg  >/dev/null 2>&1 || echo "Warning: FFmpeg not found. Rendering will not work. Install with 'brew install ffmpeg'."

# ── 2. Install uv if needed ──
if ! command -v uv >/dev/null 2>&1; then
    echo "--- Installing uv package manager ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 3. Spin up Qdrant Vector DB ──
echo "--- Ensuring Qdrant is running ---"
docker compose up -d qdrant

# ── 4. Setup Virtual Environment ──
if [ ! -d ".venv" ]; then
    echo "--- Creating virtual environment ---"
    uv venv --python 3.12 .venv
fi

source .venv/bin/activate

# ── 5. Install / Sync Dependencies ──
echo "--- Syncing dependencies ---"
uv pip install -e ".[dev]"

# ── 6. Create data directories ──
mkdir -p data/manifests data/qdrant_storage exports models

# ── 7. Validate HEIC support ──
python3 -c "import pillow_heif; pillow_heif.register_heif_opener(); print('✓ HEIC support OK')" || {
    echo "Warning: HEIC support not available. HEIC files will be skipped."
}

# ── 8. Start Server ──
echo "--- Starting FastAPI Server (Apple Silicon Accelerated) ---"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 11. Dependency Summary

| Package | Purpose | Required By |
|---|---|---|
| `fastapi` | Web framework, REST API | API layer |
| `uvicorn[standard]` | ASGI server | Server |
| `sse-starlette` | Server-Sent Events | Progress streaming |
| `mlx` | Apple Silicon ML framework | Embedding (primary) |
| `mlx-lm` | MLX model loading + 8-bit quantization | Embedding (primary) |
| `transformers` | HuggingFace model loading | Embedding (fallback) |
| `torch` | PyTorch with MPS backend | Embedding (fallback) |
| `pyobjc-framework-Quartz` | Apple ImageIO bindings (PyObjC) | Photo decoding (primary) |
| `opencv-python` | Video frame extraction (AVFoundation) | Video decoding (primary) |
| `Pillow` | Image processing | Frame extraction (fallback) |
| `pillow-heif` | HEIC/HEIF decoding | iPhone photos (fallback) |
| `exifread` | EXIF metadata parsing | Timestamp extraction |
| `qdrant-client` | Vector database client | Vector search |
| `pydantic` | Data validation | Schemas, config |
| `pydantic-settings` | Environment config | Settings |
| `xxhash` | Fast file hashing | Manifest |
| `numpy` | Array operations | Embedding pipeline |
| `psutil` | Process memory monitoring | Telemetry, benchmarking |
| `ffmpeg` (system) | Media encoding/rendering | Rendering only (not extraction) |
| `docker` (system) | Container runtime | Qdrant |
