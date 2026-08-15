# Implementation Plan: Local AI Moments Generator — Phase 0 MVP

**Reference Docs:**
- [Product Requirements](file:///Users/rushivyas/development/projects/moments-generator/docs/product-requirements.md)
- [Technical Architecture](file:///Users/rushivyas/development/projects/moments-generator/docs/technical-architecture.md)
- [Project Setup](file:///Users/rushivyas/development/projects/moments-generator/docs/project-setup.md)

---

## Approach

Build the system **bottom-up**, one layer at a time. Each milestone produces something **runnable and testable** — we validate before moving on. No milestone depends on code that hasn't been validated in a prior milestone.

### Agent-Friendliness
Each milestone includes **explicit function signatures with type hints** so that any coding agent (including smaller models) can implement a module given just this plan + the referenced docs. If implementing a milestone, read the corresponding sections of [technical-architecture.md](file:///Users/rushivyas/development/projects/moments-generator/docs/technical-architecture.md) for full context.

### UI-First Development
The web UI and API skeleton are built in **Milestone 2** (not at the end). Each subsequent milestone adds features to the UI so you can always test incrementally from the browser — inspect manifest data, see scan results, watch pipeline progress, and play rendered output.

```
Milestone 1    Project Scaffolding + SigLIP 2 Spike
    ▼
Milestone 2    Config + API Shell + Debug UI
    ▼
Milestone 3    Manifest System (SQLite)
    ▼
Milestone 4    Scanner + Metadata Extraction      ← UI: show scan results
    ▼
Milestone 5    Frame Extractor (FFmpeg)
    ▼
Milestone 6    Embedding Engine (SigLIP 2)
    ▼
Milestone 7    Qdrant Integration
    ▼
Milestone 8    Parallel Ingestion Pipeline         ← UI: live pipeline progress
    ▼
Milestone 9    Curator (Bucketing + Scoring + Dedup)
    ▼
Milestone 10   Renderer + Full End-to-End          ← UI: video playback
```

---

## Milestone 1: Project Scaffolding + SigLIP 2 Spike

**Goal:** Repository structure, virtual environment, all dependencies installed and importable. Early validation that SigLIP 2 loads and produces embeddings on this machine.

### Proposed Changes

#### [NEW] `pyproject.toml`
- Project metadata, all dependencies (fastapi, mlx, torch, transformers, pillow, pillow-heif, qdrant-client, etc.), dev extras (pytest, ruff, httpx).

#### [NEW] `.python-version`
- Pin to `3.12`.

#### [NEW] `.gitignore`
- Exclude `.venv/`, `data/`, `models/`, `exports/`, `__pycache__/`, `.env`, `.DS_Store`, etc.

#### [NEW] `.env.example`
- Template for all `MOMENTS_*` environment variables with defaults.

#### [NEW] `docker-compose.yml`
- Qdrant service: image `qdrant/qdrant:v1.12.1`, ports `6333`/`6334`, volume mount `./data/qdrant_storage`, memory limit `2G`.

#### [NEW] `start.sh`
- Bootstrap script: validate deps, install `uv`, create venv, install packages, start Qdrant, run server.

#### [NEW] Directory stubs
- `app/__init__.py`, `app/api/__init__.py`, `app/core/__init__.py`, `app/db/__init__.py`
- `app/ui/` (empty for now)
- `tests/__init__.py`, `tests/conftest.py`
- `scripts/download_model.py` (placeholder)
- `models/.gitkeep`, `exports/.gitkeep`

#### [NEW] Working data directories
The `data/` directory is the **working/temp area** for all intermediate and runtime files. It is git-ignored and structured as:
```
data/
├── manifests/          # SQLite manifest DBs (one per corpus)
├── qdrant_storage/     # Qdrant Docker volume mount
└── cache/              # Future: intermediate files if needed
```
This is configurable via `MOMENTS_DATA_DIR` (defaults to `./data`). The user's input corpus and output exports are separate paths — `data/` is purely for internal working state.

#### [NEW] `scripts/spike_siglip2.py` — Early model validation
A standalone script to test SigLIP 2 loading and embedding **immediately** after deps are installed. This de-risks the most uncertain part of the stack before we write any framework code.

```python
"""Spike: Verify SigLIP 2 loads and produces embeddings on this machine.
Run this right after Milestone 1 to confirm model viability.
"""
import time
import numpy as np

print("=== SigLIP 2 Spike Test ===")

# 1. Try MLX
mlx_ok = False
try:
    import mlx.core as mx
    print(f"✓ MLX imported (version: {mx.__version__})")
    # MLX SigLIP 2 loading would go here once we confirm the path
    # For now, just verify MLX works
except ImportError:
    print("✗ MLX not available — will use PyTorch MPS")

# 2. Test PyTorch MPS
import torch
print(f"PyTorch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
assert torch.backends.mps.is_available(), "MPS not available — cannot run on this machine"

# 3. Load SigLIP 2 via transformers
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

model_name = "google/siglip2-base-patch16-224"
print(f"\nLoading {model_name}...")
t0 = time.time()

model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("mps")
model.eval()
print(f"✓ Model loaded in {time.time() - t0:.1f}s")

processor = AutoImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Test text embedding
text = "a beautiful mountain landscape"
inputs = tokenizer(text, return_tensors="pt", padding=True).to("mps")
with torch.no_grad():
    text_emb = model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
print(f"✓ Text embedding shape: {text_emb.shape}")

# 5. Test image embedding
from PIL import Image
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=dummy_img, return_tensors="pt").to("mps")
with torch.no_grad():
    img_emb = model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
print(f"✓ Image embedding shape: {img_emb.shape}")

# 6. Test cosine similarity
sim = (text_emb @ img_emb.T).item()
print(f"✓ Text-Image similarity: {sim:.4f}")

# 7. Memory usage
import os
rss_mb = os.popen(f'ps -o rss= -p {os.getpid()}').read().strip()
print(f"\nProcess RSS: {int(rss_mb) // 1024} MB")
print(f"Embedding dim: {text_emb.shape[-1]}")
print("\n=== Spike PASSED ✓ ===")
```

### ✅ Checkpoint 1
```bash
# Create venv and install deps
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify all critical imports
python -c "
import fastapi; print(f'FastAPI {fastapi.__version__}')
import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')
import mlx.core; print(f'MLX OK')
import PIL; print(f'Pillow {PIL.__version__}')
import pillow_heif; print('pillow-heif OK')
import qdrant_client; print(f'Qdrant client OK')
import xxhash; print('xxhash OK')
print('✓ All imports successful')
"

# Verify Qdrant starts
docker compose up -d qdrant
curl -s http://localhost:6333/healthz
docker compose down

# Run the SigLIP 2 spike — THIS IS CRITICAL
python scripts/spike_siglip2.py
```

**Pass criteria:**
1. All imports succeed.
2. `torch.backends.mps.is_available()` returns `True`.
3. Qdrant healthcheck returns 200.
4. **SigLIP 2 spike passes** — model loads, text and image embeddings are produced, shapes are correct. This confirms we can proceed with the architecture as designed.
5. If MLX import fails, we note it and proceed with PyTorch MPS — no blocker.

---

## Milestone 2: Config + API Shell + Debug UI

**Goal:** Centralised configuration, FastAPI app skeleton, and a **living debug UI** that grows with each milestone. From this point on, you can always `uvicorn app.main:app --reload` and test from the browser.

### Proposed Changes

#### [NEW] `app/config.py`

All settings via Pydantic `BaseSettings`, loadable from `.env`.

```python
# Key signatures:
class Settings(BaseSettings):
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Model
    MODEL_NAME: str = "google/siglip2-base-patch16-224"
    MODEL_BACKEND: str = "auto"  # "auto" | "mlx" | "pytorch_mps"
    EMBEDDING_RESOLUTION: int = 224  # Model input resolution (NOT output video resolution)
    EMBED_BATCH_SIZE: int = 32

    # Pipeline
    EXTRACT_WORKERS: int = 4
    INDEX_BATCH_SIZE: int = 100

    # Curation
    MIN_SIMILARITY_THRESHOLD: float = 0.22
    MAX_OUTPUT_DURATION: int = 300
    DEFAULT_ASPECT_RATIO: str = "1:1"  # Output video aspect ratio

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "media_embeddings"

    # Paths
    DATA_DIR: str = "./data"       # Working dir: manifests, caches
    EXPORTS_DIR: str = "./exports" # Rendered output videos
    MODELS_DIR: str = "./models"   # Downloaded model weights

    # Rendering (uses ORIGINAL resolution, not embedding resolution)
    VIDEO_CODEC: str = "h264_videotoolbox"
    VIDEO_BITRATE: str = "6000k"
    VIDEO_FPS: int = 30

    class Config:
        env_file = ".env"
        env_prefix = "MOMENTS_"

def get_settings() -> Settings:  # cached singleton
    ...
```

#### [NEW] `app/main.py` — FastAPI app skeleton

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

def create_app() -> FastAPI:
    app = FastAPI(title="Moments Generator", version="0.1.0")
    app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")
    app.include_router(api_router, prefix="/api/v1")
    return app
```

#### [NEW] `app/api/routes.py` — Initial endpoints

Starts with health check and debug endpoints. Job endpoints are stubs that return `501 Not Implemented` until their milestones are built.

```python
# Available from Milestone 2:
GET  /api/v1/health          → {status, ffmpeg, qdrant, model}
GET  /api/v1/debug/config    → current settings (non-sensitive)
GET  /api/v1/debug/data      → list files in data/ working directory

# Stubs (return 501 until implemented):
POST /api/v1/jobs/generate   → 501 (Milestone 8)
GET  /api/v1/jobs/{id}/events → 501 (Milestone 8)
GET  /api/v1/jobs/{id}/download → 501 (Milestone 10)
```

#### [NEW] `app/api/schemas.py` — Pydantic models

```python
class HealthResponse(BaseModel):
    status: str
    ffmpeg: dict     # {available: bool, version: str}
    qdrant: dict     # {connected: bool}
    model: dict      # {name: str, backend: str, loaded: bool}

class GenerateRequest(BaseModel):
    corpus_path: str
    prompt: str
    target_duration_seconds: int = Field(ge=1, le=300)
    aspect_ratio: str = "1:1"       # Output video AR, NOT embedding resolution
    force_reindex: bool = False

class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str

class ProgressEvent(BaseModel):
    stage: str       # SCANNING | INDEXING | CURATING | RENDERING
    progress_pct: float
    details: str
```

#### [NEW] `app/ui/index.html`, `app/ui/app.js`, `app/ui/style.css` — Debug Dashboard

A **living UI** that starts as a debug dashboard and evolves into the full app:

| Milestone | UI Features |
|---|---|
| **M2** (now) | Health status, config viewer, data directory browser |
| **M4** | + Scan results table (files found, types, sizes, timestamps) |
| **M8** | + Job submission form, live pipeline progress bar via SSE |
| **M9** | + Curated timeline preview (which segments were selected + scores) |
| **M10** | + Video player, download button |

The M2 UI includes:
- **Header:** App name, health status indicator (green/red dots for FFmpeg, Qdrant, Model).
- **Config panel:** Shows current settings from `/api/v1/debug/config`.
- **Data browser:** Lists contents of `data/` directory — manifests, Qdrant storage, caches. Click to inspect manifest DBs.
- **Job form:** Input fields (corpus path, prompt, duration, aspect ratio) — submit button is disabled with "Coming in Milestone 8" until pipeline is wired.

### ✅ Checkpoint 2
```bash
# Start the server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test config endpoint
curl http://localhost:8000/api/v1/debug/config

# Test data browser
curl http://localhost:8000/api/v1/debug/data

# Open the UI in browser
open http://localhost:8000/ui/

# Verify config override works
MOMENTS_EXTRACT_WORKERS=8 uvicorn app.main:app --port 8001 &
curl http://localhost:8001/api/v1/debug/config | jq .extract_workers  # Should be 8
kill %1
```

**Pass criteria:**
1. FastAPI starts, serves static UI at `/ui/`.
2. Health endpoint returns status with FFmpeg version and Qdrant connectivity.
3. Config endpoint returns all settings.
4. Data browser shows the `data/` directory structure.
5. UI renders in browser with health indicators, config panel, and data browser.
6. Job form is visible but disabled.
7. Environment variable overrides work.

### ✅ Checkpoint 2
```bash
python -c "
from app.config import get_settings
s = get_settings()
print(f'Model: {s.MODEL_NAME}')
print(f'Extract workers: {s.EXTRACT_WORKERS}')
print(f'Qdrant: {s.QDRANT_HOST}:{s.QDRANT_PORT}')
print(f'Exports dir: {s.EXPORTS_DIR}')
print('✓ Config loaded')
"
```

**Pass criteria:** All settings print with correct defaults. Overriding via `MOMENTS_EXTRACT_WORKERS=8 python -c "..."` works.

---

## Milestone 3: Manifest System

**Goal:** SQLite-based manifest with full CRUD, two-tier hashing for content dedup, state machine transitions, model version detection, and skip/resume logic.

### Proposed Changes

#### [NEW] `app/core/manifest.py`
- `ManifestDB` class:
  - `open_or_create(corpus_path)` — creates/opens the SQLite DB at `data/manifests/<hash>.db`.
  - `get_meta(key)` / `set_meta(key, value)` — manifest-level metadata.
  - `upsert_file(file_path, file_hash, content_hash, file_type, ...)` — insert or update a file record.
  - `lookup(file_path)` → `FileRecord | None`.
  - `lookup_by_content_hash(content_hash)` → `FileRecord | None` — finds an existing file with identical content (for dedup across renames/copies).
  - `update_status(file_id, new_status)` — state machine transition with timestamp.
  - `set_error(file_id, message)` — mark as ERROR.
  - `get_files_by_status(status)` → list of `FileRecord`.
  - `reset_embeddings()` — reset all INDEXED/EMBEDDED files to SCANNED (for model changes).
  - `remove_deleted_files(current_paths)` — prune rows for files no longer on disk.
  - `save_timeline(job_id, segments)` / `load_timeline(job_id)` — persist/restore curated timeline.
  - All writes use transactions for crash safety.
- `FileRecord` dataclass: mirrors the `files` table columns.

#### Two-Tier Hashing Strategy
The manifest stores **two hashes** per file:

| Hash | Algorithm | Input | Speed | Purpose |
|---|---|---|---|---|
| `file_hash` (fast) | `xxhash64` | `file_size + mtime` | Instant | Quick change detection — if this matches, nothing changed, skip immediately |
| `content_hash` (dedup) | `xxhash64` | Full file content | ~10 GB/s | Content dedup — if a file is renamed, copied, or moved but content is identical, reuse its embeddings |

**Skip/resume decision tree:**
```
For each file in corpus:
  fast_hash = xxhash64(size + mtime)
  existing = manifest.lookup(file.path)

  IF existing AND existing.fast_hash == fast_hash:
      → File unchanged. Skip if INDEXED, resume if partial.

  ELSE:
      content_hash = xxhash64(file content)     ← computed only when fast_hash misses
      donor = manifest.lookup_by_content_hash(content_hash)

      IF donor AND donor.status == "INDEXED":
          → Same content exists under different path.
          → Clone donor's qdrant_point_ids to this file.
          → Mark as INDEXED immediately. No re-embedding.

      ELSE:
          → Truly new or modified file. Process from PENDING.
```

The content hash is only computed when the fast hash doesn't match — so for a fully-cached re-run, **zero content hashes are computed** (instant skip via fast hash). Content hashing only kicks in for new or modified files, where the ~10 GB/s xxhash speed keeps it negligible.

#### [NEW] `tests/test_manifest.py`
- Test: create manifest, insert files, verify status transitions.
- Test: re-open existing manifest, verify data persisted.
- Test: model version change → all files reset to SCANNED.
- Test: fast hash change → triggers content hash check.
- Test: content hash dedup — file with same content at different path reuses embeddings.
- Test: skip logic — INDEXED files with matching fast hash are skipped instantly.
- Test: deleted file removal.
- Test: timeline save/load round-trip.
- Test: concurrent reads don't block (WAL mode).

### ✅ Checkpoint 3
```bash
pytest tests/test_manifest.py -v
```

**Pass criteria:** All manifest tests pass. State machine enforces valid transitions. Re-opening a manifest preserves all data.

---

## Milestone 4: Scanner + Metadata Extraction

**Goal:** Recursive directory walk, format filtering, EXIF/ffprobe/filesystem timestamp extraction, file hashing, manifest integration.

### Proposed Changes

#### [NEW] `app/core/scanner.py`
- `scan_corpus(corpus_path, manifest) → list[MediaFile]`:
  - Recursive walk with supported extension filtering (§3.2 of PRD).
  - Compute **fast hash** `xxhash64(size + mtime)` per file.
  - Check manifest: if fast hash matches + INDEXED → skip instantly.
  - If fast hash misses: compute **content hash** `xxhash64(file content)` → check for content dedup.
  - Detect new, modified, and deleted files.
  - Return list of files needing processing.
- `MediaFile` dataclass: path, relative_path, fast_hash, content_hash, size, file_type (image/video), mime_type.

#### [NEW] `app/core/metadata.py`
- `extract_metadata(file_path, file_type) → MetadataResult`:
  - For images: EXIF via `exifread` / Pillow — extract `DateTimeOriginal`.
  - For videos: `ffprobe -v quiet -print_format json -show_format` — extract `creation_time`, `duration`.
  - Fallback: `os.stat().st_birthtime` / `st_mtime`.
  - Return `MetadataResult(creation_timestamp, timestamp_source, duration_seconds)`.
- HEIC-specific handling via `pillow_heif.register_heif_opener()`.

#### [NEW] `tests/test_scanner.py`
- Test: scan a temp directory with mixed files (jpg, png, mp4, txt) — only supported formats returned.
- Test: sub-directory recursion.
- Test: skip logic — pre-indexed files excluded from result.
- Test: modified file detected (change mtime, re-scan).

#### [NEW] `tests/test_metadata.py`
- Test: extract timestamp from a JPEG with EXIF.
- Test: extract duration from an MP4 via ffprobe.
- Test: fallback to filesystem time when no EXIF/container metadata.
- Test: handle file with no metadata gracefully.

### ✅ Checkpoint 4
```bash
pytest tests/test_scanner.py tests/test_metadata.py -v

# Manual smoke test with real files
python -c "
from app.core.scanner import scan_corpus
from app.core.manifest import ManifestDB

manifest = ManifestDB.open_or_create('/path/to/small/test/folder')
files = scan_corpus('/path/to/small/test/folder', manifest)
for f in files[:5]:
    print(f'{f.file_type:6s} {f.relative_path}')
print(f'Total: {len(files)} files to process')
"
```

**Pass criteria:** Scanner finds all supported files, ignores unsupported ones, correctly identifies images vs videos. Metadata extraction returns timestamps for EXIF-tagged files and falls back gracefully. Re-running scan on same folder returns 0 files (all already in manifest).

---

## Milestone 5: Frame Extractor

**Goal:** FFmpeg-based frame extraction for videos and image decoding for photos. 

> [!IMPORTANT]
> **Resolution clarification:** The extractor produces **two outputs** for each media file:
> 1. **Embedding thumbnail** (224×224) — small, fast, used ONLY by SigLIP 2 to understand semantic content. Never appears in the final video.
> 2. **Original file path** — preserved in the manifest. The renderer (Milestone 10) uses the original high-resolution file for the final video output.
>
> The 224×224 downscale is purely for the AI model’s input. A 4000×3000 photo retains its full resolution in the final rendered video.

### Proposed Changes

#### [NEW] `app/core/extractor.py`

```python
@dataclass
class FrameData:
    """A single frame ready for embedding."""
    file_path: str          # Original high-res file (preserved for rendering)
    frame_index: int        # 0 for images, 0..N for video frames
    pixels: np.ndarray      # Shape: (224, 224, 3) uint8 — FOR EMBEDDING ONLY
    source_offset: float    # Seconds into the source video (0.0 for images)

def extract_video_frames(
    file_path: str,
    target_size: int = 224,   # Embedding input resolution (NOT output resolution)
) -> Generator[FrameData, None, None]:
    """Extract frames at 1 FPS, scaled to target_size for embedding.
    
    Uses FFmpeg with videotoolbox HW acceleration.
    Streams raw RGB from pipe:1 — no intermediate files on disk.
    The original video file is NOT modified or re-encoded.
    """
    ...

def decode_image(
    file_path: str,
    target_size: int = 224,   # Embedding input resolution (NOT output resolution)
) -> FrameData:
    """Decode an image and resize for embedding.
    
    Supports JPEG, PNG, HEIC (via pillow-heif), WEBP, TIFF.
    The original image file is NOT modified.
    """
    ...
```

#### [NEW] `tests/test_extractor.py`
- Test: extract frames from a short MP4 — verify frame count matches duration.
- Test: decode a JPEG — verify 224×224×3 output shape.
- Test: decode a HEIC — verify it works (or skips gracefully if no test HEIC available).
- Test: corrupt video file — verify graceful error, not crash.
- Test: verify `file_path` in FrameData points to original (not a temp file).

### ✅ Checkpoint 5
```bash
pytest tests/test_extractor.py -v

# Manual test with a real video
python -c "
from app.core.extractor import extract_video_frames
frames = list(extract_video_frames('/path/to/test/video.mp4'))
print(f'Extracted {len(frames)} frames')
print(f'Embedding shape: {frames[0].pixels.shape}')  # (224, 224, 3) — for AI only
print(f'Original file: {frames[0].file_path}')       # Full path to original video
"
```

**Pass criteria:** Video extraction produces correct frame count (~1 per second). Image decoding produces 224×224×3 arrays. HEIC files decode. Corrupt files don’t crash. Original file paths are preserved.

---

## Milestone 6: Embedding Engine

**Goal:** SigLIP 2 model loading with MLX/PyTorch strategy, batch vision embedding, text embedding, L2 normalisation.

> [!NOTE]
> The embedder receives 224×224 thumbnails from the extractor (Milestone 5) and produces 768-dim semantic vectors. The original high-res files are never touched by the embedder — they’re only used by the renderer (Milestone 10).

### Proposed Changes

#### [NEW] `app/core/embedder.py`

```python
from abc import ABC, abstractmethod

class EmbedderInterface(ABC):
    @abstractmethod
    def embed_images(self, batch: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of images.
        Args: batch — list of (224, 224, 3) uint8 arrays (embedding thumbnails).
        Returns: (N, 768) float32 array, L2-normalised.
        """
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text prompt.
        Returns: (768,) float32 array, L2-normalised.
        """
        ...

    @abstractmethod
    def model_info(self) -> dict:
        """Returns {name, version, backend, embedding_dim}."""
        ...

class PyTorchMPSBackend(EmbedderInterface):
    """SigLIP 2 via transformers + PyTorch MPS (Apple Silicon GPU)."""
    def __init__(self, model_name: str, models_dir: str): ...
    def embed_images(self, batch: list[np.ndarray]) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...
    def model_info(self) -> dict: ...

class MLXBackend(EmbedderInterface):
    """SigLIP 2 via Apple MLX framework (if available)."""
    def __init__(self, model_name: str, models_dir: str): ...
    # Same interface

def create_embedder(config: Settings) -> EmbedderInterface:
    """Factory: tries MLX first (if backend=auto), falls back to PyTorch MPS."""
    ...
```

#### [NEW] `scripts/download_model.py`
- Downloads `google/siglip2-base-patch16-224` weights + processor to `./models/` via HuggingFace `snapshot_download`.

#### [NEW] `tests/test_embedder.py`
- Test: load model (PyTorch MPS backend).
- Test: embed a single 224×224×3 image → verify output shape (768,) and L2-normalised.
- Test: embed a batch of 4 images → verify output shape (4, 768).
- Test: embed a text prompt → verify output shape (768,).
- Test: cosine similarity between "a mountain" text and a mountain image > similarity with "a car" text.

### ✅ Checkpoint 6
```bash
# Download model first
python scripts/download_model.py

pytest tests/test_embedder.py -v

# Manual test — semantic sanity check
python -c "
import numpy as np
from app.core.embedder import create_embedder
from app.config import get_settings

embedder = create_embedder(get_settings())
print(f'Backend: {embedder.model_info()[\"backend\"]}')

# Embed text
v_mountain = embedder.embed_text('a beautiful mountain landscape')
v_beach = embedder.embed_text('a sandy beach with ocean waves')
print(f'Text embedding shape: {v_mountain.shape}')

# Verify they're different
sim = np.dot(v_mountain, v_beach)
print(f'Mountain vs Beach similarity: {sim:.3f} (should be < 0.9)')
print('✓ Embedder working')
"
```

**Pass criteria:** Model loads on MPS (or MLX). Embeddings are 768-dim, L2-normalised. Semantically different prompts produce different vectors. Batch embedding works.

---

## Milestone 7: Qdrant Integration

**Goal:** Qdrant client wrapper — collection management, batch upsert with payloads, filtered semantic search, point deletion.

### Proposed Changes

#### [NEW] `app/db/qdrant.py`
- `QdrantManager`:
  - `__init__(config)` — connect to `localhost:6333`.
  - `ensure_collection(name, vector_size)` — create collection if it doesn't exist (cosine distance, on-disk payload index on `creation_timestamp`).
  - `upsert_batch(points: list[PointStruct])` — batch upsert with payloads (`file_path`, `file_type`, `frame_index`, `creation_timestamp`, `duration_seconds`).
  - `search(vector, filter_conditions, limit) → list[ScoredPoint]` — filtered nearest-neighbour search.
  - `delete_points(point_ids: list[str])` — remove points by ID.
  - `delete_collection(name)` — drop entire collection.
  - `health_check() → bool`.

#### [NEW] `tests/test_qdrant.py`
- Test: create collection, upsert 10 random vectors with payloads, search by vector, verify results.
- Test: filtered search by `creation_timestamp` range.
- Test: delete points, verify they're gone.
- Test: health check returns True when Qdrant is running.
- **Requires:** Qdrant running via `docker compose up -d qdrant`.

### ✅ Checkpoint 7
```bash
# Start Qdrant
docker compose up -d qdrant

pytest tests/test_qdrant.py -v

# Manual verification
python -c "
from app.db.qdrant import QdrantManager
from app.config import get_settings
import numpy as np

qm = QdrantManager(get_settings())
qm.ensure_collection('test_collection', vector_size=768)
print(f'Health: {qm.health_check()}')
print('✓ Qdrant integration working')
qm.delete_collection('test_collection')
"
```

**Pass criteria:** Collection creation, upsert, filtered search, and deletion all work. Payloads are stored and retrievable.

---

## Milestone 8: Parallel Ingestion Pipeline

**Goal:** Wire Scanner → Extractor → Embedder → Indexer with ThreadPoolExecutor, bounded queues, backpressure, manifest checkpointing, and per-file error isolation.

### Proposed Changes

#### [NEW] `app/core/pipeline.py`
- `IngestionPipeline`:
  - `__init__(config, manifest, embedder, qdrant_manager)`.
  - `run(files: list[MediaFile], progress_callback)`:
    - Sets up bounded queues (`file_queue`, `frame_queue`, `vector_queue`).
    - Spawns extraction ThreadPool (N workers).
    - Spawns embedder thread (consumes frame_queue, batches to GPU).
    - Spawns indexer thread (consumes vector_queue, batch upserts to Qdrant).
    - Updates manifest status after each stage per file.
    - Calls `progress_callback` with stage/percentage updates.
    - Handles SENTINEL values for graceful shutdown.
  - Per-file error isolation: catch exceptions per file, mark ERROR, continue.
  - Progress tracking: count files processed vs total.

#### [NEW] `tests/test_pipeline.py`
- Test: pipeline with mocked embedder and Qdrant — verify all files reach INDEXED status.
- Test: one file throws error during extraction — verify others still complete.
- Test: verify manifest statuses are updated correctly through the pipeline.
- Test: progress callback is called with increasing percentages.

### ✅ Checkpoint 8
```bash
pytest tests/test_pipeline.py -v

# End-to-end ingestion test with a small real corpus (~10-20 files)
python -c "
from app.core.pipeline import IngestionPipeline
from app.core.scanner import scan_corpus
from app.core.manifest import ManifestDB
from app.core.embedder import create_embedder
from app.db.qdrant import QdrantManager
from app.config import get_settings

config = get_settings()
manifest = ManifestDB.open_or_create('/path/to/small/test/corpus')
embedder = create_embedder(config)
qdrant = QdrantManager(config)
qdrant.ensure_collection(config.QDRANT_COLLECTION, embedder.model_info()['embedding_dim'])

files = scan_corpus('/path/to/small/test/corpus', manifest)
print(f'Files to process: {len(files)}')

def on_progress(stage, pct, detail):
    print(f'  [{stage}] {pct:.1f}% — {detail}')

pipeline = IngestionPipeline(config, manifest, embedder, qdrant)
pipeline.run(files, progress_callback=on_progress)

indexed = manifest.get_files_by_status('INDEXED')
errors = manifest.get_files_by_status('ERROR')
print(f'✓ Indexed: {len(indexed)}, Errors: {len(errors)}')

# Re-run — should skip everything
files2 = scan_corpus('/path/to/small/test/corpus', manifest)
print(f'Files on re-run: {len(files2)} (should be 0)')
"
```

**Pass criteria:** All files in a small test corpus reach INDEXED. Errors are isolated. Re-running produces 0 files to process. Progress callback fires with correct stages. This is the **big integration test** — scanner, extractor, embedder, Qdrant, and manifest all working together.

---

## Milestone 9: Curator

**Goal:** Time-bucketing, semantic search per bucket, threshold filtering, burst deduplication, timeline assembly, timeline persistence to manifest.

### Proposed Changes

#### [NEW] `app/core/curator.py`
- `curate_timeline(prompt, target_duration, aspect_ratio, embedder, qdrant_manager, manifest, config) → Timeline`:
  - Embed the text prompt.
  - Calculate time range from manifest (`min`/`max` `creation_timestamp`).
  - Compute bucket count K and partition into intervals.
  - For each bucket: filtered Qdrant search (by `creation_timestamp` range), score against prompt.
  - Apply relevance floor (`MIN_SIMILARITY_THRESHOLD`).
  - Zero-match detection → raise `ZeroSemanticMatchesError`.
  - Burst deduplication: cosine distance and perceptual hash.
  - Video segment merging: consecutive frames within 4s → single 3s clip.
  - Allocate time quota per bucket, select top segments.
  - Assemble chronologically ordered `Timeline` (list of `TimelineSegment`).
  - Persist timeline to manifest (`timeline_segments` table).
- `Timeline` dataclass: list of `TimelineSegment`, total_duration, job_id.
- `TimelineSegment` dataclass: file_path, segment_type, start_offset, duration, similarity_score, bucket_index.

#### [NEW] `tests/test_curator.py`
- Test: time-bucketing produces correct number of buckets for various durations.
- Test: segments are chronologically ordered.
- Test: burst dedup removes near-duplicates.
- Test: video segment merging combines close frames.
- Test: zero-match scenario raises error.
- Test: total timeline duration ≈ target duration (within tolerance).
- Test: timeline persists to and loads from manifest.

### ✅ Checkpoint 9
```bash
pytest tests/test_curator.py -v

# Manual test against the corpus indexed in Milestone 8
python -c "
from app.core.curator import curate_timeline
from app.core.embedder import create_embedder
from app.core.manifest import ManifestDB
from app.db.qdrant import QdrantManager
from app.config import get_settings

config = get_settings()
manifest = ManifestDB.open_or_create('/path/to/small/test/corpus')
embedder = create_embedder(config)
qdrant = QdrantManager(config)

timeline = curate_timeline(
    prompt='beautiful outdoor scenery',
    target_duration=60,
    aspect_ratio='1:1',
    embedder=embedder,
    qdrant_manager=qdrant,
    manifest=manifest,
    config=config,
)

print(f'Timeline: {len(timeline.segments)} segments, {timeline.total_duration:.1f}s')
for seg in timeline.segments:
    print(f'  [{seg.segment_type:5s}] {seg.duration:.1f}s  score={seg.similarity_score:.3f}  {seg.file_path}')
"
```

**Pass criteria:** Timeline is assembled with correct number of buckets, chronological ordering, no duplicate bursts, and total duration close to target. Different prompts produce different segment selections.

---

## Milestone 10: Renderer + Full End-to-End

**Goal:** FFmpeg rendering from curated timeline using **original high-resolution files** (not 224×224 thumbnails). Wire up the remaining API endpoints and complete the UI for full end-to-end flow.

> [!IMPORTANT]
> **Resolution flow recap:**
> - Milestones 5-6 produced 224×224 thumbnails → 768-dim embeddings → stored in Qdrant (for semantic search).
> - Milestone 10’s renderer reads the **original high-resolution files** from the paths stored in the manifest.
> - FFmpeg scales/pads originals to the output canvas (1080×1080, etc.) — full quality preserved.

### Proposed Changes

---

#### Renderer

#### [NEW] `app/core/renderer.py`

```python
def render_video(
    timeline: Timeline,
    aspect_ratio: str,          # "1:1" | "16:9" | "9:16"
    config: Settings,
    progress_callback: Callable[[float, str], None] | None = None,
) -> str:                       # Returns path to output MP4
    """
    Render the curated timeline to an MP4 video.
    
    Uses ORIGINAL high-resolution source files (not embedding thumbnails).
    Each source is scaled/padded to fit the output canvas without
    stretching or cropping.
    """
    ...

def get_canvas_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Returns (width, height) for the output canvas."""
    return {
        "1:1":  (1080, 1080),
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
    }[aspect_ratio]
```

#### [NEW] `tests/test_renderer.py`
- Test: render a timeline of 2 images → verify output MP4 exists, duration ≈ 6s.
- Test: render with a video clip + image → verify output is playable.
- Test: 1:1, 16:9, 9:16 aspect ratios all produce correctly sized output.
- Test: verify output resolution is 1080p (NOT 224×224).

---

#### API Layer (complete remaining endpoints)

#### [MODIFY] `app/api/routes.py`
- Enable the previously stubbed endpoints:
  - `POST /api/v1/jobs/generate` — validate input, create job, launch full pipeline (scan → index → curate → render) in background thread, return 202.
  - `GET /api/v1/jobs/{job_id}/events` — SSE stream of ProgressEvents.
  - `GET /api/v1/jobs/{job_id}/download` — serve rendered MP4 via `FileResponse`.

#### [MODIFY] `app/main.py`
- Complete lifespan hooks: load model, connect Qdrant, validate FFmpeg at startup.

---

#### UI (complete remaining features)

#### [MODIFY] `app/ui/index.html`, `app/ui/app.js`
- Enable the job submission form (was disabled since M2).
- Wire SSE progress: `EventSource("/api/v1/jobs/{id}/events")` → animated progress bar.
- On `complete` event: show video player with the rendered output + download button.
- Timeline preview panel: show which segments were selected, their scores, and thumbnails.

---

### ✅ Checkpoint 10 (Final MVP Validation)
```bash
# Run all unit tests
pytest tests/ -v

# Start the full stack
docker compose up -d qdrant
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000

# In browser: http://127.0.0.1:8000/ui/
# 1. Enter a corpus path (small test folder, ~10-20 photos/videos)
# 2. Enter a prompt: "beautiful outdoor moments"
# 3. Set duration: 30s
# 4. Click Generate
# 5. Watch progress bar advance through SCANNING → INDEXING → CURATING → RENDERING
# 6. Play the output video (should be 1080p, not 224px!)
# 7. Re-run with same corpus — verify indexing is skipped (cached)
# 8. Re-run with different prompt — verify different segments selected
# 9. Check data/ directory in debug UI — verify manifest, Qdrant storage
```

**Pass criteria:**
- Full end-to-end flow works from browser.
- Progress updates stream in real time.
- Output video is playable, **full output resolution** (not 224×224), correct aspect ratio, correct approximate duration.
- Re-running same corpus skips indexing (manifest cache works).
- Different prompts produce different outputs.
- Corrupt files in corpus are skipped without crashing.
- Debug UI shows manifest state, config, and working directory contents.

---

## Resolved Decisions

| Question | Decision |
|---|---|
| **File hashing** | Two-tier: fast hash (`xxhash64(size + mtime)`) for instant skip, content hash (`xxhash64(full file)`) for dedup across renames/copies. Content hash only computed when fast hash misses. |
| **Manifest location** | Project-side: `<project>/data/manifests/<corpus_hash>.db`. Keeps user corpus folders clean. Configurable via `MOMENTS_DATA_DIR`. |
| **Working directory** | `data/` directory inside the project root (git-ignored) holds all intermediate/runtime state: manifests, Qdrant storage, and future caches. |
| **MLX feasibility** | Tested early via `scripts/spike_siglip2.py` in Milestone 1. PyTorch MPS is the confirmed fallback. |

---

## Verification Plan

### Automated Tests (Every Milestone)
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Manual Verification (Milestones 8, 9, 10)
- Process a small real corpus (~10-20 files with mixed images/videos).
- Verify incremental indexing (re-run skips processed files).
- Verify different prompts → different outputs.
- Verify corrupt file resilience.

### Final Demo (Milestone 10)
- Process a larger corpus (~100+ files) end-to-end.
- Generate videos with 2-3 different prompts.
- Verify video quality, aspect ratios, temporal distribution.
