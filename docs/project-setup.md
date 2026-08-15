# Project Structure & Setup Guide

**Version:** 1.0  
**Last Updated:** 2026-08-15  
**Reference:** [Technical Architecture](file:///Users/rushivyas/development/projects/moments-generator/docs/technical-architecture.md)

---

## 1. Directory Layout

```
moments-generator/
│
├── .gitignore                       # Git exclusions (venv, data, models, exports)
├── .python-version                  # Python version pin (3.12)
├── .env.example                     # Example environment config
├── pyproject.toml                   # Project metadata + all dependencies
├── docker-compose.yml               # Qdrant container definition
├── start.sh                         # One-command bootstrap script
├── README.md                        # User-facing project overview
│
├── docs/                            # 📚 All project documentation
│   ├── product-requirements.md      #   Finalised PRD (Phase 0, 1, 2)
│   ├── technical-architecture.md    #   Finalised tech architecture
│   ├── project-setup.md             #   This file — structure & setup guide
│   └── drafts/                      #   Original brainstorming notes (archived)
│       ├── Basic-app-requirements.md
│       ├── basic-tech-arch.md
│       └── structured-app-requirements.md
│
├── app/                             # 🐍 Application source code
│   ├── __init__.py
│   ├── main.py                      #   FastAPI entry point + lifespan hooks
│   ├── config.py                    #   Pydantic BaseSettings (all config)
│   │
│   ├── api/                         #   HTTP / API layer
│   │   ├── __init__.py
│   │   ├── routes.py                #     Endpoint definitions (POST/GET)
│   │   └── schemas.py               #     Request/response Pydantic models
│   │
│   ├── core/                        #   Core business logic
│   │   ├── __init__.py
│   │   ├── scanner.py               #     Directory walk + format detection
│   │   ├── metadata.py              #     EXIF / ffprobe / filesystem timestamps
│   │   ├── manifest.py              #     SQLite manifest manager (cache, state, checkpoint)
│   │   ├── pipeline.py              #     Parallel pipeline orchestrator
│   │   ├── extractor.py             #     FFmpeg frame extraction (video → frames)
│   │   ├── embedder.py              #     SigLIP 2 embedding (MLX / PyTorch strategy)
│   │   ├── curator.py               #     Time-bucketing, scoring, deduplication
│   │   └── renderer.py              #     FFmpeg stitching + export
│   │
│   ├── db/                          #   Persistence layer
│   │   ├── __init__.py
│   │   └── qdrant.py                #     Qdrant client wrapper
│   │
│   └── ui/                          #   Static frontend (served by FastAPI)
│       ├── index.html               #     Single-page app shell
│       ├── app.js                   #     SSE consumer, form handling
│       └── style.css                #     Styles
│
├── tests/                           # ✅ Test suite
│   ├── __init__.py
│   ├── conftest.py                  #   Shared fixtures (temp dirs, mock manifest)
│   ├── test_scanner.py
│   ├── test_metadata.py
│   ├── test_manifest.py
│   ├── test_pipeline.py
│   ├── test_curator.py
│   └── test_renderer.py
│
├── scripts/                         # 🔧 Developer utilities
│   └── download_model.py            #   One-time model weight download
│
├── data/                            # 💾 Runtime working data (GIT-IGNORED)
│   ├── manifests/                   #   SQLite manifest databases
│   └── qdrant_storage/              #   Qdrant Docker volume mount
│
├── models/                          # 🤖 Downloaded model weights (GIT-IGNORED)
│   └── .gitkeep
│
└── exports/                         # 📹 Rendered output videos (GIT-IGNORED)
    └── .gitkeep
```

### What Gets Committed vs. Ignored

| Committed to Git | Git-Ignored |
|---|---|
| `app/` — all source code | `.venv/` — virtual environment |
| `tests/` — all tests | `data/` — manifests, Qdrant storage |
| `docs/` — all documentation | `models/` — downloaded model weights |
| `scripts/` — dev utilities | `exports/` — rendered videos |
| `pyproject.toml` — dependency spec | `__pycache__/` — bytecode |
| `docker-compose.yml` — infra | `.env` — local environment overrides |
| `start.sh` — bootstrap | `.DS_Store` — macOS metadata |
| `.env.example` — config template | |

---

## 2. Module Responsibilities

### `app/` — Application Code

| Module | Responsibility | Key Classes/Functions |
|---|---|---|
| `main.py` | FastAPI app creation, lifespan hooks (model loading, Qdrant connection), static file mounting | `create_app()`, `lifespan()` |
| `config.py` | All application settings via Pydantic `BaseSettings`. Loaded from `.env` and environment variables. | `Settings` |
| `api/routes.py` | HTTP endpoint definitions | `generate_job()`, `job_events()`, `download_job()`, `health()` |
| `api/schemas.py` | Pydantic models for request validation and response serialisation | `GenerateRequest`, `JobResponse`, `ProgressEvent` |
| `core/scanner.py` | Recursive directory walk, file format filtering, file discovery | `scan_corpus()` |
| `core/metadata.py` | EXIF parsing, ffprobe execution, filesystem stat, timestamp resolution chain | `extract_metadata()` |
| `core/manifest.py` | SQLite manifest: create/open DB, file CRUD, status transitions, model version checks, skip/resume logic | `ManifestDB` |
| `core/pipeline.py` | Multi-threaded pipeline orchestrator: queue setup, worker management, backpressure, progress reporting | `IngestionPipeline` |
| `core/extractor.py` | FFmpeg frame extraction (video → pixel arrays), image decoding (Pillow/pillow-heif) | `extract_frames()`, `decode_image()` |
| `core/embedder.py` | SigLIP 2 model loading (MLX or PyTorch), batch embedding, L2 normalisation. Strategy pattern with fallback. | `Embedder`, `MLXBackend`, `PyTorchMPSBackend` |
| `core/curator.py` | Time-bucketing, Qdrant semantic search, threshold filtering, burst dedup, timeline assembly | `curate_timeline()` |
| `core/renderer.py` | FFmpeg complex filter graph construction, timeline-to-MP4 rendering, hardware codec selection | `render_video()` |
| `db/qdrant.py` | Qdrant client wrapper: collection management, upsert, filtered search, point deletion | `QdrantManager` |

### `tests/` — Test Suite

Each test module mirrors a `core/` module. Tests use:
- `pytest` as the test runner.
- `pytest-asyncio` for async endpoint tests.
- `httpx` for FastAPI test client.
- Temporary directories and mock manifests (no real media files in the repo).

### `scripts/` — Utilities

| Script | Purpose |
|---|---|
| `download_model.py` | Downloads SigLIP 2 weights from HuggingFace to `./models/`. Run once after initial setup. |

---

## 3. Setting Up the Development Environment

### Prerequisites

| Tool | Install Command | Version |
|---|---|---|
| Python 3.12+ | `brew install python@3.12` | 3.12.x |
| FFmpeg | `brew install ffmpeg` | 7.x |
| Docker Desktop | [docker.com/download](https://docker.com/products/docker-desktop/) | Latest |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Latest |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd moments-generator

# 2. Create the virtual environment
uv venv --python 3.12 .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install all dependencies (including dev tools)
uv pip install -e ".[dev]"

# 5. Create runtime directories
mkdir -p data/manifests data/qdrant_storage exports models

# 6. Copy environment config template
cp .env.example .env
# Edit .env to customise settings if needed

# 7. Start Qdrant
docker compose up -d qdrant

# 8. Download model weights (first time only)
python scripts/download_model.py

# 9. Verify HEIC support
python -c "import pillow_heif; pillow_heif.register_heif_opener(); print('✓ HEIC OK')"

# 10. Run the development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Or use the one-command bootstrap:

```bash
./start.sh
```

This script performs all of the above steps automatically.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_manifest.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing
```

### Linting & Formatting

```bash
# Check linting
ruff check app/ tests/

# Auto-fix linting issues
ruff check app/ tests/ --fix

# Format code
ruff format app/ tests/
```

---

## 4. Configuration Reference

All settings are managed via the `Settings` class in `app/config.py`. They can be overridden via:
1. A `.env` file in the project root (highest priority).
2. Environment variables prefixed with `MOMENTS_`.
3. Defaults in the `Settings` class (lowest priority).

### `.env.example`

```env
# ── Server ──
MOMENTS_HOST=127.0.0.1
MOMENTS_PORT=8000

# ── Model ──
MOMENTS_MODEL_NAME=google/siglip2-base-patch16-224
MOMENTS_MODEL_BACKEND=auto    # auto | mlx | pytorch_mps
MOMENTS_EMBED_BATCH_SIZE=32

# ── Pipeline ──
MOMENTS_EXTRACT_WORKERS=4
MOMENTS_INDEX_BATCH_SIZE=100
MOMENTS_FILE_QUEUE_SIZE=64
MOMENTS_FRAME_QUEUE_SIZE=256
MOMENTS_VECTOR_QUEUE_SIZE=512

# ── Curation ──
MOMENTS_MIN_SIMILARITY_THRESHOLD=0.22
MOMENTS_MAX_OUTPUT_DURATION=300
MOMENTS_DEFAULT_ASPECT_RATIO=1:1
MOMENTS_IMAGE_DISPLAY_DURATION=3.0
MOMENTS_VIDEO_SEGMENT_DURATION=3.0

# ── Qdrant ──
MOMENTS_QDRANT_HOST=localhost
MOMENTS_QDRANT_PORT=6333
MOMENTS_QDRANT_COLLECTION=media_embeddings

# ── Paths ──
MOMENTS_DATA_DIR=./data
MOMENTS_EXPORTS_DIR=./exports
MOMENTS_MODELS_DIR=./models

# ── Rendering ──
MOMENTS_VIDEO_CODEC=h264_videotoolbox
MOMENTS_VIDEO_BITRATE=6000k
MOMENTS_VIDEO_FPS=30
```

---

## 5. Git Workflow

### Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tested code. Always deployable. |
| `dev` | Integration branch for in-progress features. |
| `feature/<name>` | Individual feature branches (e.g., `feature/manifest`, `feature/parallel-pipeline`). |
| `fix/<name>` | Bug fix branches. |

### Commit Convention

Use conventional commits for clear history:

```
feat(manifest): add SQLite-based file tracking with state machine
fix(extractor): handle FFmpeg crash on corrupt MOV files
docs(arch): update pipeline diagram with bounded queues
test(curator): add time-bucketing edge case tests
```

---

## 6. `.gitignore`

```gitignore
# ── Python ──
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/

# ── Runtime data (large, machine-specific) ──
data/
models/
exports/

# ── Keep directory structure with .gitkeep ──
!models/.gitkeep
!exports/.gitkeep

# ── Environment ──
.env

# ── IDE ──
.idea/
.vscode/
*.swp
*.swo
*~

# ── macOS ──
.DS_Store
.AppleDouble
.LSOverride

# ── Testing ──
.coverage
htmlcov/
.pytest_cache/

# ── Ruff ──
.ruff_cache/
```
