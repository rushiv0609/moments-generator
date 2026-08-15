#!/usr/bin/env bash
set -eo pipefail

echo "=== Initializing Local AI Moments Generator ==="

# ── 1. Dependency Validation ──
command -v python3 >/dev/null 2>&1 || { echo "Error: Python3 is required."; exit 1; }
command -v ffmpeg  >/dev/null 2>&1 || { echo "Error: FFmpeg is required. Install with 'brew install ffmpeg'."; exit 1; }
command -v docker  >/dev/null 2>&1 || { echo "Error: Docker is required."; exit 1; }

# ── 2. Install uv if needed ──
if ! command -v uv >/dev/null 2>&1; then
    echo "--- Installing uv package manager ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# ── 3. Spin up Qdrant Vector DB ──
echo "--- Ensuring Qdrant is running ---"
docker compose up -d qdrant

# ── 4. Setup Virtual Environment ──
if [ ! -d ".venv" ]; then
    echo "--- Creating virtual environment (Python 3.12) ---"
    uv venv --python 3.12 .venv
fi

source .venv/bin/activate

# ── 5. Install / Sync Dependencies ──
echo "--- Syncing dependencies ---"
uv pip install -e ".[dev]"

# ── 6. Create data directories ──
mkdir -p data/manifests data/qdrant_storage data/cache exports models

# ── 7. Validate HEIC support ──
python3 -c "import pillow_heif; pillow_heif.register_heif_opener(); print('✓ HEIC support OK')" || {
    echo "Warning: HEIC support not available. HEIC files will be skipped."
}

# ── 8. Start Server ──
echo "--- Starting FastAPI Server (Apple Silicon Accelerated) ---"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
