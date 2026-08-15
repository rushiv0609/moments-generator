"""
FastAPI Route Handlers for Local AI Moments Generator.
"""

import os
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from qdrant_client import QdrantClient

from app.config import get_settings
from app.api.schemas import (
    HealthResponse,
    ComponentStatus,
    ModelStatus,
    QdrantStatus,
    DataDirResponse,
    DataDirItem,
    GenerateRequest,
    JobResponse,
)

router = APIRouter()


def check_ffmpeg() -> ComponentStatus:
    """Check if ffmpeg is available on the system."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return ComponentStatus(
            available=False,
            details="ffmpeg binary not found in PATH. Install via 'brew install ffmpeg' for rendering support.",
        )
    try:
        out = subprocess.check_output([ffmpeg_path, "-version"], stderr=subprocess.STDOUT, text=True)
        version_line = out.splitlines()[0] if out else "Unknown"
        return ComponentStatus(
            available=True,
            version=version_line,
            details=f"Found at {ffmpeg_path}",
        )
    except Exception as e:
        return ComponentStatus(
            available=False,
            details=f"Error executing ffmpeg: {str(e)}",
        )


def check_qdrant() -> QdrantStatus:
    """Check connectivity to Qdrant vector database."""
    settings = get_settings()
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=2.0)
        collections = client.get_collections()
        count = len(collections.collections)
        return QdrantStatus(
            connected=True,
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collections=count,
            details="Connected successfully",
        )
    except Exception as e:
        return QdrantStatus(
            connected=False,
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collections=0,
            details=f"Cannot reach Qdrant server: {str(e)}",
        )


def check_model_backend() -> ModelStatus:
    """Check AI embedding model engine availability."""
    settings = get_settings()
    backend_info = settings.MODEL_BACKEND
    precision_info = settings.MODEL_PRECISION
    
    # Check if MLX is available
    mlx_available = False
    try:
        import mlx.core as mx
        mlx_available = True
    except ImportError:
        pass

    # Check if PyTorch MPS is available
    mps_available = False
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
    except ImportError:
        pass

    details = []
    if mlx_available:
        details.append("Apple MLX available (Primary)")
    if mps_available:
        details.append("PyTorch MPS available (Fallback)")

    return ModelStatus(
        name=settings.MODEL_NAME,
        backend=backend_info,
        precision=precision_info,
        loaded=False,  # Loaded on-demand or during worker startup
        details=", ".join(details) if details else "No hardware acceleration backends detected",
    )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    System health check. Inspects ML backend, Qdrant connectivity, and FFmpeg installation.
    """
    ffmpeg_status = check_ffmpeg()
    qdrant_status = check_qdrant()
    model_status = check_model_backend()

    # Memory telemetry
    mem_info = {}
    try:
        import psutil
        vmem = psutil.virtual_memory()
        mem_info = {
            "total_gb": round(vmem.total / (1024**3), 2),
            "available_gb": round(vmem.available / (1024**3), 2),
            "used_pct": vmem.percent,
        }
    except ImportError:
        mem_info = {"status": "psutil not available"}

    is_healthy = qdrant_status.connected
    overall_status = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        model=model_status,
        qdrant=qdrant_status,
        ffmpeg=ffmpeg_status,
        system_memory=mem_info,
    )


@router.get("/debug/config")
def get_debug_config() -> Dict[str, Any]:
    """
    Return active configuration settings.
    """
    settings = get_settings()
    return settings.model_dump()


@router.get("/debug/data", response_model=DataDirResponse)
def get_debug_data() -> DataDirResponse:
    """
    List contents and sizes of the working data directory (manifests, caches, storage).
    """
    settings = get_settings()
    data_path = Path(settings.DATA_DIR)
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    items = []
    for item in sorted(data_path.rglob("*")):
        if item.name.startswith("."):
            continue
        try:
            stat = item.stat()
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            items.append(
                DataDirItem(
                    name=item.name,
                    path=str(item.relative_to(data_path)),
                    is_dir=item.is_dir(),
                    size_bytes=stat.st_size if item.is_file() else 0,
                    modified_at=mtime,
                )
            )
        except Exception:
            continue

    return DataDirResponse(
        data_dir=str(data_path.resolve()),
        items=items,
    )


# =========================================================================
# Future Milestone Stubs
# =========================================================================

@router.post("/jobs/generate", response_model=JobResponse, status_code=status.HTTP_501_NOT_IMPLEMENTED)
def generate_moments_job(request: GenerateRequest):
    """
    Generate moments video job stub. Implementation scheduled for Milestone 8.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Pipeline orchestration will be activated in Milestone 8 (Parallel Ingestion Pipeline).",
    )


@router.get("/jobs/{job_id}/events", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def stream_job_events(job_id: str):
    """
    Server-Sent Events progress stream stub. Implementation scheduled for Milestone 8.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Real-time SSE event streaming will be activated in Milestone 8.",
    )


@router.get("/jobs/{job_id}/download", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def download_rendered_video(job_id: str):
    """
    Rendered video download stub. Implementation scheduled for Milestone 10.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Rendered video download will be activated in Milestone 10.",
    )
