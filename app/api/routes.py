"""
FastAPI Route Handlers for Local AI Moments Generator.
"""

import os
import shutil
import subprocess
import datetime
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, status, File, Form, UploadFile
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
    ScanRequest,
    ScanResponse,
    ScannedFileItem,
    FolderPickerRequest,
    FolderPickerResponse,
    SetWorkspaceRequest,
    WorkspaceResponse,
)
from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB
from app.core.embedder import create_embedder, EmbedderInterface
from app.core.extractor import decode_image
from app.core.workspace import get_workspace_manager, open_native_finder_picker, WorkspaceManager

router = APIRouter()

# Cached embedder instance for real-time interactive testing
_embedder: Optional[EmbedderInterface] = None


def get_active_embedder() -> EmbedderInterface:
    global _embedder
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


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
    """Check connectivity to Qdrant vector database (Remote or Embedded)."""
    settings = get_settings()
    try:
        qdrant_db = QdrantVectorDB.create(settings=settings)
        collections = qdrant_db.client.get_collections()
        count = len(collections.collections)
        mode_str = "Remote Docker" if qdrant_db.mode == "remote" else "Embedded Disk Engine"
        return QdrantStatus(
            connected=True,
            host=settings.QDRANT_HOST if qdrant_db.mode == "remote" else "local",
            port=settings.QDRANT_PORT if qdrant_db.mode == "remote" else 0,
            collections=count,
            details=f"Connected successfully ({mode_str})",
        )
    except Exception as e:
        return QdrantStatus(
            connected=False,
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collections=0,
            details=f"Cannot initialize Qdrant: {str(e)}",
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
    System health check. Inspects ML backend, Qdrant connectivity, FFmpeg, and Active Workspace.
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

    workspace_mgr = get_workspace_manager()
    workspace_data = None
    if workspace_mgr.is_active:
        try:
            workspace_data = workspace_mgr.get_workspace_info().to_dict()
        except Exception:
            workspace_data = None

    return HealthResponse(
        status=overall_status,
        model=model_status,
        qdrant=qdrant_status,
        ffmpeg=ffmpeg_status,
        system_memory=mem_info,
        active_workspace=workspace_data,
    )


# =========================================================================
# Project Workspace Management
# =========================================================================

@router.post("/workspace/select-folder", response_model=FolderPickerResponse)
def select_folder_dialog(request: Optional[FolderPickerRequest] = None) -> FolderPickerResponse:
    """
    Trigger native macOS Finder folder selection window.
    """
    prompt = request.prompt if request and request.prompt else "Select Folder"
    default_p = request.default_path if request else None
    chosen = open_native_finder_picker(prompt=prompt, default_path=default_p)
    return FolderPickerResponse(
        selected_path=chosen,
        cancelled=chosen is None,
    )


@router.post("/workspace/set", response_model=WorkspaceResponse)
def set_active_workspace(request: SetWorkspaceRequest) -> WorkspaceResponse:
    """
    Activate a Project Workspace directory. Initializes .moments/ internal structure.
    """
    try:
        workspace_mgr = get_workspace_manager()
        info = workspace_mgr.set_workspace(
            workspace_path=request.workspace_path,
            corpus_path=request.corpus_path,
        )
        return WorkspaceResponse(**info.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to activate project workspace: {str(e)}",
        )


@router.get("/workspace/current", response_model=WorkspaceResponse)
def get_current_workspace() -> WorkspaceResponse:
    """
    Retrieve active Project Workspace path, database mappings, and counts.
    """
    workspace_mgr = get_workspace_manager()
    if not workspace_mgr.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active project workspace set.",
        )
    info = workspace_mgr.get_workspace_info()
    return WorkspaceResponse(**info.to_dict())



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


@router.post("/debug/embed")
async def debug_embed(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Diagnostic playground endpoint: Compute multimodal SigLIP 2 embeddings
    for arbitrary user-provided text prompts or uploaded/pasted images.
    """
    if not text and not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'text' or 'file' must be provided.",
        )

    embedder = get_active_embedder()
    res: Dict[str, Any] = {
        "model_info": embedder.model_info(),
        "text": text,
        "filename": file.filename if file else None,
    }

    text_vec = None
    if text and text.strip():
        text_vec = embedder.embed_text(text.strip())
        res["text_embedding"] = text_vec.tolist()
        res["text_dim"] = len(text_vec)

    img_vec = None
    if file:
        # Save upload to a temp file to feed into our ImageIO/Pillow pipeline
        suffix = Path(file.filename or "temp.jpg").suffix
        if not suffix:
            suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        try:
            frame_data = decode_image(tmp_path, target_size=224)
            img_vec = embedder.embed_images([frame_data.pixels])[0]
            res["image_embedding"] = img_vec.tolist()
            res["image_dim"] = len(img_vec)
            res["image_shape"] = list(frame_data.pixels.shape)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if text_vec is not None and img_vec is not None:
        similarity = float(np.dot(text_vec, img_vec))
        res["similarity"] = round(similarity, 4)

    return res



# =========================================================================
# Milestone 4: Corpus Scanner & Metadata Extraction
# =========================================================================

@router.post("/scan", response_model=ScanResponse)
def scan_media_corpus(request: ScanRequest) -> ScanResponse:
    """
    Recursively scan a media directory, extract EXIF/container metadata,
    compute two-tier hashes, and sync state with SQLite manifest.
    """
    settings = get_settings()
    try:
        workspace_mgr = get_workspace_manager()
        if workspace_mgr.is_active:
            manifest = workspace_mgr.get_manifest_db()
        else:
            manifest = ManifestDB.open_or_create(request.corpus_path, data_dir=settings.DATA_DIR)

        from app.core.scanner import scan_corpus
        summary = scan_corpus(request.corpus_path, manifest, force_reindex=request.force_reindex)

        items = [
            ScannedFileItem(
                id=f.id,
                file_path=f.file_path,
                file_type=f.file_type,
                mime_type=f.mime_type,
                file_size=f.file_size,
                status=f.status.value,
                creation_timestamp=f.creation_timestamp,
                timestamp_source=f.timestamp_source,
                duration_seconds=f.duration_seconds,
            )
            for f in summary.all_files
        ]

        return ScanResponse(
            corpus_path=summary.corpus_path,
            total_found=summary.total_found,
            new_files=summary.new_files,
            modified_files=summary.modified_files,
            skipped_files=summary.skipped_files,
            dedup_reused_files=summary.dedup_reused_files,
            deleted_files=summary.deleted_files,
            images_count=summary.images_count,
            videos_count=summary.videos_count,
            files=items,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Scanning failed: {str(e)}")


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
