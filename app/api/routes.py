"""
FastAPI Route Handlers for Local AI Moments Generator.
"""

import os
import io
import shutil
import urllib.parse
import subprocess
import datetime
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

import numpy as np
from fastapi import APIRouter, HTTPException, status, File, Form, UploadFile, Query
from fastapi.responses import StreamingResponse, FileResponse, Response
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
    IndexJobRequest,
    IndexJobResponse,
    WorkspaceSearchResponse,
    WorkspaceSearchResultItem,
    DirectorModelsResponse,
    DirectorModelItem,
    RenderVideoRequest,
    RenderVideoResponse,
)
from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB
from app.core.embedder import create_embedder, EmbedderInterface
from app.core.extractor import decode_image
from app.core.workspace import get_workspace_manager, open_native_finder_picker, WorkspaceManager
from app.core.jobs import get_job_manager, JobStatus

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
        workspace_mgr = get_workspace_manager()
        if workspace_mgr.is_active and workspace_mgr._qdrant_db:
            qdrant_db = workspace_mgr.get_qdrant_db()
        else:
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


@router.get("/debug/logs")
def get_debug_logs(lines: int = Query(default=100, ge=1, le=1000)):
    """
    Retrieve the latest log lines from the server log file.
    """
    settings = get_settings()
    log_file = Path(settings.DATA_DIR).resolve() / "moments_server.log"
    if not log_file.exists():
        return {"log_file": str(log_file), "lines_count": 0, "logs": []}

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            recent_lines = [l.rstrip("\r\n") for l in all_lines[-lines:]]
            return {
                "log_file": str(log_file),
                "lines_count": len(recent_lines),
                "logs": recent_lines,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {e}")




# =========================================================================
# Milestone 4: Corpus Scanner & Metadata Extraction
# =========================================================================

@router.post("/scan", response_model=ScanResponse)
def scan_media_corpus(request: ScanRequest) -> ScanResponse:
    """
    Recursively scan a media directory, extract EXIF/container metadata,
    compute two-tier hashes, and sync state with SQLite manifest inside the active Project Workspace.
    """
    try:
        workspace_mgr = get_workspace_manager()
        if request.workspace_path:
            workspace_mgr.set_workspace(request.workspace_path, corpus_path=request.corpus_path)
            manifest = workspace_mgr.get_manifest_db()
        elif workspace_mgr.is_active:
            manifest = workspace_mgr.get_manifest_db()
        else:
            default_ws = Path.home() / "Moments_Projects" / "Default"
            workspace_mgr.set_workspace(default_ws, corpus_path=request.corpus_path)
            manifest = workspace_mgr.get_manifest_db()

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
# Milestone 8: Parallel Ingestion Pipeline & Background Worker
# =========================================================================

@router.post("/jobs/index", response_model=IndexJobResponse)
def submit_indexing_job(request: Optional[IndexJobRequest] = None) -> IndexJobResponse:
    """
    Start an asynchronous background job to decode, embed, and index media
    files inside the active project workspace.
    """
    workspace_mgr = get_workspace_manager()
    ws_path = request.workspace_path if request and request.workspace_path else (str(workspace_mgr.workspace_path) if workspace_mgr.is_active else None)
    corpus_p = request.corpus_path if request and request.corpus_path else (str(workspace_mgr.corpus_path) if workspace_mgr.corpus_path else None)
    force_reidx = request.force_reindex if request else False

    if not ws_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No project workspace specified. Please select a project workspace directory first.",
        )

    job_mgr = get_job_manager()
    job = job_mgr.submit_indexing_job(
        workspace_dir=ws_path,
        corpus_dir=corpus_p,
        force_reindex=force_reidx,
    )

    return IndexJobResponse(
        job_id=job.id,
        status=job.status.value,
        workspace_dir=job.workspace_dir,
        corpus_dir=job.corpus_dir,
        message="Background indexing job started.",
        created_at=job.created_at,
    )


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Retrieve live progress, status, and telemetry for a background job.
    """
    job_mgr = get_job_manager()
    job = job_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found.")
    return job.to_dict()


@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Real-time Server-Sent Events (SSE) stream for live progress bars and telemetry.
    """
    job_mgr = get_job_manager()
    job = job_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found.")

    return StreamingResponse(
        job_mgr.subscribe(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel an active background job.
    """
    job_mgr = get_job_manager()
    success = job_mgr.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job could not be cancelled.")
    return {"job_id": job_id, "status": "CANCELLED", "message": "Job cancellation requested."}


# =========================================================================
# Media Serving & Visual Semantic Search Explorer (Active Workspace)
# =========================================================================

def find_live_photo_companion(photo_path: Path) -> Optional[Path]:
    """Find companion Live Photo MOV/MP4 video for a photo file."""
    if photo_path.suffix.lower() not in [".heic", ".heif", ".jpg", ".jpeg", ".png"]:
        return None
    for ext in [".MOV", ".mov", ".mp4", ".MP4"]:
        companion = photo_path.with_suffix(ext)
        if companion.exists() and companion.is_file():
            return companion
    return None


@router.api_route("/media/file", methods=["GET", "HEAD"])
def get_media_file(
    path: str = Query(..., description="Absolute path to media file"),
    offset: Optional[float] = Query(default=None, description="Timestamp offset in seconds for video frame preview"),
    thumbnail: bool = Query(default=False, description="Extract a JPEG poster frame at offset"),
):
    """
    Safely stream local photos/videos to the browser.
    Extracts exact frame thumbnails at offset seconds and converts Apple HEIC on-the-fly.
    Supports HTTP Range requests and HEAD probing for HTML5 video players.
    """
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found on disk.")

    suffix = p.suffix.lower()

    # Exact video frame thumbnail extraction (only when thumbnail=True or offset is explicitly specified for image preview)
    if suffix in [".mp4", ".mov", ".m4v", ".avi", ".mkv"] and thumbnail:
        try:
            import cv2
            cap = cv2.VideoCapture(str(p))
            target_sec = max(0.0, float(offset or 0.0))
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return Response(
                    content=encoded.tobytes(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"}
                )
        except Exception as e:
            pass

    if suffix in [".heic", ".heif"]:
        try:
            with Image.open(str(p)) as img:
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return Response(
                    content=buf.getvalue(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"}
                )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to decode HEIC: {e}")
    elif suffix in [".jpg", ".jpeg"]:
        return FileResponse(str(p), media_type="image/jpeg", headers={"Accept-Ranges": "bytes"})
    elif suffix in [".png"]:
        return FileResponse(str(p), media_type="image/png", headers={"Accept-Ranges": "bytes"})
    elif suffix in [".webp"]:
        return FileResponse(str(p), media_type="image/webp", headers={"Accept-Ranges": "bytes"})
    elif suffix in [".mp4", ".m4v", ".mov"]:
        # video/mp4 with Accept-Ranges allows Safari and Chrome to play Apple MOV/HEVC video smoothly
        return FileResponse(
            str(p),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
            }
        )
    else:
        return FileResponse(str(p), headers={"Accept-Ranges": "bytes"})


@router.api_route("/media/thumbnail", methods=["GET", "HEAD"])
def get_media_thumbnail(
    path: str = Query(..., description="Absolute path to media file"),
    offset: Optional[float] = Query(default=0.0, description="Timestamp offset in seconds for video frame extraction"),
):
    """
    Convenience endpoint for media thumbnail previews.
    Extracts video frame at offset or serves converted image.
    """
    return get_media_file(path=path, offset=offset, thumbnail=True)


@router.api_route("/media/clip", methods=["GET", "HEAD"])
def get_media_clip_preview(
    path: str = Query(..., description="Absolute path to video file"),
    start_offset: float = Query(default=0.0, description="Start offset in seconds"),
    duration: float = Query(default=3.0, description="Clip duration in seconds"),
    fps: int = Query(default=12, description="Target animation FPS"),
    max_width: int = Query(default=640, description="Max width for preview frames"),
):
    """
    Extract and stream an animated WebP preview clip spanning [start_offset, start_offset + duration].
    Provides 100% universal browser-compatible motion playback for any video codec.
    """
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found on disk.")

    suffix = p.suffix.lower()
    if suffix not in [".mp4", ".mov", ".m4v", ".avi", ".mkv"]:
        # If it's an image, check for companion Live Photo MOV
        companion = find_live_photo_companion(p)
        if companion:
            p = companion
        else:
            return get_media_file(path=str(p))

    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_offset) * 1000.0)

        total_frames = int(max(1.0, min(10.0, duration)) * fps)
        step = max(1, int(src_fps / fps))

        frames = []
        count = 0
        while len(frames) < total_frames and count < total_frames * step:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                h, w = frame.shape[:2]
                if w > max_width:
                    new_h = int(h * (max_width / w))
                    frame = cv2.resize(frame, (max_width, new_h))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            count += 1
        cap.release()

        if not frames:
            return get_media_thumbnail(path=str(p), offset=start_offset)

        buf = io.BytesIO()
        frame_duration = int(1000 / max(1, fps))
        frames[0].save(
            buf,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            quality=75,
        )
        return Response(
            content=buf.getvalue(),
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        logger.warning("Animated clip generation error: %s", e)
        return get_media_thumbnail(path=str(p), offset=start_offset)


@router.get("/media/info")
def get_media_info(path: str = Query(..., description="Absolute path to media file")) -> Dict[str, Any]:
    """
    Get detailed media information, codec support, and Live Photo companion detection.
    """
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found on disk.")

    suffix = p.suffix.lower()
    is_video = suffix in [".mp4", ".mov", ".m4v", ".avi", ".mkv"]
    companion = find_live_photo_companion(p) if not is_video else None

    return {
        "file_path": str(p),
        "file_name": p.name,
        "is_video": is_video,
        "has_live_photo": bool(companion),
        "live_photo_path": str(companion) if companion else None,
        "live_photo_url": f"/api/v1/media/file?path={companion}" if companion else None,
        "media_url": f"/api/v1/media/file?path={p}",
        "thumbnail_url": f"/api/v1/media/thumbnail?path={p}",
    }


@router.get("/workspace/search", response_model=WorkspaceSearchResponse)
def search_workspace_media(
    query: str = Query(..., min_length=1, description="Natural language semantic search query"),
    top_k: int = Query(default=12, ge=1, le=100, description="Max ranked results to return"),
    granularity: Optional[str] = Query(default="all", description="'all' | 'frame' | 'scene'"),
    file_type: Optional[str] = Query(default="all", description="'all' | 'image' | 'video'"),
):
    """
    Execute real-time semantic search against the active Project Workspace Qdrant vector database.
    Embeds the user's natural language query using SigLIP 2 and returns ranked results with thumbnails.
    """
    workspace_mgr = get_workspace_manager()
    if not workspace_mgr.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active project workspace set. Please activate a workspace first.",
        )

    qdrant = workspace_mgr.get_qdrant_db()
    embedder = get_active_embedder()

    # Compute text query embedding (L2 normalized 768-dim vector)
    query_vec = embedder.embed_text(query)

    granularity_val = None if granularity == "all" else granularity
    file_type_val = None if file_type == "all" else file_type

    results = qdrant.search(
        collection_name="media_embeddings",
        query_vector=query_vec,
        limit=top_k,
        granularity=granularity_val,
        file_type=file_type_val,
    )

    items = []
    for r in results:
        target_offset = r.source_offset if r.source_offset is not None else (r.scene_start or 0.0)

        if r.file_type == "video":
            thumb_url = f"/api/v1/media/file?path={r.file_path}&offset={target_offset}&thumbnail=true"
            playback_url = f"/api/v1/media/file?path={r.file_path}#t={target_offset}"
        else:
            thumb_url = f"/api/v1/media/file?path={r.file_path}"
            playback_url = f"/api/v1/media/file?path={r.file_path}"

        items.append(
            WorkspaceSearchResultItem(
                point_id=r.point_id,
                score=round(r.score, 4),
                file_path=r.file_path,
                file_name=Path(r.file_path).name,
                file_type=r.file_type,
                frame_index=r.frame_index,
                source_offset=r.source_offset,
                granularity=r.granularity,
                scene_id=r.scene_id,
                scene_start=r.scene_start,
                scene_end=r.scene_end,
                is_scene_representative=r.is_scene_representative,
                media_url=f"/api/v1/media/file?path={r.file_path}",
                thumbnail_url=thumb_url,
                playback_url=playback_url,
            )
        )

    return WorkspaceSearchResponse(
        query=query,
        workspace_dir=str(workspace_mgr.workspace_path),
        total_results=len(items),
        results=items,
    )


@router.get("/workspace/video/scenes")
def get_video_scenes_breakdown(
    file_path: str = Query(..., description="Absolute path to video file"),
):
    """
    Retrieve all PySceneDetect scene boundaries and 1-FPS frames for a specific video file.
    """
    workspace_mgr = get_workspace_manager()
    if not workspace_mgr.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active project workspace set.",
        )

    qdrant = workspace_mgr.get_qdrant_db()
    points = qdrant.get_points_by_file(collection_name="media_embeddings", file_path=file_path)

    # Group points by scene_id
    scenes_dict = {}
    frames_list = []

    for pt in points:
        if pt.granularity == "scene":
            s_id = pt.scene_id if pt.scene_id is not None else 0
            scenes_dict[s_id] = {
                "scene_id": s_id,
                "point_id": pt.point_id,
                "start_sec": pt.scene_start if pt.scene_start is not None else 0.0,
                "end_sec": pt.scene_end if pt.scene_end is not None else pt.duration_seconds or 0.0,
                "duration_sec": round((pt.scene_end or 0.0) - (pt.scene_start or 0.0), 2),
                "frame_count": pt.payload.get("scene_frame_count", 0),
                "thumbnail_url": f"/api/v1/media/file?path={file_path}&offset={pt.scene_start or 0.0}&thumbnail=true",
                "playback_url": f"/api/v1/media/file?path={file_path}#t={pt.scene_start or 0.0}",
            }
        else:
            frames_list.append({
                "point_id": pt.point_id,
                "frame_index": pt.frame_index,
                "source_offset": pt.source_offset,
                "scene_id": pt.scene_id,
                "thumbnail_url": f"/api/v1/media/file?path={file_path}&offset={pt.source_offset}&thumbnail=true",
                "playback_url": f"/api/v1/media/file?path={file_path}#t={pt.source_offset}",
            })

    # Sort scenes by start time
    sorted_scenes = sorted(scenes_dict.values(), key=lambda s: s["start_sec"])

    return {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "total_scenes": len(sorted_scenes),
        "total_frames": len(frames_list),
        "scenes": sorted_scenes,
        "frames": frames_list,
    }


@router.get("/workspace/similar")
def find_similar_media(
    point_id: str = Query(..., description="Qdrant point ID to find similar visual items for"),
    top_k: int = Query(default=12, ge=1, le=100, description="Max ranked similar results"),
    granularity: Optional[str] = Query(default="all", description="'all' | 'frame' | 'scene'"),
):
    """
    Find visually/semantically similar photos, frames, or scenes across the workspace given a point ID.
    Uses nearest-neighbor vector search on the underlying 768-dim SigLIP 2 embedding.
    """
    workspace_mgr = get_workspace_manager()
    if not workspace_mgr.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active project workspace set.",
        )

    qdrant = workspace_mgr.get_qdrant_db()
    point_data = qdrant.get_point(collection_name="media_embeddings", point_id=point_id)
    if not point_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Point {point_id} not found in vector database.")

    source_item, query_vec = point_data

    granularity_val = None if granularity == "all" else granularity
    results = qdrant.search(
        collection_name="media_embeddings",
        query_vector=query_vec,
        limit=top_k + 1,
        granularity=granularity_val,
    )

    # Filter out the source point itself
    filtered = [r for r in results if r.point_id != point_id][:top_k]

    items = []
    for r in filtered:
        target_offset = r.source_offset if r.source_offset is not None else (r.scene_start or 0.0)

        if r.file_type == "video":
            thumb_url = f"/api/v1/media/file?path={r.file_path}&offset={target_offset}&thumbnail=true"
            playback_url = f"/api/v1/media/file?path={r.file_path}#t={target_offset}"
        else:
            thumb_url = f"/api/v1/media/file?path={r.file_path}"
            playback_url = f"/api/v1/media/file?path={r.file_path}"

        items.append({
            "point_id": r.point_id,
            "score": round(r.score, 4),
            "file_path": r.file_path,
            "file_name": Path(r.file_path).name,
            "file_type": r.file_type,
            "frame_index": r.frame_index,
            "source_offset": r.source_offset,
            "granularity": r.granularity,
            "scene_id": r.scene_id,
            "scene_start": r.scene_start,
            "scene_end": r.scene_end,
            "is_scene_representative": r.is_scene_representative,
            "media_url": f"/api/v1/media/file?path={r.file_path}",
            "thumbnail_url": thumb_url,
            "playback_url": playback_url,
        })

    return {
        "source_point_id": point_id,
        "source_file_name": Path(source_item.file_path).name,
        "source_granularity": source_item.granularity,
        "source_source_offset": source_item.source_offset,
        "total_results": len(items),
        "results": items,
    }



# =========================================================================
# Milestone 9: Director Agent Generation & Model Discovery
# =========================================================================

@router.get("/director/models", response_model=DirectorModelsResponse)
def get_director_models():
    """
    List supported and installed local and cloud LLM models for the LangGraph Director Agent.
    Checks Ollama connectivity and Cloud API key configurations dynamically.
    """
    import os
    import urllib.request
    import json
    from app.core.ollama_service import ensure_ollama_running
    from app.config import get_settings

    settings = get_settings()
    gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or ""

    installed_names = []
    ollama_connected = ensure_ollama_running()
    if ollama_connected:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "MomentsApp/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                installed_names = [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass

    # Match exact installed tags or fallback to base name
    qwen_tag = next((n for n in installed_names if "qwen3.5" in n or "qwen3" in n), "qwen3.5:9b")
    gemma_e4b_tag = next((n for n in installed_names if "gemma4" in n and "e4b" in n), "gemma4:e4b")
    gemma_e2b_tag = next((n for n in installed_names if "gemma4" in n and "e2b" in n), "gemma4:e2b")

    models_list = [
        # Google Gemini Cloud Models
        # Google Cloud Gemini Models
        DirectorModelItem(
            name="gemini-3.5-flash",
            display_name="✨ Gemini 3.5 Flash (Ultra Fast & Recommended)",
            size_vram="Cloud API",
            provider="gemini",
            recommended=True,
            installed=bool(gemini_key),
            description="Ultra fast (~1.5s) multimodal intelligence for creative visual storytelling and search queries.",
        ),
        DirectorModelItem(
            name="gemini-3.7-flash",
            display_name="✨ Gemini 3.7 Flash (Hybrid Reasoning)",
            size_vram="Cloud API",
            provider="gemini",
            recommended=False,
            installed=bool(gemini_key),
            description="Google's flagship hybrid reasoning model for complex visual storytelling.",
        ),
        DirectorModelItem(
            name="gemini-3.6-flash",
            display_name="✨ Gemini 3.6 Flash (Fast Cloud)",
            size_vram="Cloud API",
            provider="gemini",
            recommended=False,
            installed=bool(gemini_key),
            description="Instant response time with full structured planning output.",
        ),
        DirectorModelItem(
            name="gemini-3.5-flash-lite",
            display_name="✨ Gemini 3.5 Flash Lite (Lightweight)",
            size_vram="Cloud API",
            provider="gemini",
            recommended=False,
            installed=bool(gemini_key),
            description="High throughput lightweight cloud model.",
        ),
        # Groq Cloud Models
        DirectorModelItem(
            name="groq:llama-3.3-70b-versatile",
            display_name="⚡ Groq Llama 3.3 70B (Sub-Second LPU)",
            size_vram="Cloud LPU",
            provider="groq",
            recommended=True,
            installed=bool(groq_key),
            description="Ultra-fast LPU hardware (~350+ tokens/sec) for instantaneous multi-cut generation.",
        ),
        DirectorModelItem(
            name="groq:llama-3.1-8b-instant",
            display_name="⚡ Groq Llama 3.1 8B (Instant)",
            size_vram="Cloud LPU",
            provider="groq",
            recommended=False,
            installed=bool(groq_key),
            description="Lightweight Groq model with lightning sub-300ms latency.",
        ),
        # Local Ollama Models
        DirectorModelItem(
            name=gemma_e4b_tag,
            display_name=f"🏠 Gemma 4 E4B {'(MLX)' if 'mlx' in gemma_e4b_tag else ''} (Fast Local)",
            size_vram="~4.8 GB",
            provider="ollama",
            recommended=True,
            installed=any("gemma4" in n and "e4b" in n for n in installed_names),
            description="Optimal on-device balance of speed (~7-15s) and cinematic structuring.",
        ),
        DirectorModelItem(
            name=qwen_tag,
            display_name=f"🏠 Qwen 3.5 9B VL {'(MLX)' if 'mlx' in qwen_tag else ''} (Deep Thinking)",
            size_vram="~6.0 GB",
            provider="ollama",
            recommended=False,
            installed=any("qwen3.5" in n or "qwen3" in n for n in installed_names),
            description="Deep internal chain-of-thought reasoning for complex multi-shot narratives.",
        ),
        DirectorModelItem(
            name=gemma_e2b_tag,
            display_name="🏠 Gemma 4 E2B (Ultra Lightweight)",
            size_vram="~2.4 GB",
            provider="ollama",
            recommended=False,
            installed=any("gemma4" in n and "e2b" in n for n in installed_names),
            description="Ultra fast on-device storyboard curation with near-zero memory pressure.",
        ),
        DirectorModelItem(
            name="mock",
            display_name="🧪 Mock Director (Instant Simulation)",
            size_vram="0 MB",
            provider="mock",
            recommended=False,
            installed=True,
            description="Runs instant deterministic state transitions without requiring local LLM weights or cloud keys.",
        ),
    ]

    # Select smart default model:
    # 1) If Gemini configured -> gemini-3.5-flash
    # 2) If Groq configured -> groq:llama-3.3-70b-versatile
    # 3) If Gemma E4B installed -> gemma_e4b_tag
    # 4) If Qwen installed -> qwen_tag
    # 5) Fallback -> first installed or mock
    default_model = "mock"
    if bool(gemini_key):
        default_model = "gemini-3.5-flash"
    elif bool(groq_key):
        default_model = "groq:llama-3.3-70b-versatile"
    elif any("gemma4" in n and "e4b" in n for n in installed_names):
        default_model = gemma_e4b_tag
    elif any("qwen3.5" in n or "qwen3" in n for n in installed_names):
        default_model = qwen_tag
    elif installed_names:
        default_model = installed_names[0]

    return DirectorModelsResponse(
        ollama_connected=ollama_connected,
        gemini_configured=bool(gemini_key),
        groq_configured=bool(groq_key),
        default_model=default_model,
        models=models_list,
    )


@router.post("/director/unload")
def unload_director_model(model_name: Optional[str] = Query(default=None, description="Optional specific model name to unload")) -> Dict[str, Any]:
    """
    Immediately stop/unload running Ollama local LLM models from GPU/VRAM to free system resources.
    The Ollama server daemon remains active and ready.
    """
    settings = get_settings()
    base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
    unloaded_models = []

    try:
        import httpx
        # If no specific model requested, query ps to find all currently loaded models and unload them
        if not model_name:
            ps_resp = httpx.get(f"{base_url}/api/ps", timeout=3.0)
            if ps_resp.status_code == 200:
                loaded = ps_resp.json().get("models", [])
                for m in loaded:
                    m_name = m.get("name") or m.get("model")
                    if m_name:
                        httpx.post(f"{base_url}/api/generate", json={"model": m_name, "keep_alive": 0}, timeout=5.0)
                        unloaded_models.append(m_name)
        else:
            httpx.post(f"{base_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5.0)
            unloaded_models.append(model_name)

        return {
            "status": "UNLOADED",
            "message": f"Successfully unloaded {len(unloaded_models)} local model(s) from GPU/VRAM.",
            "unloaded_models": unloaded_models,
        }
    except Exception as e:
        logger.debug("Notice while unloading models: %s", e)
        return {
            "status": "OK",
            "message": f"Unload request processed: {str(e)}",
            "unloaded_models": unloaded_models,
        }


@router.post("/jobs/generate", response_model=JobResponse)
def generate_moments_job(request: GenerateRequest):
    """
    Launch a LangGraph Director Agent video timeline curation job in the background.
    """
    workspace_mgr = get_workspace_manager()

    # Determine target workspace and corpus directories
    if request.workspace_path:
        workspace_dir = request.workspace_path
        corpus_dir = request.corpus_path
    elif workspace_mgr.is_active and workspace_mgr.workspace_path:
        workspace_dir = str(workspace_mgr.workspace_path)
        corpus_dir = request.corpus_path or (str(workspace_mgr.corpus_path) if workspace_mgr.corpus_path else None)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active workspace. Provide 'workspace_path' or initialize a workspace first.",
        )

    # Launch background generation job
    job_mgr = get_job_manager()
    job = job_mgr.start_generation_job(
        workspace_dir=workspace_dir,
        prompt=request.prompt,
        corpus_dir=corpus_dir,
        target_duration=request.target_duration_seconds,
        model_name=request.model_name,
        api_key=request.api_key,
        retrieval_mode=request.retrieval_mode,
        generate_alternatives=request.generate_alternatives,
    )

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        corpus_path=corpus_dir,
        prompt=request.prompt,
        created_at=job.created_at,
        message=job.message,
        workspace_dir=workspace_dir,
    )


@router.get("/workspaces/current/timelines")
def get_current_workspace_timelines():
    """
    Retrieve all curated timeline segments saved in the active workspace SQLite manifest.
    """
    workspace_mgr = get_workspace_manager()
    manifest = workspace_mgr.get_manifest_db()

    conn = manifest._get_connection()
    try:
        # Group by job_id
        cursor = conn.execute("""
            SELECT t.job_id, t.position, t.segment_type, t.duration, t.start_offset, t.similarity_score,
                   f.file_path, f.file_type, f.creation_timestamp, f.duration_seconds as total_duration
            FROM timeline_segments t
            JOIN files f ON t.file_id = f.id
            ORDER BY t.job_id, t.position ASC
        """)
        rows = cursor.fetchall()
        jobs_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            jid = r["job_id"]
            if jid not in jobs_map:
                jobs_map[jid] = []
            jobs_map[jid].append({
                "position": r["position"],
                "file_path": r["file_path"],
                "file_name": Path(r["file_path"]).name,
                "file_type": r["file_type"],
                "segment_type": r["segment_type"],
                "duration": r["duration"],
                "start_offset": r["start_offset"],
                "similarity_score": r["similarity_score"],
                "total_duration": r["total_duration"],
                "thumbnail_url": f"/api/v1/media/thumbnail?path={urllib.parse.quote(r['file_path'])}",
                "media_url": f"/api/v1/media/file?path={urllib.parse.quote(r['file_path'])}",
            })

        return {
            "workspace_dir": str(workspace_mgr.workspace_path) if workspace_mgr.is_active else None,
            "total_timeline_jobs": len(jobs_map),
            "timelines": jobs_map,
        }
    finally:
        conn.close()


@router.post("/director/render", response_model=RenderVideoResponse)
def render_director_video(request: RenderVideoRequest) -> RenderVideoResponse:
    """
    Compile and render a storyboard timeline into a final MP4 video montage.
    Applies Ken Burns motion to photos and cross-dissolve transitions.
    """
    if not request.storyboard:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot render an empty storyboard.",
        )

    settings = get_settings()
    workspace_mgr = get_workspace_manager()

    # Determine exports directory
    if workspace_mgr.is_active and workspace_mgr.workspace_path:
        exports_dir = (workspace_mgr.workspace_path / "exports").resolve()
    else:
        exports_dir = Path(settings.EXPORTS_DIR).resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Resolution from aspect ratio
    if request.aspect_ratio == "9:16":
        width, height = 1080, 1920
    else:
        width, height = 1920, 1080

    # Output file name
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = request.job_id or "director_cut"
    out_filename = request.output_filename or f"{file_prefix}_{timestamp_str}.mp4"
    out_path = exports_dir / out_filename

    from app.core.compiler import VideoCompiler
    compiler = VideoCompiler(
        output_width=width,
        output_height=height,
        fps=request.fps,
        transition_duration=request.transition_duration,
        use_hardware_accel=True,
    )

    try:
        render_meta = compiler.render(
            storyboard=request.storyboard,
            output_file=str(out_path),
        )

        stream_url = f"/api/v1/exports/{out_filename}"
        download_url = f"/api/v1/exports/{out_filename}?download=true"

        return RenderVideoResponse(
            status="COMPLETED",
            file_name=out_filename,
            file_path=str(out_path),
            download_url=download_url,
            stream_url=stream_url,
            file_size_bytes=render_meta.get("file_size_bytes", 0),
            duration_seconds=render_meta.get("duration_seconds", 0.0),
            resolution=render_meta.get("resolution", f"{width}x{height}"),
            fps=render_meta.get("fps", request.fps),
            total_segments=render_meta.get("total_segments", len(request.storyboard)),
        )
    except Exception as e:
        logger.error("Director video render failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rendering failed: {str(e)}",
        )


@router.get("/exports/{filename}")
def get_rendered_export_file(filename: str, download: bool = False):
    """
    Stream or download a rendered MP4 video file from the exports directory.
    Supports HTTP Range requests for instant HTML5 video seeking.
    """
    settings = get_settings()
    workspace_mgr = get_workspace_manager()

    possible_dirs = []
    if workspace_mgr.is_active and workspace_mgr.workspace_path:
        possible_dirs.append((workspace_mgr.workspace_path / "exports").resolve())
    possible_dirs.append(Path(settings.EXPORTS_DIR).resolve())
    possible_dirs.append(Path("./exports").resolve())

    target_file = None
    for p_dir in possible_dirs:
        candidate = (p_dir / filename).resolve()
        if candidate.exists() and candidate.is_file():
            target_file = candidate
            break

    if not target_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rendered export file '{filename}' not found.",
        )

    media_type = "video/mp4"
    if download:
        return FileResponse(
            str(target_file),
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    return FileResponse(
        str(target_file),
        media_type=media_type,
        filename=filename,
    )


@router.get("/jobs/{job_id}/download")
def download_rendered_video(job_id: str):
    """
    Download rendered video for a completed generation job.
    """
    settings = get_settings()
    workspace_mgr = get_workspace_manager()

    possible_dirs = []
    if workspace_mgr.is_active and workspace_mgr.workspace_path:
        possible_dirs.append((workspace_mgr.workspace_path / "exports").resolve())
    possible_dirs.append(Path(settings.EXPORTS_DIR).resolve())

    # Look for any file starting with job_id
    for p_dir in possible_dirs:
        if p_dir.exists():
            for f in p_dir.glob(f"{job_id}*.mp4"):
                return FileResponse(
                    str(f),
                    media_type="video/mp4",
                    filename=f.name,
                    headers={"Content-Disposition": f"attachment; filename={f.name}"},
                )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No rendered video found for job ID '{job_id}'. Please render it first.",
    )


