"""
Pydantic Schemas for Local AI Moments Generator API.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    available: bool
    details: Optional[str] = None
    version: Optional[str] = None


class ModelStatus(BaseModel):
    name: str
    backend: str
    precision: str
    loaded: bool
    details: Optional[str] = None


class QdrantStatus(BaseModel):
    connected: bool
    host: str
    port: int
    collections: int = 0
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model: ModelStatus
    qdrant: QdrantStatus
    ffmpeg: ComponentStatus
    system_memory: Dict[str, Any]
    active_workspace: Optional[Dict[str, Any]] = None


class FolderPickerRequest(BaseModel):
    prompt: Optional[str] = "Select Folder"
    default_path: Optional[str] = None


class FolderPickerResponse(BaseModel):
    selected_path: Optional[str] = None
    cancelled: bool = False


class SetWorkspaceRequest(BaseModel):
    workspace_path: str = Field(..., description="Absolute path for project workspace directory")
    corpus_path: Optional[str] = Field(default=None, description="Optional default media corpus directory")


class WorkspaceResponse(BaseModel):
    workspace_dir: str
    corpus_dir: Optional[str] = None
    manifest_db_path: str
    qdrant_storage_path: str
    exports_dir: str
    cache_dir: str
    created_at: float
    updated_at: float
    total_files: int = 0
    indexed_files: int = 0
    total_vectors: int = 0


class DataDirItem(BaseModel):
    name: str
    path: str
    is_dir: bool
    size_bytes: int
    modified_at: str


class DataDirResponse(BaseModel):
    data_dir: str
    items: List[DataDirItem]


class ScanRequest(BaseModel):
    corpus_path: str = Field(..., description="Absolute path to media folder to scan")
    workspace_path: Optional[str] = Field(default=None, description="Absolute path for project workspace directory")
    force_reindex: bool = Field(default=False, description="Whether to bypass fast hash cache")


class ScannedFileItem(BaseModel):
    id: Optional[int]
    file_path: str
    file_type: str
    mime_type: Optional[str]
    file_size: int
    status: str
    creation_timestamp: Optional[float]
    timestamp_source: Optional[str]
    duration_seconds: Optional[float]


class ScanResponse(BaseModel):
    corpus_path: str
    total_found: int
    new_files: int
    modified_files: int
    skipped_files: int
    dedup_reused_files: int
    deleted_files: int
    images_count: int
    videos_count: int
    files: List[ScannedFileItem]


class GenerateRequest(BaseModel):
    corpus_path: str = Field(..., description="Absolute path to media folder on the local machine")
    workspace_path: Optional[str] = Field(default=None, description="Absolute path for project workspace directory")
    prompt: str = Field(..., min_length=1, description="Natural language search prompt describing the moment")
    target_duration_seconds: int = Field(default=60, ge=5, le=300, description="Target duration of highlight video in seconds")
    aspect_ratio: Literal["1:1", "16:9", "9:16"] = Field(default="1:1", description="Target video aspect ratio")
    force_reindex: bool = Field(default=False, description="Whether to bypass manifest checkpoint cache and recompute embeddings")


class JobResponse(BaseModel):
    job_id: str
    status: Literal["QUEUED", "SCANNING", "INDEXING", "CURATING", "RENDERING", "COMPLETED", "FAILED"]
    corpus_path: str
    prompt: str
    created_at: str
    message: str


class IndexJobRequest(BaseModel):
    workspace_path: Optional[str] = Field(default=None, description="Project workspace directory")
    corpus_path: Optional[str] = Field(default=None, description="Media folder path to scan & index")
    force_reindex: bool = Field(default=False, description="Force re-extraction and re-embedding of already indexed files")


class IndexJobResponse(BaseModel):
    job_id: str
    status: str
    workspace_dir: str
    corpus_dir: Optional[str] = None
    message: str
    created_at: float


class ProgressEvent(BaseModel):
    job_id: str
    stage: Literal["SCANNING", "INDEXING", "CURATING", "RENDERING", "COMPLETED", "FAILED"]
    progress_pct: float = Field(ge=0.0, le=100.0)
    details: str
    message: Optional[str] = None
    eta_seconds: Optional[float] = None
    error: Optional[str] = None


class WorkspaceSearchResultItem(BaseModel):
    point_id: str
    score: float
    file_path: str
    file_name: str
    file_type: str
    frame_index: int
    source_offset: float
    granularity: str = "frame"
    scene_id: Optional[int] = None
    scene_start: Optional[float] = None
    scene_end: Optional[float] = None
    is_scene_representative: bool = False
    media_url: str


class WorkspaceSearchResponse(BaseModel):
    query: str
    workspace_dir: str
    total_results: int
    results: List[WorkspaceSearchResultItem]

