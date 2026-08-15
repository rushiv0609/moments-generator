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


class DataDirItem(BaseModel):
    name: str
    path: str
    is_dir: bool
    size_bytes: int
    modified_at: str


class DataDirResponse(BaseModel):
    data_dir: str
    items: List[DataDirItem]


class GenerateRequest(BaseModel):
    corpus_path: str = Field(..., description="Absolute path to media folder on the local machine")
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


class ProgressEvent(BaseModel):
    job_id: str
    stage: Literal["SCANNING", "INDEXING", "CURATING", "RENDERING", "COMPLETED", "FAILED"]
    progress_pct: float = Field(ge=0.0, le=100.0)
    details: str
    eta_seconds: Optional[float] = None
    error: Optional[str] = None
