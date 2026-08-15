"""
Data models and state representations for SQLite Manifest.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class FileStatus(str, Enum):
    PENDING = "PENDING"
    SCANNED = "SCANNED"
    EXTRACTED = "EXTRACTED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    ERROR = "ERROR"


VALID_TRANSITIONS = {
    FileStatus.PENDING: {FileStatus.SCANNED, FileStatus.ERROR},
    FileStatus.SCANNED: {FileStatus.EXTRACTED, FileStatus.EMBEDDED, FileStatus.INDEXED, FileStatus.ERROR, FileStatus.PENDING},
    FileStatus.EXTRACTED: {FileStatus.EMBEDDED, FileStatus.INDEXED, FileStatus.ERROR, FileStatus.SCANNED},
    FileStatus.EMBEDDED: {FileStatus.INDEXED, FileStatus.ERROR, FileStatus.SCANNED},
    FileStatus.INDEXED: {FileStatus.SCANNED, FileStatus.PENDING, FileStatus.ERROR},
    FileStatus.ERROR: {FileStatus.PENDING, FileStatus.SCANNED, FileStatus.EXTRACTED, FileStatus.EMBEDDED, FileStatus.INDEXED},
}


@dataclass
class FileRecord:
    id: Optional[int]
    file_path: str
    file_hash: str
    file_size: int
    file_type: str  # 'image' | 'video'
    content_hash: Optional[str] = None
    mime_type: Optional[str] = None
    creation_timestamp: Optional[float] = None
    timestamp_source: Optional[str] = None  # 'exif' | 'container' | 'filesystem'
    duration_seconds: Optional[float] = None
    status: FileStatus = FileStatus.PENDING
    error_message: Optional[str] = None
    frame_count: Optional[int] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    qdrant_point_ids: List[str] = field(default_factory=list)
    embedded_at: Optional[float] = None
    scanned_at: Optional[float] = None
    updated_at: Optional[float] = None


@dataclass
class TimelineSegmentRecord:
    id: Optional[int]
    job_id: str
    position: int
    file_id: int
    segment_type: str  # 'image' | 'video_clip'
    duration: float
    start_offset: Optional[float] = None
    similarity_score: Optional[float] = None
    time_bucket: Optional[int] = None
