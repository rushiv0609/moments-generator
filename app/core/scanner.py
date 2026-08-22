"""
Corpus Scanner & File Discovery Engine for Local AI Moments Generator.
Recursively traverses media directories, detects media formats, performs two-tier hashing,
detects content duplicates, extracts temporal metadata, and updates the SQLite manifest.
"""

import mimetypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

from app.db.manifest import ManifestDB, compute_fast_hash, compute_content_hash
from app.db.models import FileRecord, FileStatus
from app.core.metadata import extract_metadata, MetadataResult

# Supported file extension mappings
SUPPORTED_IMAGE_EXTS: Set[str] = {
    ".heic", ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"
}

SUPPORTED_VIDEO_EXTS: Set[str] = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv"
}

# Ensure HEIC MIME type is registered
mimetypes.add_type("image/heic", ".heic")
mimetypes.add_type("image/heif", ".heif")
mimetypes.add_type("video/quicktime", ".mov")


@dataclass
class ScannedFile:
    record: FileRecord
    is_new: bool = False
    is_modified: bool = False
    is_skipped: bool = False
    is_dedup_reused: bool = False


@dataclass
class ScanSummary:
    corpus_path: str
    total_found: int
    new_files: int
    modified_files: int
    skipped_files: int
    dedup_reused_files: int
    deleted_files: int
    images_count: int
    videos_count: int
    files_to_process: List[FileRecord] = field(default_factory=list)
    all_files: List[FileRecord] = field(default_factory=list)


def get_file_type_and_mime(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Determine media type ('image' or 'video') and MIME type using magic bytes with extension fallback."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except Exception:
        header = b""

    true_type = None
    if header.startswith(b"\xff\xd8\xff"):
        true_type = "image"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        true_type = "image"
    elif len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heca", b"mif1", b"msf1"):
            true_type = "image"
        elif brand in (b"qt  ", b"mp41", b"mp42", b"isom", b"avc1", b"M4V "):
            true_type = "video"

    suffix = path.suffix.lower()
    
    if true_type == "image" or (true_type is None and suffix in SUPPORTED_IMAGE_EXTS):
        mime, _ = mimetypes.guess_type(path.name)
        return "image", mime or "image/jpeg"
    elif true_type == "video" or (true_type is None and suffix in SUPPORTED_VIDEO_EXTS):
        mime, _ = mimetypes.guess_type(path.name)
        return "video", mime or "video/mp4"

    return None, None


def scan_corpus(
    corpus_path: str,
    manifest: ManifestDB,
    force_reindex: bool = False,
) -> ScanSummary:
    """
    Recursively scans the corpus directory, performs two-tier hashing,
    populates the manifest, and returns a summary of discovered files.
    """
    root = Path(corpus_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Corpus path '{corpus_path}' does not exist or is not a directory.")

    discovered_paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            suffix = p.suffix.lower()
            if suffix in SUPPORTED_IMAGE_EXTS or suffix in SUPPORTED_VIDEO_EXTS:
                discovered_paths.append(p)

    active_paths_str = [str(p) for p in discovered_paths]
    
    # 1. Prune files that were removed from the disk
    deleted_info = manifest.remove_deleted_files(active_paths_str)
    deleted_count = len(deleted_info)

    new_count = 0
    modified_count = 0
    skipped_count = 0
    dedup_count = 0
    images_count = 0
    videos_count = 0
    files_to_process: List[FileRecord] = []
    all_records: List[FileRecord] = []

    for path in discovered_paths:
        abs_path_str = str(path)
        stat = path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        file_type, mime_type = get_file_type_and_mime(path)

        if not file_type:
            continue

        if file_type == "image":
            images_count += 1
        else:
            videos_count += 1

        fast_hash = compute_fast_hash(file_size, mtime)
        existing = manifest.lookup(abs_path_str)

        # Case 1: Existing file with matching fast hash and not forced re-index
        if existing and existing.file_hash == fast_hash and not force_reindex:
            if existing.status == FileStatus.INDEXED:
                skipped_count += 1
                all_records.append(existing)
                continue
            else:
                # Needs processing from current state
                files_to_process.append(existing)
                all_records.append(existing)
                continue

        # Case 2: New or Modified file (Fast hash missed or force_reindex)
        is_modified = existing is not None and existing.file_hash != fast_hash
        content_hash = compute_content_hash(abs_path_str)

        # Check for identical content in an existing indexed file (Content Deduplication)
        donor = manifest.lookup_by_content_hash(content_hash)
        if donor and donor.status == FileStatus.INDEXED and donor.qdrant_point_ids:
            # Re-use existing embeddings without re-embedding
            meta = extract_metadata(abs_path_str, file_type)
            record = manifest.upsert_file(
                file_path=abs_path_str,
                file_hash=fast_hash,
                content_hash=content_hash,
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type,
                creation_timestamp=meta.creation_timestamp,
                timestamp_source=meta.timestamp_source,
                duration_seconds=meta.duration_seconds,
                status=FileStatus.INDEXED,
            )
            manifest.update_embedding_info(
                file_id=record.id,
                qdrant_point_ids=donor.qdrant_point_ids,
                model_name=donor.model_name or "siglip2",
                model_version=donor.model_version or "1.0",
            )
            dedup_count += 1
            all_records.append(record)
            continue

        # Case 3: Truly new or modified file needing full processing
        if is_modified:
            modified_count += 1
        else:
            new_count += 1

        meta = extract_metadata(abs_path_str, file_type)
        record = manifest.upsert_file(
            file_path=abs_path_str,
            file_hash=fast_hash,
            content_hash=content_hash,
            file_size=file_size,
            file_type=file_type,
            mime_type=mime_type,
            creation_timestamp=meta.creation_timestamp,
            timestamp_source=meta.timestamp_source,
            duration_seconds=meta.duration_seconds,
            status=FileStatus.SCANNED,
        )
        files_to_process.append(record)
        all_records.append(record)

    return ScanSummary(
        corpus_path=str(root),
        total_found=len(discovered_paths),
        new_files=new_count,
        modified_files=modified_count,
        skipped_files=skipped_count,
        dedup_reused_files=dedup_count,
        deleted_files=deleted_count,
        images_count=images_count,
        videos_count=videos_count,
        files_to_process=files_to_process,
        all_files=all_records,
    )
