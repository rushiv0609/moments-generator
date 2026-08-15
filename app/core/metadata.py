"""
Metadata Extraction Engine for Local AI Moments Generator.
Extracts temporal metadata (EXIF for photos, container duration & timestamps for videos)
with best-effort graceful fallbacks.
"""

import os
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ExifTags
import pillow_heif
import exifread
import cv2

# Register HEIC opener for Pillow
pillow_heif.register_heif_opener()


@dataclass
class MetadataResult:
    creation_timestamp: Optional[float]  # Unix epoch timestamp
    timestamp_source: str               # 'exif' | 'container' | 'filesystem'
    duration_seconds: Optional[float]    # Video duration in seconds, None for images
    width: Optional[int] = None
    height: Optional[int] = None


def parse_exif_date_str(date_str: str) -> Optional[float]:
    """Parse EXIF date string formats (e.g. '2025:08:15 14:30:00') into Unix epoch timestamp."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(date_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def extract_image_metadata(file_path: str) -> MetadataResult:
    """Extract EXIF creation date and dimensions from photo files."""
    path = Path(file_path)
    creation_ts = None
    source = "filesystem"
    width, height = None, None

    # 1. Try exifread first (handles HEIC, JPEG, TIFF raw EXIF tags)
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="DateTimeOriginal")
            date_tag = (
                tags.get("EXIF DateTimeOriginal")
                or tags.get("EXIF DateTimeDigitized")
                or tags.get("Image DateTime")
            )
            if date_tag:
                ts = parse_exif_date_str(str(date_tag))
                if ts:
                    creation_ts = ts
                    source = "exif"
    except Exception:
        pass

    # 2. Try Pillow as secondary EXIF & dimension source
    try:
        with Image.open(path) as img:
            width, height = img.size
            if not creation_ts:
                exif = img.getexif()
                if exif:
                    # Common EXIF date tag IDs: 36867 (DateTimeOriginal), 36868 (DateTimeDigitized), 306 (DateTime)
                    for tag_id in (36867, 36868, 306):
                        val = exif.get(tag_id)
                        if val:
                            ts = parse_exif_date_str(str(val))
                            if ts:
                                creation_ts = ts
                                source = "exif"
                                break
    except Exception:
        pass

    # 3. Fallback to filesystem timestamps (macOS APFS st_birthtime or st_mtime)
    if not creation_ts:
        stat = path.stat()
        # On macOS APFS, st_birthtime is the file creation date
        creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)
        source = "filesystem"

    return MetadataResult(
        creation_timestamp=creation_ts,
        timestamp_source=source,
        duration_seconds=None,
        width=width,
        height=height,
    )


def extract_video_metadata(file_path: str) -> MetadataResult:
    """Extract duration, dimensions, and timestamps from video files using OpenCV / container headers."""
    path = Path(file_path)
    duration_sec = None
    width, height = None, None
    creation_ts = None
    source = "filesystem"

    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps > 0 and frame_count > 0:
                duration_sec = frame_count / fps
            cap.release()
    except Exception:
        pass

    # Filesystem creation time
    stat = path.stat()
    creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)

    return MetadataResult(
        creation_timestamp=creation_ts,
        timestamp_source=source,
        duration_seconds=duration_sec,
        width=width,
        height=height,
    )


def extract_metadata(file_path: str, file_type: str) -> MetadataResult:
    """
    Extract metadata based on media type ('image' or 'video').
    """
    if file_type == "video":
        return extract_video_metadata(file_path)
    return extract_image_metadata(file_path)
