"""
Unit tests for Metadata Extraction Engine (Milestone 4).
"""

import time
import datetime
from pathlib import Path
from PIL import Image
import pytest

from app.core.metadata import (
    extract_metadata,
    extract_image_metadata,
    extract_video_metadata,
    parse_exif_date_str,
)


def test_parse_exif_date_str():
    """Verify various EXIF date string formats parse correctly."""
    ts1 = parse_exif_date_str("2025:08:15 14:30:00")
    assert ts1 is not None
    dt1 = datetime.datetime.fromtimestamp(ts1)
    assert dt1.year == 2025
    assert dt1.month == 8
    assert dt1.day == 15
    assert dt1.hour == 14
    assert dt1.minute == 30

    ts2 = parse_exif_date_str("2025-08-15T14:30:00Z")
    assert ts2 is not None

    assert parse_exif_date_str("invalid_date") is None
    assert parse_exif_date_str("") is None
    assert parse_exif_date_str(None) is None


def test_image_metadata_fallback_to_filesystem(tmp_path):
    """Test that an image without EXIF falls back gracefully to filesystem creation time."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (320, 240), color="blue")
    img.save(img_path)

    res = extract_metadata(str(img_path), "image")
    assert res.width == 320
    assert res.height == 240
    assert res.timestamp_source == "filesystem"
    assert res.creation_timestamp is not None
    assert res.duration_seconds is None


def test_video_metadata_nonexistent_file(tmp_path):
    """Test video metadata extraction handles non-existent or invalid video gracefully."""
    fake_video = tmp_path / "missing.mp4"
    fake_video.write_bytes(b"dummy header")

    res = extract_metadata(str(fake_video), "video")
    assert res.creation_timestamp is not None
    assert res.timestamp_source == "filesystem"
