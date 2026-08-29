"""
Tests for Video Compiler and Montage Renderer.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.compiler import (
    VideoCompiler,
    KenBurnsMotion,
    get_ken_burns_pattern,
    build_ken_burns_filter,
    build_video_normalization_filter,
)

client = TestClient(app)


def test_ken_burns_patterns():
    """Verify Ken Burns patterns rotate deterministically."""
    p0 = get_ken_burns_pattern(0)
    p1 = get_ken_burns_pattern(1)
    p2 = get_ken_burns_pattern(2)
    p3 = get_ken_burns_pattern(3)
    p4 = get_ken_burns_pattern(4)

    assert p0 == KenBurnsMotion.ZOOM_IN
    assert p1 == KenBurnsMotion.PAN_RIGHT
    assert p2 == KenBurnsMotion.ZOOM_OUT
    assert p3 == KenBurnsMotion.PAN_LEFT
    assert p4 == KenBurnsMotion.ZOOM_IN


def test_ken_burns_filter_syntax():
    """Verify FFmpeg zoompan filter string construction."""
    flt = build_ken_burns_filter(
        motion=KenBurnsMotion.ZOOM_IN,
        duration=2.5,
        fps=30,
        width=1920,
        height=1080,
    )
    assert "zoompan=" in flt
    assert "s=1920x1080" in flt
    assert "fps=30" in flt
    assert "d=75" in flt


def test_video_normalization_filter():
    """Verify video normalization crop-fill and letterbox syntax."""
    flt_crop = build_video_normalization_filter(1920, 1080, 30, mode="crop_fill")
    assert "crop=1920:1080" in flt_crop

    flt_letter = build_video_normalization_filter(1920, 1080, 30, mode="letterbox")
    assert "pad=1920:1080" in flt_letter


def test_video_compiler_render_photos(tmp_path):
    """Test full FFmpeg render of synthetic photos with Ken Burns into an MP4 montage."""
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")

    # Create 3 synthetic test images
    img1 = tmp_path / "photo1.jpg"
    img2 = tmp_path / "photo2.jpg"
    img3 = tmp_path / "photo3.jpg"

    Image.new("RGB", (640, 480), color="red").save(str(img1))
    Image.new("RGB", (800, 600), color="blue").save(str(img2))
    Image.new("RGB", (1024, 768), color="green").save(str(img3))

    storyboard = [
        {"file_path": str(img1), "duration": 1.5, "segment_type": "image", "justification": "Red start"},
        {"file_path": str(img2), "duration": 1.5, "segment_type": "image", "justification": "Blue middle"},
        {"file_path": str(img3), "duration": 1.5, "segment_type": "image", "justification": "Green end"},
    ]

    out_mp4 = tmp_path / "test_montage.mp4"
    compiler = VideoCompiler(
        output_width=640,
        output_height=360,
        fps=24,
        transition_duration=0.3,
        use_hardware_accel=False,
    )

    progress_events = []
    def on_prog(pct, stage, msg):
        progress_events.append((pct, stage))

    meta = compiler.render(storyboard, str(out_mp4), progress_callback=on_prog)

    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 1000
    assert meta["total_segments"] == 3
    assert len(progress_events) >= 3


def test_render_api_endpoint(tmp_path):
    """Test POST /api/v1/director/render and GET /api/v1/exports/{filename}."""
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")

    img = tmp_path / "test_api_photo.jpg"
    Image.new("RGB", (400, 300), color="purple").save(str(img))

    payload = {
        "storyboard": [
            {"file_path": str(img), "duration": 1.0, "segment_type": "image"}
        ],
        "job_id": "pytest_job_123",
        "aspect_ratio": "16:9",
        "fps": 24,
    }

    resp = client.post("/api/v1/director/render", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "COMPLETED"
    assert "stream_url" in data
    assert data["total_segments"] == 1

    # Verify stream/download endpoint
    filename = data["file_name"]
    stream_resp = client.get(f"/api/v1/exports/{filename}")
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"] == "video/mp4"
