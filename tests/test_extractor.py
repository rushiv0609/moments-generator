"""
Unit and integration tests for Frame Extraction & Image Decoding Engine (Milestone 5).
"""

from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pytest

from app.core.extractor import (
    decode_image,
    decode_image_pillow,
    decode_image_imageio,
    extract_video_frames,
    pad_to_square,
    FrameData,
)


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample non-square JPEG image (400x300)."""
    img_path = tmp_path / "sample_photo.jpg"
    img = Image.new("RGB", (400, 300), color=(120, 180, 240))
    img.save(img_path, format="JPEG")
    return str(img_path)


@pytest.fixture
def sample_video(tmp_path):
    """Create a synthetic 2-second 30fps MP4 video."""
    video_path = tmp_path / "sample_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))

    # Write 60 frames (2 seconds)
    for i in range(60):
        frame = np.full((240, 320, 3), fill_value=i * 4, dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(video_path)


def test_pad_to_square():
    """Verify zero-padding preserves composition without cropping."""
    rect_arr = np.ones((168, 224, 3), dtype=np.uint8) * 200
    padded = pad_to_square(rect_arr, target_size=224)

    assert padded.shape == (224, 224, 3)
    assert padded.dtype == np.uint8
    # Center should contain the original pixels
    assert padded[112, 112, 0] == 200
    # Edges should contain zero-padding (black bars)
    assert padded[0, 0, 0] == 0


def test_decode_image(sample_image):
    """Verify decoding an image produces valid 224x224x3 FrameData."""
    frame = decode_image(sample_image, target_size=224)

    assert isinstance(frame, FrameData)
    assert frame.pixels.shape == (224, 224, 3)
    assert frame.pixels.dtype == np.uint8
    assert frame.frame_index == 0
    assert frame.source_offset == 0.0
    assert frame.file_type == "image"
    assert frame.file_path == str(Path(sample_image).resolve())


def test_decode_image_pillow_fallback(sample_image):
    """Verify Pillow fallback decoder produces identical dimensions."""
    pixels = decode_image_pillow(sample_image, target_size=224)
    assert pixels.shape == (224, 224, 3)
    assert pixels.dtype == np.uint8


def test_extract_video_frames(sample_video):
    """Verify video frame extraction samples at 1 FPS and produces 224x224 RGB frames."""
    frames = list(extract_video_frames(sample_video, target_size=224, sampling_fps=1.0))

    # For a 2-second video sampled at 1 FPS, expect 2 frames (at t=0.0s and t=1.0s)
    assert len(frames) == 2

    assert frames[0].frame_index == 0
    assert frames[0].source_offset == 0.0
    assert frames[0].pixels.shape == (224, 224, 3)
    assert frames[0].file_type == "video"
    assert frames[0].file_path == str(Path(sample_video).resolve())

    assert frames[1].frame_index == 1
    assert abs(frames[1].source_offset - 1.0) < 0.1
    assert frames[1].pixels.shape == (224, 224, 3)


def test_extract_video_frames_invalid_file(tmp_path):
    """Verify invalid video file raises ValueError gracefully."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"invalid data")

    with pytest.raises(ValueError, match="Unable to open video"):
        list(extract_video_frames(str(corrupt_file)))
