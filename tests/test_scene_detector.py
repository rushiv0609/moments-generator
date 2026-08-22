"""
Test suite for Video Scene Boundary Detection and Dual-Granularity Ingestion.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from app.core.scene_detector import (
    SceneBoundary,
    detect_video_scenes,
    get_video_duration_cv2,
)
from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB, VectorPoint
from app.core.pipeline import IngestionPipeline
from app.core.embedder import EmbedderInterface


class MockPipelineEmbedder(EmbedderInterface):
    """Deterministic mock embedder."""
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.model_name = "mock-siglip2"

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.ones(self.dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_images(self, batch_pixels: list) -> np.ndarray:
        batch_size = len(batch_pixels)
        vecs = np.ones((batch_size, self.dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def model_info(self) -> dict:
        return {"model_name": self.model_name, "backend": "mock", "precision": "fp32"}


def create_synthetic_video_with_cuts(file_path: Path, fps: int = 30, scene_dur_sec: int = 3) -> str:
    """
    Creates a video with 2 distinct colored scenes to trigger scene detection.
    Scene 1: Solid Red (0 to 3s)
    Scene 2: Solid Green (3 to 6s)
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, float(fps), (width, height))

    total_frames = fps * scene_dur_sec * 2
    for i in range(total_frames):
        if i < fps * scene_dur_sec:
            # Red frame (BGR: 0, 0, 255)
            frame = np.full((height, width, 3), (0, 0, 255), dtype=np.uint8)
        else:
            # Green frame (BGR: 0, 255, 0)
            frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
        out.write(frame)

    out.release()
    return str(file_path)


def test_scene_boundary_dataclass():
    """Verify SceneBoundary calculation and midpoint helper."""
    sb = SceneBoundary(scene_id=1, start_sec=2.5, end_sec=7.5, duration_sec=5.0)
    assert sb.scene_id == 1
    assert sb.duration_sec == 5.0
    assert sb.midpoint_sec == 5.0


def test_detect_video_scenes_synthetic(tmp_path):
    """Verify AdaptiveDetector correctly splits video with scene cuts."""
    vid_path = tmp_path / "two_scenes.mp4"
    create_synthetic_video_with_cuts(vid_path, fps=30, scene_dur_sec=3)

    assert vid_path.exists()
    dur = get_video_duration_cv2(str(vid_path))
    assert dur >= 5.5

    scenes = detect_video_scenes(str(vid_path), adaptive_threshold=2.0, min_scene_length_sec=1.5)
    assert len(scenes) >= 1
    assert scenes[0].start_sec == 0.0
    assert scenes[-1].end_sec > 0.0


def test_detect_video_scenes_nonexistent_file():
    """Verify error on nonexistent video path."""
    with pytest.raises(FileNotFoundError):
        detect_video_scenes("/nonexistent/video.mp4")


def test_qdrant_dual_granularity_search(tmp_path):
    """
    Verify Qdrant filtering by granularity ('frame' vs 'scene')
    and is_scene_representative flag.
    """
    qdrant = QdrantVectorDB(in_memory=True)
    collection = "test_dual_granularity"
    qdrant.ensure_collection(collection, vector_size=768)

    dim = 768
    vec = np.ones(dim, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    # 1. Add 3 frame-level points
    frame_points = [
        VectorPoint(
            vector=vec,
            file_path="/videos/trip.mp4",
            file_type="video",
            frame_index=i,
            source_offset=float(i),
            granularity="frame",
            scene_id=0,
            is_scene_representative=False,
        )
        for i in range(3)
    ]
    qdrant.upsert_points(collection, frame_points)

    # 2. Add 1 scene-representative point
    scene_point = VectorPoint(
        vector=vec,
        file_path="/videos/trip.mp4",
        file_type="video",
        frame_index=-1,
        source_offset=0.0,
        granularity="scene",
        scene_id=0,
        scene_start=0.0,
        scene_end=3.0,
        scene_frame_count=3,
        is_scene_representative=True,
    )
    qdrant.upsert_points(collection, [scene_point])

    # Assert total count is 4
    assert qdrant.count(collection) == 4

    # Search with granularity="scene" -> must return exactly 1 result
    scene_results = qdrant.search(collection, query_vector=vec, limit=10, granularity="scene")
    assert len(scene_results) == 1
    assert scene_results[0].granularity == "scene"
    assert scene_results[0].is_scene_representative is True

    # Search with granularity="frame" -> must return exactly 3 results
    frame_results = qdrant.search(collection, query_vector=vec, limit=10, granularity="frame")
    assert len(frame_results) == 3
    for fr in frame_results:
        assert fr.granularity == "frame"
        assert fr.is_scene_representative is False


def test_pipeline_dual_granularity_end_to_end(tmp_path):
    """
    Test end-to-end ingestion pipeline with dual-granularity Qdrant indexing:
    - 1 image (photo) -> 1 frame vector
    - 1 video (6s synthetic) -> ~6 frame vectors + at least 1 scene summary vector
    """
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    # 1. Create a test photo
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    img_path = media_dir / "photo.jpg"
    img.save(img_path)

    # 2. Create a test video (6s with cuts)
    vid_path = media_dir / "clip.mp4"
    create_synthetic_video_with_cuts(vid_path, fps=30, scene_dur_sec=3)

    manifest_file = tmp_path / "manifest.db"
    manifest = ManifestDB(str(manifest_file))

    # Populate manifest records
    manifest.upsert_file(
        file_path=str(img_path),
        file_hash="hash_photo",
        file_size=img_path.stat().st_size,
        file_type="image",
        creation_timestamp=1700000000.0,
    )
    manifest.upsert_file(
        file_path=str(vid_path),
        file_hash="hash_video",
        file_size=vid_path.stat().st_size,
        file_type="video",
        creation_timestamp=1700000100.0,
        duration_seconds=6.0,
    )

    qdrant = QdrantVectorDB(in_memory=True)
    qdrant.ensure_collection("media_embeddings", vector_size=768)

    mock_embedder = MockPipelineEmbedder(dim=768)
    pipeline = IngestionPipeline(
        manifest=manifest,
        qdrant=qdrant,
        embedder=mock_embedder,
        max_decode_workers=2,
        batch_size=4,
    )

    summary = pipeline.run()
    assert summary.total_files == 2
    assert summary.indexed_files == 2
    assert summary.error_files == 0

    # Query Qdrant for scene representatives vs frames
    dummy_query = np.ones(768, dtype=np.float32)
    dummy_query /= np.linalg.norm(dummy_query)

    scene_res = qdrant.search("media_embeddings", dummy_query, limit=20, granularity="scene")
    assert len(scene_res) >= 1
    for sr in scene_res:
        assert sr.granularity == "scene"
        assert sr.is_scene_representative is True
        assert sr.file_type == "video"

    frame_res = qdrant.search("media_embeddings", dummy_query, limit=50, granularity="frame")
    assert len(frame_res) >= 2  # 1 photo + multiple video frames
    file_types = {fr.file_type for fr in frame_res}
    assert "image" in file_types
    assert "video" in file_types
