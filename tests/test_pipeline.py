"""
Unit and integration tests for Milestone 8: Multithreaded Ingestion Pipeline.
"""

import time
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from app.db.manifest import ManifestDB
from app.db.qdrant import QdrantVectorDB
from app.db.models import FileStatus
from app.core.pipeline import IngestionPipeline, PipelineProgress, PipelineSummary
from app.core.embedder import EmbedderInterface


class MockEmbedder(EmbedderInterface):
    """Fast deterministic mock embedder for high-throughput pipeline testing."""
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

    def empty_cache(self) -> None:
        pass



def create_sample_image(path: Path, color=(255, 0, 0)):
    """Helper to generate a test JPEG image."""
    img = Image.new("RGB", (320, 240), color=color)
    img.save(str(path), format="JPEG")


def test_ingestion_pipeline_end_to_end(tmp_path):
    """
    Test end-to-end ingestion pipeline:
    1. Populate manifest with 5 test image records.
    2. Run IngestionPipeline with 12 workers and MockEmbedder.
    3. Assert all 5 files transition to INDEXED.
    4. Assert Qdrant vector collection has 5 points with correct payloads.
    """
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    # Create 5 test images
    for i in range(5):
        create_sample_image(media_dir / f"photo_{i}.jpg", color=(i * 40, 100, 150))

    manifest_file = tmp_path / "manifest.db"
    manifest = ManifestDB(str(manifest_file))

    # Populate manifest records
    for i in range(5):
        p = media_dir / f"photo_{i}.jpg"
        manifest.upsert_file(
            file_path=str(p),
            file_hash=f"hash_{i}",
            file_size=p.stat().st_size,
            file_type="image",
            creation_timestamp=1700000000.0 + i * 100,
        )

    qdrant = QdrantVectorDB(in_memory=True)
    qdrant.ensure_collection("media_embeddings", vector_size=768)

    mock_embedder = MockEmbedder(dim=768)
    progress_events: list[PipelineProgress] = []

    def on_progress(p: PipelineProgress):
        progress_events.append(p)

    pipeline = IngestionPipeline(
        manifest=manifest,
        qdrant=qdrant,
        embedder=mock_embedder,
        max_decode_workers=4,
        batch_size=2,
    )

    summary: PipelineSummary = pipeline.run(progress_callback=on_progress)

    assert summary.total_files == 5
    assert summary.indexed_files == 5
    assert summary.error_files == 0
    assert summary.total_vectors == 5

    # Check manifest records are updated to INDEXED
    all_records = manifest.get_all_files()
    for rec in all_records:
        assert rec.status == FileStatus.INDEXED
        assert rec.qdrant_point_ids is not None
        assert rec.model_name == "mock-siglip2"

    # Check Qdrant point count
    assert qdrant.count("media_embeddings") == 5

    # Verify search on Qdrant returns indexed files
    query_vec = np.ones(768, dtype=np.float32)
    results = qdrant.search("media_embeddings", query_vec, limit=10)
    assert len(results) == 5

    # Check progress callbacks were emitted
    assert len(progress_events) > 0
    assert progress_events[-1].stage == "COMPLETED"
    assert progress_events[-1].percentage == 100.0


def test_ingestion_pipeline_skips_already_indexed(tmp_path):
    """Verify pipeline skips files that are already INDEXED unless force_reindex=True."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    create_sample_image(media_dir / "already_done.jpg")

    manifest = ManifestDB(str(tmp_path / "manifest.db"))
    f = manifest.upsert_file(
        file_path=str(media_dir / "already_done.jpg"),
        file_hash="h1",
        file_size=1024,
        file_type="image",
    )
    manifest.update_embedding_info(f.id, ["pt-1"], model_name="mock-siglip2")

    qdrant = QdrantVectorDB(in_memory=True)
    qdrant.ensure_collection("media_embeddings", vector_size=768)

    pipeline = IngestionPipeline(manifest=manifest, qdrant=qdrant, embedder=MockEmbedder())

    # First run without force_reindex
    summary = pipeline.run(force_reindex=False)
    assert summary.total_files == 1
    assert summary.indexed_files == 1

    # Second run with force_reindex
    summary_forced = pipeline.run(force_reindex=True)
    assert summary_forced.total_files == 1
    assert summary_forced.indexed_files == 1


def test_ingestion_pipeline_handles_corrupted_files(tmp_path):
    """Verify corrupted or unreadable files are set to ERROR status without crashing the pipeline."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    # 1 valid image, 1 corrupted text file masquerading as an image
    create_sample_image(media_dir / "good.jpg")
    corrupted_file = media_dir / "bad.jpg"
    corrupted_file.write_text("This is not an image binary")

    manifest = ManifestDB(str(tmp_path / "manifest.db"))
    manifest.upsert_file(file_path=str(media_dir / "good.jpg"), file_hash="h_good", file_size=100, file_type="image")
    manifest.upsert_file(file_path=str(corrupted_file), file_hash="h_bad", file_size=100, file_type="image")

    qdrant = QdrantVectorDB(in_memory=True)
    qdrant.ensure_collection("media_embeddings", vector_size=768)

    pipeline = IngestionPipeline(manifest=manifest, qdrant=qdrant, embedder=MockEmbedder())
    summary = pipeline.run()

    assert summary.total_files == 2
    assert summary.indexed_files == 1
    assert summary.error_files == 1

    rec_good = manifest.lookup(str(media_dir / "good.jpg"))
    rec_bad = manifest.lookup(str(corrupted_file))

    assert rec_good.status == FileStatus.INDEXED
    assert rec_bad.status == FileStatus.ERROR
