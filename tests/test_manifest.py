"""
Unit and integration tests for SQLite Manifest System (Milestone 3).
"""

import os
import tempfile
import threading
from pathlib import Path
import pytest

from app.db.manifest import ManifestDB, compute_fast_hash, compute_content_hash
from app.db.models import FileRecord, TimelineSegmentRecord, FileStatus


@pytest.fixture
def temp_manifest(tmp_path):
    """Fixture providing an isolated manifest instance."""
    db_file = tmp_path / "test_manifest.db"
    return ManifestDB(str(db_file))


def test_open_or_create_deterministic(tmp_path):
    """Verify open_or_create produces stable database names based on corpus path."""
    corpus = "/Users/test/media_photos"
    m1 = ManifestDB.open_or_create(corpus, data_dir=str(tmp_path))
    m2 = ManifestDB.open_or_create(corpus, data_dir=str(tmp_path))

    assert m1.db_path == m2.db_path
    assert m1.get_meta("corpus_path") == str(Path(corpus).resolve())


def test_upsert_and_lookup_file(temp_manifest):
    """Test inserting and retrieving media files."""
    rec = temp_manifest.upsert_file(
        file_path="/media/photo1.heic",
        file_hash="fast_hash_1",
        file_size=4096,
        file_type="image",
        content_hash="content_hash_1",
        creation_timestamp=1700000000.0,
        timestamp_source="exif",
    )

    assert rec.id is not None
    assert rec.file_path == "/media/photo1.heic"
    assert rec.status == FileStatus.PENDING
    assert rec.creation_timestamp == 1700000000.0

    # Lookup
    fetched = temp_manifest.lookup("/media/photo1.heic")
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.file_hash == "fast_hash_1"
    assert fetched.content_hash == "content_hash_1"


def test_status_transitions_and_error(temp_manifest):
    """Test state machine progression and error tracking."""
    rec = temp_manifest.upsert_file(
        file_path="/media/video1.mov",
        file_hash="fast_hash_v1",
        file_size=10485760,
        file_type="video",
    )

    # Transition to SCANNED
    temp_manifest.update_status(rec.id, FileStatus.SCANNED)
    scanned_rec = temp_manifest.lookup_by_id(rec.id)
    assert scanned_rec.status == FileStatus.SCANNED

    # Update embedding info -> INDEXED
    temp_manifest.update_embedding_info(
        file_id=rec.id,
        qdrant_point_ids=["pt-uuid-1", "pt-uuid-2"],
        model_name="google/siglip2-base-patch16-224",
        model_version="1.0",
    )
    indexed_rec = temp_manifest.lookup_by_id(rec.id)
    assert indexed_rec.status == FileStatus.INDEXED
    assert indexed_rec.qdrant_point_ids == ["pt-uuid-1", "pt-uuid-2"]
    assert indexed_rec.model_name == "google/siglip2-base-patch16-224"

    # Transition to ERROR
    temp_manifest.set_error(rec.id, "Corrupt MOOV atom header")
    error_rec = temp_manifest.lookup_by_id(rec.id)
    assert error_rec.status == FileStatus.ERROR
    assert error_rec.error_message == "Corrupt MOOV atom header"


def test_content_hash_dedup_lookup(temp_manifest):
    """Test finding donor files with identical content across paths."""
    temp_manifest.upsert_file(
        file_path="/media/originals/vacation.jpg",
        file_hash="fast_1",
        content_hash="identical_hash_abc",
        file_size=2048,
        file_type="image",
        status=FileStatus.INDEXED,
    )

    donor = temp_manifest.lookup_by_content_hash("identical_hash_abc")
    assert donor is not None
    assert donor.file_path == "/media/originals/vacation.jpg"

    # Non-existent hash
    assert temp_manifest.lookup_by_content_hash("unknown_hash") is None


def test_reset_embeddings_on_model_change(temp_manifest):
    """Test that reset_embeddings rolls INDEXED/EMBEDDED files back to SCANNED while keeping metadata."""
    temp_manifest.upsert_file(
        file_path="/media/p1.jpg",
        file_hash="h1",
        file_size=100,
        file_type="image",
        creation_timestamp=1700000000.0,
        status=FileStatus.INDEXED,
    )
    temp_manifest.upsert_file(
        file_path="/media/p2.jpg",
        file_hash="h2",
        file_size=200,
        file_type="image",
        creation_timestamp=1700000100.0,
        status=FileStatus.PENDING,
    )

    affected = temp_manifest.reset_embeddings()
    assert affected == 1

    p1 = temp_manifest.lookup("/media/p1.jpg")
    p2 = temp_manifest.lookup("/media/p2.jpg")

    assert p1.status == FileStatus.SCANNED
    assert p1.creation_timestamp == 1700000000.0
    assert p1.qdrant_point_ids == []
    assert p2.status == FileStatus.PENDING


def test_remove_deleted_files(temp_manifest):
    """Test pruning files that were deleted from disk."""
    temp_manifest.upsert_file(file_path="/media/keep.jpg", file_hash="h1", file_size=100, file_type="image")
    rec2 = temp_manifest.upsert_file(file_path="/media/delete.jpg", file_hash="h2", file_size=200, file_type="image")
    temp_manifest.update_embedding_info(rec2.id, ["point-123"], "siglip2")

    deleted = temp_manifest.remove_deleted_files(["/media/keep.jpg"])
    assert len(deleted) == 1
    assert deleted[0][0] == rec2.id
    assert deleted[0][1] == ["point-123"]

    assert temp_manifest.lookup("/media/delete.jpg") is None
    assert temp_manifest.lookup("/media/keep.jpg") is not None


def test_timeline_save_and_load(temp_manifest):
    """Test persisting and restoring curated timeline segments."""
    f1 = temp_manifest.upsert_file(file_path="/media/clip1.mov", file_hash="h1", file_size=500, file_type="video")
    f2 = temp_manifest.upsert_file(file_path="/media/photo1.jpg", file_hash="h2", file_size=200, file_type="image")

    job_id = "job-uuid-888"
    segments = [
        TimelineSegmentRecord(
            id=None,
            job_id=job_id,
            position=0,
            file_id=f1.id,
            segment_type="video_clip",
            start_offset=1.5,
            duration=3.0,
            similarity_score=0.85,
            time_bucket=1,
        ),
        TimelineSegmentRecord(
            id=None,
            job_id=job_id,
            position=1,
            file_id=f2.id,
            segment_type="image",
            start_offset=0.0,
            duration=3.0,
            similarity_score=0.78,
            time_bucket=2,
        ),
    ]

    temp_manifest.save_timeline(job_id, segments)
    loaded = temp_manifest.load_timeline(job_id)

    assert len(loaded) == 2
    assert loaded[0].file_id == f1.id
    assert loaded[0].start_offset == 1.5
    assert loaded[1].file_id == f2.id
    assert loaded[1].similarity_score == 0.78


def test_concurrent_wal_mode_reads_and_writes(temp_manifest):
    """Verify SQLite WAL mode handles multi-threaded operations without locking."""
    errors = []

    def writer_task(thread_id):
        try:
            for i in range(25):
                temp_manifest.upsert_file(
                    file_path=f"/media/thread_{thread_id}_file_{i}.jpg",
                    file_hash=f"hash_{thread_id}_{i}",
                    file_size=1000 + i,
                    file_type="image",
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer_task, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    stats = temp_manifest.get_stats()
    assert stats["total_files"] == 100


def test_hashing_utilities(tmp_path):
    """Test fast hash and full content hash helpers."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Local AI Moments Generator Test Payload")

    fast_h = compute_fast_hash(100, 1700000000.0)
    assert len(fast_h) == 16  # 64-bit hex

    content_h = compute_content_hash(str(test_file))
    assert len(content_h) == 16
