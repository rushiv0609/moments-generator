"""
Unit and integration tests for Corpus Scanner Engine (Milestone 4).
"""

import time
from pathlib import Path
from PIL import Image
import pytest

from app.core.scanner import scan_corpus, get_file_type_and_mime
from app.db.manifest import ManifestDB
from app.db.models import FileStatus


@pytest.fixture
def test_corpus(tmp_path):
    """Create a structured test corpus with mixed supported and unsupported files."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create images
    (corpus_dir / "subfolder").mkdir()
    img1 = Image.new("RGB", (100, 100), color="red")
    img1.save(corpus_dir / "photo1.jpg")

    img2 = Image.new("RGB", (200, 200), color="green")
    img2.save(corpus_dir / "subfolder" / "photo2.png")

    # Create dummy video file
    video1 = corpus_dir / "clip1.mp4"
    video1.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42")

    # Unsupported files
    (corpus_dir / "readme.txt").write_text("ignore this")
    (corpus_dir / ".hidden_photo.jpg").write_text("ignore hidden")

    return corpus_dir


def test_file_type_and_mime():
    """Verify format detection and MIME type mapping."""
    ftype, mime = get_file_type_and_mime(Path("photo.heic"))
    assert ftype == "image"
    assert "heic" in mime.lower() or "heif" in mime.lower()

    ftype, mime = get_file_type_and_mime(Path("video.mov"))
    assert ftype == "video"
    assert "quicktime" in mime.lower() or "mp4" in mime.lower()

    ftype, mime = get_file_type_and_mime(Path("notes.txt"))
    assert ftype is None
    assert mime is None


def test_scan_corpus_discovery_and_manifest_population(test_corpus, tmp_path):
    """Test recursive scan discovers media, populates manifest, and ignores non-media."""
    manifest = ManifestDB.open_or_create(str(test_corpus), data_dir=str(tmp_path))
    summary = scan_corpus(str(test_corpus), manifest)

    assert summary.total_found == 3  # photo1.jpg, photo2.png, clip1.mp4
    assert summary.images_count == 2
    assert summary.videos_count == 1
    assert summary.new_files == 3
    assert len(summary.files_to_process) == 3

    # Check manifest DB
    stats = manifest.get_stats()
    assert stats["total_files"] == 3


def test_scan_corpus_skip_indexed_files(test_corpus, tmp_path):
    """Test that unchanged indexed files are skipped in subsequent scans."""
    manifest = ManifestDB.open_or_create(str(test_corpus), data_dir=str(tmp_path))
    scan_corpus(str(test_corpus), manifest)

    # Mark all files as INDEXED
    for f in manifest.get_all_files():
        manifest.update_embedding_info(f.id, [f"pt-{f.id}"], "siglip2")

    # Second scan
    summary2 = scan_corpus(str(test_corpus), manifest)
    assert summary2.total_found == 3
    assert summary2.skipped_files == 3
    assert summary2.new_files == 0
    assert len(summary2.files_to_process) == 0


def test_scan_corpus_content_deduplication(test_corpus, tmp_path):
    """Test duplicate file at a new path reuses donor embeddings without re-embedding."""
    manifest = ManifestDB.open_or_create(str(test_corpus), data_dir=str(tmp_path))
    scan_corpus(str(test_corpus), manifest)

    # Mark photo1 as INDEXED
    p1 = manifest.lookup(str(test_corpus / "photo1.jpg"))
    manifest.update_embedding_info(p1.id, ["donor-point-uuid-1"], "siglip2")

    # Copy photo1 to a new location (identical content)
    copy_path = test_corpus / "photo1_copy.jpg"
    copy_path.write_bytes((test_corpus / "photo1.jpg").read_bytes())

    # Re-scan
    summary = scan_corpus(str(test_corpus), manifest)
    assert summary.dedup_reused_files == 1

    p1_copy = manifest.lookup(str(copy_path))
    assert p1_copy is not None
    assert p1_copy.status == FileStatus.INDEXED
    assert p1_copy.qdrant_point_ids == ["donor-point-uuid-1"]


def test_scan_corpus_prunes_deleted_files(test_corpus, tmp_path):
    """Test deleting a file from disk removes it from manifest on scan."""
    manifest = ManifestDB.open_or_create(str(test_corpus), data_dir=str(tmp_path))
    scan_corpus(str(test_corpus), manifest)
    assert manifest.get_stats()["total_files"] == 3

    # Delete photo1 from disk
    (test_corpus / "photo1.jpg").unlink()

    summary = scan_corpus(str(test_corpus), manifest)
    assert summary.deleted_files == 1
    assert summary.total_found == 2
    assert manifest.get_stats()["total_files"] == 2
