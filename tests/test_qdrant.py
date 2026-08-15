"""
Unit and integration tests for Milestone 7: Qdrant Vector DB Integration.
"""

import uuid
import numpy as np
import pytest
from app.db.qdrant import QdrantVectorDB, VectorPoint, SearchResult
from app.db.manifest import ManifestDB
from app.db.models import FileStatus, FileRecord


@pytest.fixture
def in_memory_qdrant():
    """Create an isolated in-memory Qdrant instance for unit testing."""
    return QdrantVectorDB(in_memory=True)


def test_vector_point_dataclass():
    """Verify VectorPoint handles automatic UUID generation and array casting."""
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    point = VectorPoint(
        vector=vec,
        file_path="/media/photo1.jpg",
        file_type="image",
        creation_timestamp=1700000000.0,
    )
    assert point.id is not None
    assert isinstance(point.id, str)
    assert len(point.id) == 36  # UUID length
    assert isinstance(point.vector, list)
    assert len(point.vector) == 3

    payload = point.to_payload()
    assert payload["file_path"] == "/media/photo1.jpg"
    assert payload["file_type"] == "image"
    assert payload["creation_timestamp"] == 1700000000.0
    assert payload["frame_index"] == 0


def test_ensure_and_delete_collection(in_memory_qdrant):
    """Verify collection creation, idempotent double-ensure, and deletion."""
    col_name = "test_media_collection"
    assert not in_memory_qdrant.collection_exists(col_name)

    # 1. Ensure collection creates it
    res = in_memory_qdrant.ensure_collection(col_name, vector_size=768)
    assert res is True
    assert in_memory_qdrant.collection_exists(col_name)

    info = in_memory_qdrant.get_collection_info(col_name)
    assert info is not None
    assert info["name"] == col_name
    assert info["points_count"] == 0

    # 2. Double ensure is idempotent
    res_again = in_memory_qdrant.ensure_collection(col_name, vector_size=768)
    assert res_again is True

    # 3. Delete collection
    del_res = in_memory_qdrant.delete_collection(col_name)
    assert del_res is True
    assert not in_memory_qdrant.collection_exists(col_name)
    assert in_memory_qdrant.get_collection_info(col_name) is None


def test_batch_upsert_and_count(in_memory_qdrant):
    """Verify batch upsert of photo and video points and count accuracy."""
    col_name = "test_upsert_col"
    dim = 4

    # Create 5 synthetic vector points
    points = [
        VectorPoint(
            vector=[1.0, 0.0, 0.0, 0.0],
            file_path="/media/pic1.jpg",
            file_id=1,
            file_type="image",
            creation_timestamp=100.0,
        ),
        VectorPoint(
            vector=[0.0, 1.0, 0.0, 0.0],
            file_path="/media/pic2.jpg",
            file_id=2,
            file_type="image",
            creation_timestamp=200.0,
        ),
        VectorPoint(
            vector=[0.0, 0.0, 1.0, 0.0],
            file_path="/media/vid1.mp4",
            file_id=3,
            file_type="video",
            frame_index=0,
            source_offset=0.0,
            creation_timestamp=300.0,
        ),
        VectorPoint(
            vector=[0.0, 0.0, 0.9, 0.1],
            file_path="/media/vid1.mp4",
            file_id=3,
            file_type="video",
            frame_index=1,
            source_offset=1.0,
            creation_timestamp=301.0,
        ),
    ]

    point_ids = in_memory_qdrant.upsert_points(col_name, points, batch_size=2)
    assert len(point_ids) == 4
    assert in_memory_qdrant.count(col_name) == 4


def test_cosine_similarity_search_and_ranking(in_memory_qdrant):
    """Verify Top-K cosine similarity retrieval returns best match first."""
    col_name = "test_search_col"
    dim = 4

    points = [
        VectorPoint(
            vector=[1.0, 0.0, 0.0, 0.0],  # Exact match for query
            file_path="/media/beach_sunset.jpg",
            creation_timestamp=1000.0,
        ),
        VectorPoint(
            vector=[0.7071, 0.7071, 0.0, 0.0],  # Partial match
            file_path="/media/beach_walk.jpg",
            creation_timestamp=1100.0,
        ),
        VectorPoint(
            vector=[0.0, 0.0, 1.0, 0.0],  # Orthogonal / zero similarity
            file_path="/media/city_night.jpg",
            creation_timestamp=1200.0,
        ),
    ]
    in_memory_qdrant.upsert_points(col_name, points)

    # Search with exact beach query [1.0, 0.0, 0.0, 0.0]
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = in_memory_qdrant.search(col_name, query, limit=3)

    assert len(results) == 3
    # First result should be exact match
    assert results[0].file_path == "/media/beach_sunset.jpg"
    assert pytest.approx(results[0].score, abs=1e-4) == 1.0

    # Second result should be partial match
    assert results[1].file_path == "/media/beach_walk.jpg"
    assert pytest.approx(results[1].score, abs=1e-4) == 0.7071

    # Third result should be orthogonal
    assert results[2].file_path == "/media/city_night.jpg"
    assert pytest.approx(results[2].score, abs=1e-4) == 0.0


def test_temporal_range_filtering(in_memory_qdrant):
    """Verify time-bucketed search returns only points within timestamp bounds."""
    col_name = "test_time_filter_col"

    points = [
        VectorPoint(
            vector=[1.0, 0.0, 0.0, 0.0],
            file_path="/media/jan_trip.jpg",
            creation_timestamp=100.0,
        ),
        VectorPoint(
            vector=[1.0, 0.0, 0.0, 0.0],
            file_path="/media/june_trip.jpg",
            creation_timestamp=500.0,
        ),
        VectorPoint(
            vector=[1.0, 0.0, 0.0, 0.0],
            file_path="/media/dec_trip.jpg",
            creation_timestamp=900.0,
        ),
    ]
    in_memory_qdrant.upsert_points(col_name, points)

    query = [1.0, 0.0, 0.0, 0.0]

    # Filter between t=400 and t=600 (should only return june_trip)
    results = in_memory_qdrant.search(
        col_name,
        query,
        start_timestamp=400.0,
        end_timestamp=600.0,
    )
    assert len(results) == 1
    assert results[0].file_path == "/media/june_trip.jpg"

    # Filter with only start_timestamp >= 500.0
    results_after = in_memory_qdrant.search(
        col_name,
        query,
        start_timestamp=500.0,
    )
    assert len(results_after) == 2
    paths = {r.file_path for r in results_after}
    assert paths == {"/media/june_trip.jpg", "/media/dec_trip.jpg"}


def test_file_type_and_exclusion_filter(in_memory_qdrant):
    """Verify filtering by file_type and excluding specific files."""
    col_name = "test_type_filter_col"

    points = [
        VectorPoint(vector=[1.0, 0.0, 0.0, 0.0], file_path="/media/p1.jpg", file_type="image"),
        VectorPoint(vector=[1.0, 0.0, 0.0, 0.0], file_path="/media/p2.jpg", file_type="image"),
        VectorPoint(vector=[1.0, 0.0, 0.0, 0.0], file_path="/media/v1.mp4", file_type="video"),
    ]
    in_memory_qdrant.upsert_points(col_name, points)

    query = [1.0, 0.0, 0.0, 0.0]

    # 1. Filter video only
    vid_results = in_memory_qdrant.search(col_name, query, file_type="video")
    assert len(vid_results) == 1
    assert vid_results[0].file_path == "/media/v1.mp4"

    # 2. Exclude p1.jpg
    exclude_results = in_memory_qdrant.search(
        col_name,
        query,
        file_type="image",
        must_not_file_paths=["/media/p1.jpg"],
    )
    assert len(exclude_results) == 1
    assert exclude_results[0].file_path == "/media/p2.jpg"


def test_delete_points_and_by_file_path(in_memory_qdrant):
    """Verify point deletion by UUID list and by file_path filter."""
    col_name = "test_del_col"

    p1 = VectorPoint(vector=[1.0, 0.0, 0.0, 0.0], file_path="/media/photo.jpg")
    p2 = VectorPoint(vector=[0.0, 1.0, 0.0, 0.0], file_path="/media/vid.mp4", frame_index=0)
    p3 = VectorPoint(vector=[0.0, 0.0, 1.0, 0.0], file_path="/media/vid.mp4", frame_index=1)

    point_ids = in_memory_qdrant.upsert_points(col_name, [p1, p2, p3])
    assert in_memory_qdrant.count(col_name) == 3

    # 1. Delete p1 by ID
    in_memory_qdrant.delete_points(col_name, [p1.id])
    assert in_memory_qdrant.count(col_name) == 2

    # 2. Delete all frames of vid.mp4 by file_path
    in_memory_qdrant.delete_by_file_path(col_name, "/media/vid.mp4")
    assert in_memory_qdrant.count(col_name) == 0


def test_qdrant_manifest_integration_workflow(tmp_path):
    """
    Verify complete integration between ManifestDB and QdrantVectorDB:
    File inserted -> Vector embedded and upserted -> Manifest updated to INDEXED with point IDs.
    """
    manifest = ManifestDB.open_or_create(tmp_path / "corpus", data_dir=tmp_path / "data")
    qdrant = QdrantVectorDB(in_memory=True)
    col_name = "integration_test_collection"

    # Step 1: Scanner inserts discovered file into Manifest
    saved_file = manifest.upsert_file(
        file_path=str(tmp_path / "corpus/sunset.jpg"),
        file_hash="hash123",
        file_size=1024,
        file_type="image",
        status=FileStatus.PENDING,
    )
    assert saved_file.status == FileStatus.PENDING

    # Step 2: Extraction & Embedding pipeline runs
    manifest.update_status(saved_file.id, FileStatus.SCANNED)
    manifest.update_status(saved_file.id, FileStatus.EXTRACTED)
    manifest.update_status(saved_file.id, FileStatus.EMBEDDED)

    # Step 3: Indexer upserts into Qdrant
    vector = np.random.randn(768).astype(np.float32)
    vector /= np.linalg.norm(vector)  # L2 normalize

    point = VectorPoint(
        vector=vector,
        file_path=saved_file.file_path,
        file_id=saved_file.id,
        file_type=saved_file.file_type,
        creation_timestamp=1700000000.0,
    )
    point_ids = qdrant.upsert_points(col_name, [point])
    assert len(point_ids) == 1

    # Step 4: Manifest updated to INDEXED with stored point IDs
    manifest.update_embedding_info(
        saved_file.id,
        qdrant_point_ids=point_ids,
        model_name="google/siglip2-base-patch16-224",
    )

    indexed_file = manifest.lookup(saved_file.file_path)
    assert indexed_file.status == FileStatus.INDEXED
    assert indexed_file.qdrant_point_ids == point_ids
    assert indexed_file.model_name == "google/siglip2-base-patch16-224"

    # Step 5: Querying Qdrant returns the exact file
    search_res = qdrant.search(col_name, vector, limit=1)
    assert len(search_res) == 1
    assert search_res[0].file_id == saved_file.id
    assert search_res[0].file_path == saved_file.file_path
    assert pytest.approx(search_res[0].score, abs=1e-4) == 1.0
