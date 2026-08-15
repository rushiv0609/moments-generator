"""
Unit and integration tests for Project Workspace Bundle Architecture.
"""

import os
import json
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.workspace import WorkspaceManager, WorkspaceInfo
from app.db.models import FileRecord, FileStatus
from app.db.qdrant import VectorPoint


def test_workspace_manager_directory_structure(tmp_path):
    """Verify workspace initialization creates isolated .moments and exports directories."""
    project_dir = tmp_path / "SummerVacation2025"
    corpus_dir = tmp_path / "MyPhotos"
    corpus_dir.mkdir()

    mgr = WorkspaceManager()
    info = mgr.set_workspace(project_dir, corpus_path=corpus_dir)

    assert info.workspace_dir == str(project_dir.resolve())
    assert info.corpus_dir == str(corpus_dir.resolve())
    assert Path(info.manifest_db_path).exists()
    assert Path(info.qdrant_storage_path).exists()
    assert Path(info.exports_dir).exists()
    assert Path(info.cache_dir).exists()

    # Verify workspace_meta.json
    meta_file = project_dir / "workspace_meta.json"
    assert meta_file.exists()
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert meta["workspace_dir"] == str(project_dir.resolve())
    assert meta["corpus_dir"] == str(corpus_dir.resolve())


def test_workspace_isolation_between_two_projects(tmp_path):
    """
    Verify complete database and vector isolation between two separate projects.
    Project A vectors and manifest records must NEVER bleed into Project B.
    """
    proj_a = tmp_path / "Album_Europe"
    proj_b = tmp_path / "Album_Japan"

    mgr_a = WorkspaceManager(proj_a)
    mgr_b = WorkspaceManager(proj_b)

    # Ingest file and vector into Project A
    manifest_a = mgr_a.get_manifest_db()
    qdrant_a = mgr_a.get_qdrant_db()

    f_a = manifest_a.upsert_file(
        file_path=str(proj_a / "paris.jpg"),
        file_hash="hash_a",
        file_size=2048,
        file_type="image",
    )
    vec_a = np.zeros(768, dtype=np.float32)
    vec_a[0] = 1.0
    point_a = VectorPoint(
        vector=vec_a,
        file_path=f_a.file_path,
        file_id=f_a.id,
        file_type="image",
    )
    point_ids_a = qdrant_a.upsert_points("media_embeddings", [point_a])
    manifest_a.update_embedding_info(f_a.id, point_ids_a, model_name="siglip2")

    # Ingest different file and vector into Project B
    manifest_b = mgr_b.get_manifest_db()
    qdrant_b = mgr_b.get_qdrant_db()

    f_b = manifest_b.upsert_file(
        file_path=str(proj_b / "tokyo.jpg"),
        file_hash="hash_b",
        file_size=4096,
        file_type="image",
    )
    vec_b = np.zeros(768, dtype=np.float32)
    vec_b[1] = 1.0
    point_b = VectorPoint(
        vector=vec_b,
        file_path=f_b.file_path,
        file_id=f_b.id,
        file_type="image",
    )
    point_ids_b = qdrant_b.upsert_points("media_embeddings", [point_b])
    manifest_b.update_embedding_info(f_b.id, point_ids_b, model_name="siglip2")

    # Assert Project A only has paris.jpg and 1 vector
    assert manifest_a.lookup(str(proj_a / "paris.jpg")) is not None
    assert manifest_a.lookup(str(proj_b / "tokyo.jpg")) is None
    assert qdrant_a.count("media_embeddings") == 1
    res_a = qdrant_a.search("media_embeddings", vec_a, limit=10)
    assert len(res_a) == 1
    assert res_a[0].file_path == str(proj_a / "paris.jpg")

    # Assert Project B only has tokyo.jpg and 1 vector
    assert manifest_b.lookup(str(proj_b / "tokyo.jpg")) is not None
    assert manifest_b.lookup(str(proj_a / "paris.jpg")) is None
    assert qdrant_b.count("media_embeddings") == 1
    res_b = qdrant_b.search("media_embeddings", vec_b, limit=10)
    assert len(res_b) == 1
    assert res_b[0].file_path == str(proj_b / "tokyo.jpg")


def test_workspace_api_endpoints(tmp_path):
    """Test REST API routes for setting and querying active project workspace."""
    client = TestClient(app)
    project_dir = tmp_path / "APITestProject"

    # 1. Set active workspace
    res = client.post(
        "/api/v1/workspace/set",
        json={"workspace_path": str(project_dir), "corpus_path": str(tmp_path / "Media")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["workspace_dir"] == str(project_dir.resolve())
    assert Path(data["manifest_db_path"]).exists()

    # 2. Get current active workspace
    res_curr = client.get("/api/v1/workspace/current")
    assert res_curr.status_code == 200
    curr_data = res_curr.json()
    assert curr_data["workspace_dir"] == str(project_dir.resolve())

    # 3. Health check reflects active workspace
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["active_workspace"] is not None
    assert health_data["active_workspace"]["workspace_dir"] == str(project_dir.resolve())
