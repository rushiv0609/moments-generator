"""
Unit and integration tests for Background Job Manager and SSE Streaming.
"""

import time
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.jobs import JobManager, JobStatus, JobEvent
from app.core.workspace import get_workspace_manager


def create_sample_image(path: Path, color=(0, 255, 0)):
    """Helper to generate a test JPEG image."""
    img = Image.new("RGB", (200, 200), color=color)
    img.save(str(path), format="JPEG")


def test_job_manager_indexing_lifecycle(tmp_path):
    """Test full background job lifecycle from submission to completion."""
    ws_dir = tmp_path / "TestJobWorkspace"
    corpus_dir = tmp_path / "TestCorpus"
    corpus_dir.mkdir(parents=True)

    # Add 2 images to corpus
    create_sample_image(corpus_dir / "img1.jpg")
    create_sample_image(corpus_dir / "img2.jpg")

    # Initialize workspace and scan
    workspace_mgr = get_workspace_manager()
    workspace_mgr.set_workspace(ws_dir, corpus_path=corpus_dir)
    manifest = workspace_mgr.get_manifest_db()
    manifest.upsert_file(file_path=str(corpus_dir / "img1.jpg"), file_hash="h1", file_size=100, file_type="image")
    manifest.upsert_file(file_path=str(corpus_dir / "img2.jpg"), file_hash="h2", file_size=100, file_type="image")

    job_mgr = JobManager()
    job = job_mgr.submit_indexing_job(
        workspace_dir=str(ws_dir),
        corpus_dir=str(corpus_dir),
    )

    assert job.id.startswith("job_idx_")
    assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    # Wait for completion (up to 10 seconds)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        curr = job_mgr.get_job(job.id)
        if curr and curr.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.1)

    completed_job = job_mgr.get_job(job.id)
    assert completed_job is not None
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.progress == 100.0
    assert completed_job.summary is not None
    assert completed_job.summary["indexed_files"] == 2


def test_job_api_endpoints(tmp_path):
    """Test REST API routes for job submission, querying, and cancellation."""
    client = TestClient(app)
    ws_dir = tmp_path / "APIJobWorkspace"
    corpus_dir = tmp_path / "APICorpus"
    corpus_dir.mkdir(parents=True)
    create_sample_image(corpus_dir / "pic.jpg")

    # Set workspace first
    client.post("/api/v1/workspace/set", json={"workspace_path": str(ws_dir), "corpus_path": str(corpus_dir)})
    client.post("/api/v1/scan", json={"corpus_path": str(corpus_dir), "workspace_path": str(ws_dir)})

    # Submit indexing job
    res = client.post("/api/v1/jobs/index", json={"workspace_path": str(ws_dir), "corpus_path": str(corpus_dir)})
    assert res.status_code == 200
    data = res.json()
    job_id = data["job_id"]
    assert job_id.startswith("job_idx_")

    # Query status
    res_status = client.get(f"/api/v1/jobs/{job_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["id"] == job_id
    assert status_data["status"] in ("QUEUED", "RUNNING", "COMPLETED")
