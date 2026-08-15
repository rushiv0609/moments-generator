"""
Unit and integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_redirect():
    """Verify root / redirects to /ui/."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_health_endpoint():
    """Verify /api/v1/health returns structured status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "qdrant" in data
    assert "ffmpeg" in data
    assert "system_memory" in data
    assert data["model"]["name"] == "google/siglip2-base-patch16-224"


def test_debug_config_endpoint():
    """Verify /api/v1/debug/config returns settings."""
    response = client.get("/api/v1/debug/config")
    assert response.status_code == 200
    data = response.json()
    assert data["EXTRACT_WORKERS"] == 12
    assert data["EMBED_BATCH_SIZE"] == 64
    assert data["MODEL_PRECISION"] == "fp16"


def test_debug_data_endpoint():
    """Verify /api/v1/debug/data returns data folder listing."""
    response = client.get("/api/v1/debug/data")
    assert response.status_code == 200
    data = response.json()
    assert "data_dir" in data
    assert "items" in data
    assert isinstance(data["items"], list)


def test_job_stubs_return_501():
    """Verify future milestone stubs return 501 Not Implemented."""
    res_gen = client.post("/api/v1/jobs/generate", json={
        "corpus_path": "/fake/path",
        "prompt": "sunset on the beach",
        "target_duration_seconds": 60,
    })
    assert res_gen.status_code == 501

    res_events = client.get("/api/v1/jobs/test-id/events")
    assert res_events.status_code == 501

    res_dl = client.get("/api/v1/jobs/test-id/download")
    assert res_dl.status_code == 501
