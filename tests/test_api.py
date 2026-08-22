"""
Unit and integration tests for API endpoints.
"""

from pathlib import Path
from PIL import Image
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


def test_scan_api_endpoint(tmp_path):
    """Verify POST /api/v1/scan triggers directory scan and returns summary."""
    corpus = tmp_path / "api_corpus"
    corpus.mkdir()
    img = Image.new("RGB", (100, 100), color="yellow")
    img.save(corpus / "sample.jpg")

    response = client.post("/api/v1/scan", json={"corpus_path": str(corpus)})
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] == 1
    assert data["images_count"] == 1
    assert data["videos_count"] == 0
    assert len(data["files"]) == 1
    assert data["files"][0]["file_type"] == "image"


def test_job_stubs_return_501():
    """Verify future milestone stubs return 501 Not Implemented."""
    res_gen = client.post("/api/v1/jobs/generate", json={
        "corpus_path": "/fake/path",
        "prompt": "sunset on the beach",
        "target_duration_seconds": 60,
    })
    assert res_gen.status_code == 501

    res_events = client.get("/api/v1/jobs/nonexistent-id/events")
    assert res_events.status_code == 404

    res_dl = client.get("/api/v1/jobs/test-id/download")
    assert res_dl.status_code == 501


def test_debug_embed_endpoint(tmp_path):
    """Verify POST /api/v1/debug/embed computes text and image embeddings."""
    # 1. Test Text-only embedding
    res_text = client.post("/api/v1/debug/embed", data={"text": "a snowy mountain peak"})
    assert res_text.status_code == 200
    data_text = res_text.json()
    assert "text_embedding" in data_text
    assert len(data_text["text_embedding"]) == 768

    # 2. Test Image-only embedding
    img_path = tmp_path / "test_embed.jpg"
    img = Image.new("RGB", (224, 224), color="blue")
    img.save(img_path)

    with open(img_path, "rb") as f:
        res_img = client.post("/api/v1/debug/embed", files={"file": ("test_embed.jpg", f, "image/jpeg")})
    assert res_img.status_code == 200
    data_img = res_img.json()
    assert "image_embedding" in data_img
    assert len(data_img["image_embedding"]) == 768

    # 3. Test Combined Text + Image embedding with similarity
    with open(img_path, "rb") as f:
        res_both = client.post(
            "/api/v1/debug/embed",
            data={"text": "a clear blue sky"},
            files={"file": ("test_embed.jpg", f, "image/jpeg")},
        )
    assert res_both.status_code == 200
    data_both = res_both.json()
    assert "text_embedding" in data_both
    assert "image_embedding" in data_both
    assert "similarity" in data_both
    assert isinstance(data_both["similarity"], float)


def test_media_file_serving_endpoint(tmp_path):
    """Verify GET /api/v1/media/file streams images and handles 404s."""
    img_path = tmp_path / "sample_stream.jpg"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(img_path)

    # Valid file stream
    res = client.get(f"/api/v1/media/file?path={img_path}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # Nonexistent file -> 404
    res_404 = client.get("/api/v1/media/file?path=/nonexistent/file.jpg")
    assert res_404.status_code == 404


def test_workspace_search_endpoint(tmp_path):
    """Verify GET /api/v1/workspace/search executes query and returns ranked media items."""
    ws_dir = tmp_path / "search_ws"
    corpus_dir = tmp_path / "search_corpus"
    corpus_dir.mkdir(parents=True)

    img_path = corpus_dir / "mountain.jpg"
    img = Image.new("RGB", (224, 224), color="blue")
    img.save(img_path)

    # Set workspace
    res_set = client.post("/api/v1/workspace/set", json={
        "workspace_path": str(ws_dir),
        "corpus_path": str(corpus_dir),
    })
    assert res_set.status_code == 200

    # Scan & Index
    res_idx = client.post("/api/v1/jobs/index", json={
        "workspace_path": str(ws_dir),
        "corpus_path": str(corpus_dir),
    })
    assert res_idx.status_code == 200

    import time
    time.sleep(2.0)  # Wait for background indexer

    # Search workspace media
    res_search = client.get("/api/v1/workspace/search?query=blue+mountain&top_k=5")
    assert res_search.status_code == 200
    data = res_search.json()
    assert "query" in data
    assert "results" in data
    assert data["query"] == "blue mountain"
    assert data["workspace_dir"] == str(ws_dir.resolve())


