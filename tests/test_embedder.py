"""
Unit and integration tests for SigLIP 2 Embedding Engine (Milestone 6).
"""

import numpy as np
import pytest

from app.core.embedder import create_embedder, EmbedderInterface, MLXEmbedder, PyTorchMPSEmbedder
from app.config import Settings


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder instance to avoid reloading weights repeatedly."""
    settings = Settings(MODEL_NAME="google/siglip2-base-patch16-224", MODEL_BACKEND="auto")
    return create_embedder(settings)


def test_embedder_factory_creation(embedder):
    """Verify factory returns an instance implementing EmbedderInterface."""
    assert isinstance(embedder, EmbedderInterface)
    info = embedder.model_info()
    assert "name" in info
    assert "backend" in info
    assert info["embedding_dim"] == 768


def test_embed_single_image(embedder):
    """Verify single 224x224x3 image produces 768-dim L2-normalized vector."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emb = embedder.embed_images(dummy_img)

    assert emb.shape == (1, 768)
    assert emb.dtype == np.float32

    # Verify L2 normalization: norm == 1.0
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-4


def test_embed_image_batch(embedder):
    """Verify batched embedding produces (N, 768) matrix."""
    batch = [
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.ones((224, 224, 3), dtype=np.uint8) * 128,
        np.ones((224, 224, 3), dtype=np.uint8) * 255,
    ]
    embs = embedder.embed_images(batch)

    assert embs.shape == (3, 768)
    for i in range(3):
        norm = np.linalg.norm(embs[i])
        assert abs(norm - 1.0) < 1e-4


def test_embed_text(embedder):
    """Verify text embedding produces (768,) L2-normalized vector."""
    query = "friends having dinner outdoors on a summer evening"
    text_emb = embedder.embed_text(query)

    assert text_emb.shape == (768,)
    assert text_emb.dtype == np.float32

    norm = np.linalg.norm(text_emb)
    assert abs(norm - 1.0) < 1e-4


def test_semantic_differentiation(embedder):
    """Verify distinct semantic prompts produce meaningfully different embedding vectors."""
    v1 = embedder.embed_text("a snowy mountain summit in winter")
    v2 = embedder.embed_text("a tropical beach with palm trees and ocean")

    similarity = float(np.dot(v1, v2))
    # Distinct concepts should have cosine similarity substantially lower than identical concepts
    assert similarity < 0.85
