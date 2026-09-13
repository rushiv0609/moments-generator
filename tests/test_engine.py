"""
Unit tests for Director Engine Strategy Pattern (LangGraph vs Pydantic Engine).
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from app.core.director.engine import create_director_engine, PydanticDirectorEngine, LangGraphDirectorEngine
from app.core.director.llm import MockDirectorLLM
from app.core.embedder import EmbedderInterface
from app.db.qdrant import QdrantVectorDB, VectorPoint
from app.db.manifest import ManifestDB


class DummyEmbedder(EmbedderInterface):
    def embed_images(self, batch_pixels):
        return np.ones((len(batch_pixels), 768), dtype=np.float32)

    def embed_text(self, text: str):
        vec = np.ones(768, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def model_info(self):
        return {"name": "dummy-embedder"}

    def empty_cache(self):
        pass


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        qdrant = QdrantVectorDB(in_memory=True)
        manifest = ManifestDB.for_workspace(tmp_path)
        embedder = DummyEmbedder()
        collection = "test_engine_coll"
        qdrant.ensure_collection(collection, vector_size=768)

        vec = np.ones(768, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        qdrant.upsert_points(collection, [
            VectorPoint(vector=vec, file_path=str(tmp_path / "img1.jpg"), file_type="image"),
            VectorPoint(vector=vec, file_path=str(tmp_path / "img2.jpg"), file_type="image"),
        ])

        yield {
            "embedder": embedder,
            "qdrant": qdrant,
            "manifest": manifest,
            "collection": collection,
            "llm": MockDirectorLLM(),
        }


def test_factory_creation(env):
    lg_engine = create_director_engine("langgraph", env["embedder"], env["qdrant"], env["collection"], env["llm"], env["manifest"])
    assert isinstance(lg_engine, LangGraphDirectorEngine)

    py_engine = create_director_engine("pydantic", env["embedder"], env["qdrant"], env["collection"], env["llm"], env["manifest"])
    assert isinstance(py_engine, PydanticDirectorEngine)


def test_pydantic_engine_run(env):
    engine = PydanticDirectorEngine(
        embedder=env["embedder"],
        qdrant=env["qdrant"],
        collection_name=env["collection"],
        llm=env["llm"],
        manifest=env["manifest"],
    )

    res = engine.run(prompt="Mountain trek", target_duration=10)
    assert "storyboard" in res
    assert len(res["storyboard"]) > 0
    assert res["approved"] is True
    assert len(res["agent_telemetry"]) >= 4


def test_pydantic_engine_generate_alternatives(env):
    engine = PydanticDirectorEngine(
        embedder=env["embedder"],
        qdrant=env["qdrant"],
        collection_name=env["collection"],
        llm=env["llm"],
        manifest=env["manifest"],
    )

    alts = engine.generate_alternatives(prompt="Beach vacation", target_duration=15)
    assert "candidate_1" in alts
    assert "candidate_2" in alts
    assert "candidate_3" in alts
    assert len(alts) == 3  # No heuristic 4th alternative
    for k, v in alts.items():
        assert "storyboard" in v
        assert len(v["storyboard"]) > 0

