"""
Unit and Integration Tests for Milestone 9: LangGraph Director Agent.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from app.core.director.state import (
    DirectorState,
    TimelineSegment,
    PlannerOutput,
    DraftingOutput,
    EditorOutput,
    CandidateItem,
)
from app.core.director.llm import MockDirectorLLM, OllamaDirectorLLM
from app.core.director.nodes import (
    make_planner_node,
    make_retrieval_node,
    make_drafting_node,
    make_editor_node,
    make_compiler_node,
)
from app.core.director.graph import build_director_graph, DirectorAgent
from app.core.embedder import EmbedderInterface
from app.db.qdrant import QdrantVectorDB, VectorPoint
from app.db.manifest import ManifestDB


class DummyEmbedder(EmbedderInterface):
    """Deterministic dummy embedder for testing."""

    def embed_images(self, batch_pixels):
        return np.ones((len(batch_pixels), 768), dtype=np.float32)

    def embed_text(self, text: str):
        # Deterministic 768-dim normalized vector
        vec = np.ones(768, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def model_info(self):
        return {"name": "dummy-embedder"}

    def empty_cache(self):
        pass


@pytest.fixture
def mock_llm():
    return MockDirectorLLM()


@pytest.fixture
def test_env():
    """Setup temporary in-memory Qdrant and SQLite manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        qdrant = QdrantVectorDB(in_memory=True)
        manifest = ManifestDB.for_workspace(tmp_path)
        embedder = DummyEmbedder()
        collection_name = "test_director_collection"
        qdrant.ensure_collection(collection_name, vector_size=768)

        # Seed some vector points
        vec = np.ones(768, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)

        points = [
            VectorPoint(
                vector=vec,
                file_path=str(tmp_path / "photo1.jpg"),
                file_type="image",
                granularity="frame",
                source_offset=0.0,
            ),
            VectorPoint(
                vector=vec,
                file_path=str(tmp_path / "video1.mp4"),
                file_type="video",
                granularity="scene",
                scene_id=0,
                scene_start=0.0,
                scene_end=10.0,
                is_scene_representative=True,
            ),
            VectorPoint(
                vector=vec,
                file_path=str(tmp_path / "video1.mp4"),
                file_type="video",
                granularity="frame",
                source_offset=2.0,
                scene_id=0,
            ),
        ]
        qdrant.upsert_points(collection_name, points)

        yield {
            "tmp_path": tmp_path,
            "qdrant": qdrant,
            "manifest": manifest,
            "embedder": embedder,
            "collection_name": collection_name,
        }


def test_planner_node(mock_llm):
    planner = make_planner_node(mock_llm)
    state: DirectorState = {
        "user_prompt": "Hiking in the Swiss Alps",
        "target_duration": 30,
        "retrieval_mode": "dual",
        "search_queries": [],
        "retrieved_candidates": [],
        "storyboard": [],
        "editor_feedback": [],
        "narrative_arc": "",
        "iteration_count": 0,
        "approved": False,
        "llm_model": "mock",
        "run_label": "test",
        "error": None,
    }
    result = planner(state)
    assert len(result["search_queries"]) >= 1
    assert "narrative_arc" in result


def test_retrieval_node(test_env):
    retrieval = make_retrieval_node(
        embedder=test_env["embedder"],
        qdrant=test_env["qdrant"],
        collection_name=test_env["collection_name"],
    )
    state: DirectorState = {
        "user_prompt": "Mountain trip",
        "target_duration": 30,
        "retrieval_mode": "dual",
        "search_queries": ["mountains landscape", "scenic trail"],
        "retrieved_candidates": [],
        "storyboard": [],
        "editor_feedback": [],
        "narrative_arc": "",
        "iteration_count": 0,
        "approved": False,
        "llm_model": "mock",
        "run_label": "test",
        "error": None,
    }
    result = retrieval(state)
    assert "retrieved_candidates" in result
    assert len(result["retrieved_candidates"]) > 0


def test_drafting_and_editor_nodes(mock_llm, test_env):
    drafting = make_drafting_node(mock_llm)
    editor = make_editor_node(mock_llm)

    state: DirectorState = {
        "user_prompt": "Sunset at the beach",
        "target_duration": 15,
        "retrieval_mode": "dual",
        "search_queries": ["sunset", "beach"],
        "retrieved_candidates": [
            {
                "file_path": "/path/to/img1.jpg",
                "file_type": "image",
                "score": 0.95,
                "source_offset": 0.0,
                "granularity": "frame",
            },
            {
                "file_path": "/path/to/vid1.mp4",
                "file_type": "video",
                "score": 0.90,
                "source_offset": 5.0,
                "granularity": "scene",
                "scene_id": 1,
            },
        ],
        "storyboard": [],
        "editor_feedback": [],
        "narrative_arc": "Calm relaxing mood",
        "iteration_count": 0,
        "approved": False,
        "llm_model": "mock",
        "run_label": "test",
        "error": None,
    }

    # Draft
    draft_res = drafting(state)
    assert len(draft_res["storyboard"]) > 0

    state["storyboard"] = draft_res["storyboard"]

    # Edit
    edit_res = editor(state)
    assert edit_res["iteration_count"] == 1
    assert "approved" in edit_res


def test_director_agent_end_to_end(test_env, mock_llm):
    agent = DirectorAgent(
        embedder=test_env["embedder"],
        qdrant=test_env["qdrant"],
        collection_name=test_env["collection_name"],
        llm=mock_llm,
        manifest=test_env["manifest"],
    )

    final_state = agent.run(
        prompt="Golden hour road trip",
        target_duration=30,
        job_id="test_job_1",
    )

    assert final_state["approved"] is True
    assert len(final_state["storyboard"]) > 0
    assert final_state["iteration_count"] >= 1


def test_director_multi_alternatives(test_env, mock_llm):
    agent = DirectorAgent(
        embedder=test_env["embedder"],
        qdrant=test_env["qdrant"],
        collection_name=test_env["collection_name"],
        llm=mock_llm,
        manifest=test_env["manifest"],
    )

    alternatives = agent.generate_alternatives(
        prompt="Family gathering picnic",
        target_duration=20,
    )

    assert "alt_a_scene" in alternatives
    assert "alt_b_frame" in alternatives
    assert "alt_c_dual" in alternatives
    assert "alt_d_heuristic" in alternatives

    for key, alt_state in alternatives.items():
        assert alt_state["approved"] is True
        assert len(alt_state["storyboard"]) > 0


def test_candidate_and_timeline_segment_timestamps():
    c = CandidateItem(
        file_path="/path/to/img1.jpg",
        file_type="image",
        score=0.135,
        creation_timestamp=1720000000.0,
    )
    assert c.creation_timestamp == 1720000000.0
    dumped = c.model_dump()
    assert dumped["creation_timestamp"] == 1720000000.0

    seg = TimelineSegment(
        file_path="/path/to/img1.jpg",
        duration=3.0,
        creation_timestamp=1720000000.0,
    )
    assert seg.creation_timestamp == 1720000000.0


def test_heuristic_curator_chronological_and_diversity():
    from app.core.director.heuristic import HeuristicCurator

    # Generate 10 candidates over 5 days
    base_ts = 1720000000.0
    candidates = [
        {"file_path": f"/media/day5_b.jpg", "file_type": "image", "score": 0.14, "creation_timestamp": base_ts + 86400 * 4},
        {"file_path": f"/media/day1_a.jpg", "file_type": "image", "score": 0.13, "creation_timestamp": base_ts},
        {"file_path": f"/media/day3_a.jpg", "file_type": "image", "score": 0.12, "creation_timestamp": base_ts + 86400 * 2},
        {"file_path": f"/media/day2_a.jpg", "file_type": "image", "score": 0.11, "creation_timestamp": base_ts + 86400 * 1},
        {"file_path": f"/media/day4_a.jpg", "file_type": "image", "score": 0.12, "creation_timestamp": base_ts + 86400 * 3},
        {"file_path": f"/media/day1_b.jpg", "file_type": "image", "score": 0.09, "creation_timestamp": base_ts + 3600},
    ]

    curator = HeuristicCurator()
    res = curator.curate(candidates, target_duration=15)
    storyboard = res["storyboard"]

    assert len(storyboard) > 0
    # Verify strict chronological order
    timestamps = [s["creation_timestamp"] for s in storyboard if s.get("creation_timestamp")]
    assert timestamps == sorted(timestamps)

    # Verify no duplicate file paths
    paths = [s["file_path"] for s in storyboard]
    assert len(paths) == len(set(paths))


def test_heuristic_curator_score_filtering():
    from app.core.director.heuristic import HeuristicCurator

    candidates = [
        {"file_path": "/media/high1.jpg", "file_type": "image", "score": 0.14, "creation_timestamp": 100.0},
        {"file_path": "/media/high2.jpg", "file_type": "image", "score": 0.12, "creation_timestamp": 200.0},
        {"file_path": "/media/noise1.jpg", "file_type": "image", "score": 0.05, "creation_timestamp": 150.0},
        {"file_path": "/media/noise2.jpg", "file_type": "image", "score": 0.03, "creation_timestamp": 180.0},
    ]

    curator = HeuristicCurator()
    res = curator.curate(candidates, target_duration=10)
    storyboard = res["storyboard"]

    # Low score noise items (< 0.08) should be excluded when valid items exist
    storyboard_paths = [s["file_path"] for s in storyboard]
    assert "/media/high1.jpg" in storyboard_paths
    assert "/media/noise1.jpg" not in storyboard_paths
    assert "/media/noise2.jpg" not in storyboard_paths


def test_get_director_llm_factory():
    from app.core.director.llm import (
        get_director_llm,
        GeminiDirectorLLM,
        GroqDirectorLLM,
        OllamaDirectorLLM,
        MockDirectorLLM,
    )

    # 1. Gemini
    llm_gemini = get_director_llm("gemini-3.7-flash", api_key="fake-gemini-key")
    assert isinstance(llm_gemini, GeminiDirectorLLM)
    assert llm_gemini.model_info()["backend"] == "gemini"

    # 2. Groq
    llm_groq = get_director_llm("groq:llama-3.3-70b-versatile", api_key="fake-groq-key")
    assert isinstance(llm_groq, GroqDirectorLLM)
    assert llm_groq.model_info()["backend"] == "groq"

    # 3. Mock
    llm_mock = get_director_llm("mock-director")
    assert isinstance(llm_mock, MockDirectorLLM)

    # 4. Ollama
    llm_ollama = get_director_llm("gemma4:e4b-mlx")
    assert isinstance(llm_ollama, OllamaDirectorLLM)


def test_gemini_director_llm_mocked_http(monkeypatch):
    from app.core.director.llm import GeminiDirectorLLM
    import httpx

    llm = GeminiDirectorLLM(model_name="gemini-3.7-flash", api_key="test-api-key", fallback_to_mock=False)

    fake_json_text = '{"search_queries": ["query 1", "query 2"], "mood_or_narrative": "Cinematic test", "target_duration_seconds": 25}'
    fake_resp_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": fake_json_text}]
                }
            }
        ]
    }

    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return fake_resp_data

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json, **kwargs):
            return MockResponse()

    monkeypatch.setattr(httpx, "Client", MockClient)

    res = llm.structured_generate(
        system_prompt="Test system",
        user_prompt="Test user",
        response_schema=PlannerOutput,
    )
    assert isinstance(res, PlannerOutput)
    assert len(res.search_queries) == 2
    assert llm.last_telemetry["backend"] == "gemini"
    assert llm.last_telemetry["schema"] == "PlannerOutput"


def test_groq_director_llm_mocked_http(monkeypatch):
    from app.core.director.llm import GroqDirectorLLM
    import httpx

    llm = GroqDirectorLLM(model_name="llama-3.3-70b-versatile", api_key="test-groq-key", fallback_to_mock=False)

    fake_json_text = '{"approved": true, "feedback": "Looks great!", "pacing_score": 9.5, "suggested_modifications": []}'
    fake_resp_data = {
        "choices": [
            {
                "message": {
                    "content": fake_json_text
                }
            }
        ]
    }

    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return fake_resp_data

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json, headers, **kwargs):
            return MockResponse()

    monkeypatch.setattr(httpx, "Client", MockClient)

    res = llm.structured_generate(
        system_prompt="Test system",
        user_prompt="Test user",
        response_schema=EditorOutput,
    )
    assert isinstance(res, EditorOutput)
    assert res.approved is True
    assert res.pacing_score == 9.5
    assert llm.last_telemetry["backend"] == "groq"


def test_ollama_director_llm_unload(monkeypatch):
    from app.core.director.llm import OllamaDirectorLLM
    import httpx

    llm = OllamaDirectorLLM(model_name="gemma4:e4b-mlx", fallback_to_mock=False)
    llm._available = True

    called_payloads = []

    def mock_post(url, json=None, timeout=None):
        called_payloads.append(json)
        class Resp:
            status_code = 200
        return Resp()

    monkeypatch.setattr(httpx, "post", mock_post)

    success = llm.unload()
    assert success is True
    assert len(called_payloads) == 1
    assert called_payloads[0]["model"] == "gemma4:e4b-mlx"
    assert called_payloads[0]["keep_alive"] == 0
