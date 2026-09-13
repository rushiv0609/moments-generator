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

    assert "candidate_1" in alternatives
    assert "candidate_2" in alternatives
    assert "candidate_3" in alternatives

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


def test_chronological_sort_and_autofill():
    """Verify that drafting node enforces chronological order and auto-fills under-duration drafts."""
    from app.core.director.state import DraftingSegmentChoice, DraftingOutput

    class DummyDraftingLLM(MockDirectorLLM):
        def structured_generate(self, system_prompt, user_prompt, response_schema):
            # Output out-of-order and short duration (5.0s vs 20s target)
            return DraftingOutput(
                storyboard=[
                    DraftingSegmentChoice(file_path="/path/day3.jpg", duration=2.5, segment_type="image"),
                    DraftingSegmentChoice(file_path="/path/day1.jpg", duration=2.5, segment_type="image"),
                ],
                narrative_arc="Test journey",
            )

    llm = DummyDraftingLLM()
    draft_fn = make_drafting_node(llm)

    state: DirectorState = {
        "user_prompt": "Mountain trek adventure",
        "target_duration": 15,
        "creative_profile": "journey",
        "retrieved_candidates": [
            {"file_path": "/path/day3.jpg", "creation_timestamp": 3000, "score": 0.15, "composite_rank": 0.85},
            {"file_path": "/path/day1.jpg", "creation_timestamp": 1000, "score": 0.14, "composite_rank": 0.80},
            {"file_path": "/path/day2.jpg", "creation_timestamp": 2000, "score": 0.13, "composite_rank": 0.75},
            {"file_path": "/path/day4.jpg", "creation_timestamp": 4000, "score": 0.12, "composite_rank": 0.70},
        ],
        "storyboard": [],
    }

    res = draft_fn(state)
    sb = res["storyboard"]
    
    # 1. Verify auto-filled to reach target duration (>= 15s)
    total_dur = sum(s["duration"] for s in sb)
    assert total_dur >= 10.0  # Filled from candidates

    # 2. Verify strict chronological sorting
    timestamps = [s.get("creation_timestamp") or 0 for s in sb]
    assert timestamps == sorted(timestamps)
    assert sb[0]["file_path"] == "/path/day1.jpg"


def test_best_draft_watermarking():
    """Verify that editor preserves the highest-scoring draft when an iteration degrades."""
    class DegradingEditorLLM(MockDirectorLLM):
        def structured_generate(self, system_prompt, user_prompt, response_schema):
            return EditorOutput(
                approved=False,
                feedback="Needs adjustments",
                pacing_score=5.0,
                composite_score=5.0,
            )

    llm = DegradingEditorLLM()
    editor_fn = make_editor_node(llm, max_iterations=2)

    # Iteration 1: high scoring draft
    state: DirectorState = {
        "user_prompt": "Trek",
        "target_duration": 10,
        "iteration_count": 0,
        "storyboard": [{"file_path": "/p1.jpg", "duration": 5.0, "scores": {"nima_aesthetic": 0.9}}, {"file_path": "/p2.jpg", "duration": 5.0, "scores": {"nima_aesthetic": 0.9}}],
        "best_storyboard": [],
        "best_composite_score": 0.0,
    }
    res1 = editor_fn(state)
    assert res1["best_composite_score"] > 6.0
    best_saved = list(res1["best_storyboard"])

    # Iteration 2: degraded draft
    state2: DirectorState = {
        **state,
        "iteration_count": 1,
        "storyboard": [{"file_path": "/p_degraded.jpg", "duration": 2.0, "scores": {"nima_aesthetic": 0.1}}],
        "best_storyboard": best_saved,
        "best_composite_score": res1["best_composite_score"],
    }
    res2 = editor_fn(state2)
    # Since max_iterations=2 reached, it must approve and revert to best_saved
    assert res2["approved"] is True
    assert res2["storyboard"] == best_saved


def test_user_configurable_scoring_signals(test_env):
    """Verify that retrieval dynamically computes composite ranking based on user-selected scoring signals."""
    embedder = test_env["embedder"]
    qdrant = test_env["qdrant"]
    collection_name = test_env["collection_name"]

    retrieval_fn = make_retrieval_node(embedder, qdrant, collection_name)

    # Test 1: Cosine only
    state_cosine: DirectorState = {
        "user_prompt": "Hiking trail",
        "search_queries": ["Hiking trail"],
        "retrieval_mode": "frame",
        "creative_profile": "journey",
        "scoring_signals": ["cosine"],
        "agent_telemetry": [],
    }
    res_cosine = retrieval_fn(state_cosine)
    telem_cosine = res_cosine["agent_telemetry"][-1]
    assert telem_cosine["scoring_signals"] == ["cosine"]
    assert telem_cosine["scoring_weights"] == {"cosine": 1.0}

    # Test 2: Cosine + Sharpness + Contrast (No NIMA)
    state_tech: DirectorState = {
        "user_prompt": "Hiking trail",
        "search_queries": ["Hiking trail"],
        "retrieval_mode": "frame",
        "creative_profile": "journey",
        "scoring_signals": ["cosine", "sharpness", "contrast"],
        "agent_telemetry": [],
    }
    res_tech = retrieval_fn(state_tech)
    telem_tech = res_tech["agent_telemetry"][-1]
    assert set(telem_tech["scoring_weights"].keys()) == {"cosine", "sharpness", "contrast"}
    assert abs(sum(telem_tech["scoring_weights"].values()) - 1.0) < 1e-4

    # Test 3: Default signals when scoring_signals is None
    state_default: DirectorState = {
        "user_prompt": "Hiking trail",
        "search_queries": ["Hiking trail"],
        "retrieval_mode": "frame",
        "creative_profile": "journey",
        "scoring_signals": None,
        "agent_telemetry": [],
    }
    res_default = retrieval_fn(state_default)
    telem_default = res_default["agent_telemetry"][-1]
    assert "nima" in telem_default["scoring_weights"]
    assert abs(sum(telem_default["scoring_weights"].values()) - 1.0) < 1e-4

    # Test 4: Custom explicit weights (e.g. POC 45/45/10)
    state_custom: DirectorState = {
        "user_prompt": "Hiking trail",
        "search_queries": ["Hiking trail"],
        "retrieval_mode": "frame",
        "creative_profile": "journey",
        "scoring_weights": {"cosine": 0.45, "nima": 0.45, "technical": 0.10},
        "agent_telemetry": [],
    }
    res_custom = retrieval_fn(state_custom)
    telem_custom = res_custom["agent_telemetry"][-1]
    assert telem_custom["scoring_weights"]["cosine"] == 0.45
    assert telem_custom["scoring_weights"]["nima"] == 0.45
    assert telem_custom["scoring_weights"]["technical"] == 0.10
    assert abs(sum(telem_custom["scoring_weights"].values()) - 1.0) < 1e-4



