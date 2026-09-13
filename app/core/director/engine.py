"""
Strategy Pattern Engine Interface for Director Agent.
Provides interchangeable backends:
1. LangGraphDirectorEngine (Stable production state machine)
2. PydanticDirectorEngine (Pure Pydantic zero-dependency state machine)
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List, Optional, Callable

from app.core.director.state import DirectorState
from app.core.director.llm import DirectorLLMInterface, get_director_llm
from app.core.director.profiles import get_creative_profile
from app.core.director.nodes import (
    make_planner_node,
    make_retrieval_node,
    make_drafting_node,
    make_editor_node,
    make_compiler_node,
)
from app.core.director.graph import DirectorAgent
from app.core.embedder import EmbedderInterface
from app.db.qdrant import QdrantVectorDB
from app.db.manifest import ManifestDB

logger = logging.getLogger(__name__)


class DirectorEngineInterface(ABC):
    """Abstract interface for Director orchestration engines."""

    @abstractmethod
    def run(
        self,
        prompt: str,
        target_duration: int = 30,
        retrieval_mode: str = "dual",
        creative_profile: str = "journey",
        job_id: Optional[str] = None,
        run_label: str = "default",
        search_queries: Optional[List[str]] = None,
        narrative_arc: Optional[str] = None,
        epic_reference_vector: Optional[List[float]] = None,
        scoring_signals: Optional[List[str]] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the curation pipeline for a single prompt."""
        pass

    @abstractmethod
    def generate_alternatives(
        self,
        prompt: str,
        target_duration: int = 30,
        job_id_prefix: str = "alt",
        epic_reference_vector: Optional[List[float]] = None,
        scoring_signals: Optional[List[str]] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate multi-alternative storyboard cuts."""
        pass


class LangGraphDirectorEngine(DirectorEngineInterface):
    """Production LangGraph StateGraph engine wrapper."""

    def __init__(
        self,
        embedder: EmbedderInterface,
        qdrant: QdrantVectorDB,
        collection_name: str,
        llm: Optional[DirectorLLMInterface] = None,
        manifest: Optional[ManifestDB] = None,
        max_iterations: int = 3,
    ):
        self.agent = DirectorAgent(
            embedder=embedder,
            qdrant=qdrant,
            collection_name=collection_name,
            llm=llm,
            manifest=manifest,
            max_iterations=max_iterations,
        )

    def run(self, **kwargs) -> Dict[str, Any]:
        return self.agent.run(**kwargs)

    def generate_alternatives(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        return self.agent.generate_alternatives(**kwargs)


class PydanticDirectorEngine(DirectorEngineInterface):
    """
    Pure Python / Pydantic state machine engine.
    Executes the exact same shared node functions with zero LangGraph overhead.
    """

    def __init__(
        self,
        embedder: EmbedderInterface,
        qdrant: QdrantVectorDB,
        collection_name: str,
        llm: Optional[DirectorLLMInterface] = None,
        manifest: Optional[ManifestDB] = None,
        max_iterations: int = 3,
    ):
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.manifest = manifest
        self.max_iterations = max_iterations
        self.llm = llm or get_director_llm()

    def run(
        self,
        prompt: str,
        target_duration: int = 30,
        retrieval_mode: str = "dual",
        creative_profile: str = "journey",
        job_id: Optional[str] = None,
        run_label: str = "default",
        search_queries: Optional[List[str]] = None,
        narrative_arc: Optional[str] = None,
        epic_reference_vector: Optional[List[float]] = None,
        scoring_signals: Optional[List[str]] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        profile_config = get_creative_profile(creative_profile)

        state: DirectorState = {
            "user_prompt": prompt,
            "target_duration": target_duration,
            "retrieval_mode": retrieval_mode,
            "creative_profile": creative_profile,
            "creative_config": profile_config,
            "epic_reference_vector": epic_reference_vector,
            "scoring_signals": scoring_signals,
            "scoring_weights": scoring_weights,
            "search_queries": search_queries or [],
            "retrieved_candidates": [],
            "storyboard": [],
            "editor_feedback": [],
            "narrative_arc": narrative_arc or "",
            "iteration_count": 0,
            "approved": False,
            "best_storyboard": [],
            "best_composite_score": 0.0,
            "llm_model": self.llm.model_info().get("model_name", "unknown"),
            "run_label": run_label,
            "error": None,
            "agent_telemetry": [],
        }

        # Initialize node factories
        planner_fn = make_planner_node(self.llm, step_callback=step_callback)
        retrieval_fn = make_retrieval_node(self.embedder, self.qdrant, self.collection_name, step_callback=step_callback)
        drafting_fn = make_drafting_node(self.llm, step_callback=step_callback)
        editor_fn = make_editor_node(self.llm, max_iterations=self.max_iterations, step_callback=step_callback)
        compiler_fn = make_compiler_node(self.manifest, job_id, step_callback=step_callback)

        try:
            # 1. Planner Node
            plan_res = planner_fn(state)
            state.update(plan_res)

            # 2. Retrieval Node
            ret_res = retrieval_fn(state)
            state.update(ret_res)

            # 3. Drafting & Editor Critique Loop
            while True:
                draft_res = drafting_fn(state)
                state.update(draft_res)

                edit_res = editor_fn(state)
                state.update(edit_res)

                if state.get("approved", False) or state.get("iteration_count", 0) >= self.max_iterations:
                    break

            # 4. Compiler Node
            comp_res = compiler_fn(state)
            state.update(comp_res)

            return state
        finally:
            if hasattr(self.llm, "unload"):
                self.llm.unload()

    def generate_alternatives(
        self,
        prompt: str,
        target_duration: int = 30,
        job_id_prefix: str = "alt",
        epic_reference_vector: Optional[List[float]] = None,
        scoring_signals: Optional[List[str]] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate multi-alternative storyboard cuts using pure Pydantic engine."""
        results = {}
        try:
            planner_fn = make_planner_node(self.llm, step_callback=step_callback)
            plan_state: DirectorState = {
                "user_prompt": prompt,
                "target_duration": target_duration,
                "retrieval_mode": "dual",
                "creative_profile": "journey",
                "scoring_signals": scoring_signals,
                "scoring_weights": scoring_weights,
                "search_queries": [],
                "retrieved_candidates": [],
                "storyboard": [],
                "editor_feedback": [],
                "narrative_arc": "",
                "iteration_count": 0,
                "approved": False,
                "best_storyboard": [],
                "best_composite_score": 0.0,
                "llm_model": self.llm.model_info().get("model_name", "unknown"),
                "run_label": "shared_planner",
                "error": None,
                "agent_telemetry": [],
            }
            plan_res = planner_fn(plan_state)
            shared_queries = plan_res.get("search_queries", [prompt])
            shared_narrative = plan_res.get("narrative_arc", "")
            shared_telemetry = plan_res.get("agent_telemetry", [])

            profiles_to_run = [
                ("candidate_1", "cinematic"),
                ("candidate_2", "personal"),
                ("candidate_3", "journey"),
            ]

            for label, profile_id in profiles_to_run:
                res = self.run(
                    prompt=prompt,
                    target_duration=target_duration,
                    retrieval_mode="dual",
                    creative_profile=profile_id,
                    job_id=f"{job_id_prefix}_{label}",
                    run_label=label,
                    search_queries=shared_queries,
                    narrative_arc=shared_narrative,
                    epic_reference_vector=epic_reference_vector,
                    scoring_signals=scoring_signals,
                    scoring_weights=scoring_weights,
                    step_callback=step_callback,
                )
                if shared_telemetry and res.get("agent_telemetry"):
                    res["agent_telemetry"] = shared_telemetry + res["agent_telemetry"]
                results[label] = res

            return results
        finally:
            if hasattr(self.llm, "unload"):
                self.llm.unload()


def create_director_engine(
    engine_name: str = "langgraph",
    embedder: Optional[EmbedderInterface] = None,
    qdrant: Optional[QdrantVectorDB] = None,
    collection_name: str = "media_embeddings",
    llm: Optional[DirectorLLMInterface] = None,
    manifest: Optional[ManifestDB] = None,
    max_iterations: int = 3,
) -> DirectorEngineInterface:
    """Factory creating Director engine by strategy name ('langgraph' or 'pydantic')."""
    if engine_name.lower() == "pydantic":
        return PydanticDirectorEngine(
            embedder=embedder,
            qdrant=qdrant,
            collection_name=collection_name,
            llm=llm,
            manifest=manifest,
            max_iterations=max_iterations,
        )
    return LangGraphDirectorEngine(
        embedder=embedder,
        qdrant=qdrant,
        collection_name=collection_name,
        llm=llm,
        manifest=manifest,
        max_iterations=max_iterations,
    )
