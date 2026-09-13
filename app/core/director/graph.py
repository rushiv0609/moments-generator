"""
LangGraph State Machine construction and execution engine for Director Agent.
Supports iterative reasoning, dual-granularity retrieval, and multi-alternative timeline generation.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, START, END

from app.core.director.state import DirectorState, TimelineSegment
from app.core.director.llm import DirectorLLMInterface, get_director_llm, MockDirectorLLM
from app.core.director.nodes import (
    make_planner_node,
    make_retrieval_node,
    make_drafting_node,
    make_editor_node,
    make_compiler_node,
)
from app.core.embedder import EmbedderInterface
from app.db.qdrant import QdrantVectorDB
from app.db.manifest import ManifestDB

logger = logging.getLogger(__name__)


def build_director_graph(
    llm: DirectorLLMInterface,
    embedder: EmbedderInterface,
    qdrant: QdrantVectorDB,
    collection_name: str,
    manifest: Optional[ManifestDB] = None,
    job_id: Optional[str] = None,
    max_iterations: int = 3,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Build and compile the LangGraph Director state machine.
    """
    builder = StateGraph(DirectorState)

    # 1. Register Nodes with telemetry callbacks
    builder.add_node("planner", make_planner_node(llm, step_callback=step_callback))
    builder.add_node("retrieval", make_retrieval_node(embedder, qdrant, collection_name, step_callback=step_callback))
    builder.add_node("drafting", make_drafting_node(llm, step_callback=step_callback))
    builder.add_node("editor", make_editor_node(llm, max_iterations=max_iterations, step_callback=step_callback))
    builder.add_node("compiler", make_compiler_node(manifest, job_id, step_callback=step_callback))

    # 2. Linear Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieval")
    builder.add_edge("retrieval", "drafting")
    builder.add_edge("drafting", "editor")

    # 3. Conditional Routing from Editor
    def editor_router(state: DirectorState) -> str:
        if state.get("approved", False) or state.get("iteration_count", 0) >= max_iterations:
            return "compiler"
        return "drafting"

    builder.add_conditional_edges(
        "editor",
        editor_router,
        {
            "compiler": "compiler",
            "drafting": "drafting",
        },
    )

    builder.add_edge("compiler", END)

    return builder.compile()


class DirectorAgent:
    """
    High-level Coordinator for the LangGraph Director state machine.
    """

    def __init__(
        self,
        embedder: EmbedderInterface,
        qdrant: QdrantVectorDB,
        collection_name: str,
        llm: Optional[DirectorLLMInterface] = None,
        manifest: Optional[ManifestDB] = None,
        model_name: str = "gemma4:e4b-mlx",
        api_key: Optional[str] = None,
        max_iterations: int = 3,
    ):
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.manifest = manifest
        self.max_iterations = max_iterations
        self.llm = llm or get_director_llm(model_name=model_name, api_key=api_key)

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
    ) -> DirectorState:
        """
        Execute the Director Agent state graph on a single prompt.
        """
        from app.core.director.profiles import get_creative_profile

        profile_config = get_creative_profile(creative_profile)

        graph = build_director_graph(
            llm=self.llm,
            embedder=self.embedder,
            qdrant=self.qdrant,
            collection_name=self.collection_name,
            manifest=self.manifest,
            job_id=job_id,
            max_iterations=self.max_iterations,
            step_callback=step_callback,
        )

        initial_state: DirectorState = {
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

        try:
            final_state = graph.invoke(initial_state)
            return final_state
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
    ) -> Dict[str, DirectorState]:
        """
        Generate 3 creative candidate timelines with shared planning optimization.
        """
        results = {}
        try:
            # 1. Run Planner Node ONCE to establish creative intent and visual sub-queries
            logger.info("Executing Shared Planner for candidates...")
            planner_factory = make_planner_node(self.llm, step_callback=step_callback)
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
            plan_res = planner_factory(plan_state)
            shared_queries = plan_res.get("search_queries", [prompt])
            shared_narrative = plan_res.get("narrative_arc", "")
            shared_telemetry = plan_res.get("agent_telemetry", [])

            # 2. Generate 3 Creative Profile Candidates
            profiles_to_run = [
                ("candidate_1", "cinematic"),   # Cinematic Epic
                ("candidate_2", "personal"),    # Personal Moments & Faces
                ("candidate_3", "journey"),     # Balanced Journey Narrative
            ]

            for label, profile_id in profiles_to_run:
                logger.info("Generating Director Candidate [%s] (profile=%s)...", label, profile_id)
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
                # Prepend shared planner telemetry if needed
                if shared_telemetry and res.get("agent_telemetry"):
                    res["agent_telemetry"] = shared_telemetry + res["agent_telemetry"]
                results[label] = res

            return results
        finally:
            if hasattr(self.llm, "unload"):
                self.llm.unload()

