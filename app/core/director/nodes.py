"""
LangGraph Node implementations for Director Agent state machine.
Includes Planner, Multi-Signal Retrieval, Drafting with Chronological Sort & Auto-fill,
Editor with Multi-Factor Evaluation & Best-Draft Watermarking, and Compiler nodes.
"""

import logging
import time
import datetime
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from app.core.director.state import (
    DirectorState,
    PlannerOutput,
    DraftingOutput,
    DraftingSegmentChoice,
    EditorOutput,
    TimelineSegment,
    CandidateItem,
)
from app.core.director.profiles import get_creative_profile
from app.core.director.llm import DirectorLLMInterface
from app.core.embedder import EmbedderInterface
from app.db.qdrant import QdrantVectorDB
from app.db.manifest import ManifestDB
from app.db.models import TimelineSegmentRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PLANNER NODE
# ---------------------------------------------------------------------------

def make_planner_node(
    llm: DirectorLLMInterface,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Planner node."""

    def planner_node(state: DirectorState) -> Dict[str, Any]:
        t0 = time.time()
        user_prompt = state.get("user_prompt", "")
        target_duration = state.get("target_duration", 30)
        existing_queries = state.get("search_queries", [])
        existing_narrative = state.get("narrative_arc", "")

        # If already planned, reuse queries and skip LLM invocation
        if existing_queries:
            logger.info("Reusing %d pre-planned sub-queries", len(existing_queries))
            return {
                "search_queries": existing_queries,
                "narrative_arc": existing_narrative,
            }

        system_prompt = (
            "You are an AI that breaks down video prompts into simple search queries for a visual database.\n"
            "Rules:\n"
            "1. Generate 8 to 15 ULTRA-SHORT, HIGHLY-RELEVANT search queries.\n"
            "2. MAXIMUM 2 words per query. Rarely use 3 words only if absolutely necessary for context.\n"
            "3. Queries MUST be highly specific to the user input to pinpoint the exact visual data.\n"
            "4. Queries MUST be distinct and mutually exclusive to avoid retrieving duplicate results.\n"
            "5. Do NOT use cinematic language like 'wide angle', 'slow motion', 'time lapse', 'drone shot'.\n\n"
            "Example - Prompt: 'Mountain trek adventure with friends'\n"
            "Good queries: ['mountain', 'hiking', 'friends', 'backpack', "
            "'river crossing', 'tent', 'sunrise', 'rocks', 'valley', 'walking', "
            "'forest', 'snow']\n\n"
            "Example - Prompt: 'Beach vacation family fun'\n"
            "Good queries: ['beach', 'family', 'sunset', 'swimming', "
            "'sandcastle', 'palm trees', 'children', 'boat', 'seafood', "
            "'umbrella', 'waves', 'group']"
        )
        user_msg = (
            f"User Prompt: '{user_prompt}'\n"
            f"Target Video Duration: {target_duration} seconds.\n\n"
            "Generate 8-15 ultra-short, highly relevant visual search queries (max 2-3 words) without overlap, and establish a narrative mood.\n"
            "Output ONLY valid JSON."
        )

        output: PlannerOutput = llm.structured_generate(
            system_prompt=system_prompt,
            user_prompt=user_msg,
            response_schema=PlannerOutput,
        )
        queries = output.search_queries
        narrative = output.mood_or_narrative

        elapsed = round(time.time() - t0, 3)
        telemetry_item = {
            "node": "PLANNER",
            "stage": "PLANNING",
            "latency_seconds": elapsed,
            "queries": queries,
            "narrative_arc": narrative,
            "llm_telemetry": getattr(llm, "last_telemetry", {}),
            "summary": f"Planned {len(queries)} sub-queries in {elapsed:.2f}s: {narrative}",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "search_queries": queries,
            "narrative_arc": narrative,
            "agent_telemetry": current_telemetry,
        }

    return planner_node


# ---------------------------------------------------------------------------
# 2. RETRIEVAL NODE (v3: Multi-Signal Aesthetic Ranking & Visual Diversity)
# ---------------------------------------------------------------------------

def make_retrieval_node(
    embedder: EmbedderInterface,
    qdrant: QdrantVectorDB,
    collection_name: str,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Multi-Signal Retrieval node."""

    def retrieval_node(state: DirectorState) -> Dict[str, Any]:
        t0 = time.time()
        queries = state.get("search_queries", [])
        mode = state.get("retrieval_mode", "dual")
        profile_id = state.get("creative_profile", "journey")
        profile = state.get("creative_config") or get_creative_profile(profile_id)
        rank_weights = profile.get("rank_weights", {"cosine": 0.50, "nima": 0.20, "epic_sim": 0.15, "technical": 0.15})
        epic_ref_vec = state.get("epic_reference_vector")

        candidates_map: Dict[str, CandidateItem] = {}
        query_breakdown: List[Dict[str, Any]] = []

        for query in queries:
            q_t0 = time.time()
            query_vec = embedder.embed_text(query)
            q_embed_ms = round((time.time() - q_t0) * 1000, 1)

            matched_scenes = 0
            matched_frames = 0

            # Pass 1: Scene-level search (if mode is 'scene' or 'dual')
            if mode in ("scene", "dual"):
                try:
                    scene_results = qdrant.search(
                        collection_name=collection_name,
                        query_vector=query_vec,
                        limit=15,
                        is_scene_representative=True,
                    )
                    matched_scenes = len(scene_results)
                    for r in scene_results:
                        key = f"{r.file_path}:scene_{r.scene_id}"
                        if key not in candidates_map or r.score > candidates_map[key].score:
                            candidates_map[key] = CandidateItem(
                                file_path=r.file_path,
                                file_id=r.file_id,
                                file_type=r.file_type,
                                score=r.score,
                                scores=r.scores,
                                source_offset=r.source_offset,
                                duration_seconds=r.duration_seconds,
                                granularity="scene",
                                scene_id=r.scene_id,
                                scene_start=r.scene_start,
                                scene_end=r.scene_end,
                                matched_query=query,
                                creation_timestamp=r.creation_timestamp,
                            )
                except Exception as e:
                    logger.debug("Scene search notice: %s", e)

            # Pass 2: Frame-level search (if mode is 'frame' or 'dual')
            if mode in ("frame", "dual"):
                try:
                    frame_results = qdrant.search(
                        collection_name=collection_name,
                        query_vector=query_vec,
                        limit=15,
                        granularity="frame",
                    )
                    matched_frames = len(frame_results)
                    for r in frame_results:
                        key = f"{r.file_path}:offset_{r.source_offset}"
                        if key not in candidates_map or r.score > candidates_map[key].score:
                            candidates_map[key] = CandidateItem(
                                file_path=r.file_path,
                                file_id=r.file_id,
                                file_type=r.file_type,
                                score=r.score,
                                scores=r.scores,
                                source_offset=r.source_offset,
                                duration_seconds=r.duration_seconds,
                                granularity="frame",
                                scene_id=r.scene_id,
                                scene_start=r.scene_start,
                                scene_end=r.scene_end,
                                matched_query=query,
                                creation_timestamp=r.creation_timestamp,
                            )
                except Exception as e:
                    logger.debug("Frame search notice: %s", e)

            query_breakdown.append({
                "query": query,
                "embed_ms": q_embed_ms,
                "matched_scenes": matched_scenes,
                "matched_frames": matched_frames,
            })

        # Calculate composite ranking score for every candidate based on active scoring signals / custom weights
        all_candidates = list(candidates_map.values())
        custom_weights = state.get("scoring_weights")
        if custom_weights and isinstance(custom_weights, dict) and any(float(v) > 0 for v in custom_weights.values()):
            raw_weights: Dict[str, float] = {k: float(v) for k, v in custom_weights.items() if float(v) > 0}
            active_signals = list(raw_weights.keys())
        else:
            active_signals = state.get("scoring_signals")
            if not active_signals:
                active_signals = ["cosine", "nima", "technical"]
            if "cosine" not in active_signals:
                active_signals = ["cosine"] + list(active_signals)

            # Calculate raw weights from profile rank_weights
            raw_weights: Dict[str, float] = {}
            for s in active_signals:
                if s == "epic_sim":
                    if epic_ref_vec:
                        raw_weights["epic_sim"] = rank_weights.get("epic_sim", 0.15)
                elif s in rank_weights:
                    raw_weights[s] = rank_weights.get(s, 0.1)
                elif s == "technical":
                    raw_weights["technical"] = rank_weights.get("technical", 0.10)

        total_w = sum(raw_weights.values())
        if total_w > 0:
            norm_weights = {k: v / total_w for k, v in raw_weights.items()}
        else:
            norm_weights = {"cosine": 1.0}

        for c in all_candidates:
            c_scores = c.scores or {}
            
            # Map cosine similarity (SigLIP2 ~0.08 to 0.25) to normalized [0.0, 1.0]
            cos_norm = float(np.clip((c.score - 0.05) / 0.20, 0.0, 1.0))
            
            nima_val = float(c_scores.get("nima_aesthetic", 0.5))
            sharp_val = float(c_scores.get("sharpness", 0.5))
            cont_val = float(c_scores.get("contrast", 0.5))
            color_val = float(c_scores.get("colorfulness", 0.5))
            tech_val = float(c_scores.get("technical_composite", 0.5))
            
            epic_sim_val = 0.0
            if epic_ref_vec and c.vector:
                epic_sim_val = float(np.dot(c.vector, epic_ref_vec))
                epic_sim_val = float(np.clip((epic_sim_val - 0.05) / 0.20, 0.0, 1.0))

            signal_values = {
                "cosine": cos_norm,
                "nima": nima_val,
                "sharpness": sharp_val,
                "contrast": cont_val,
                "colorfulness": color_val,
                "epic_sim": epic_sim_val,
                "technical": tech_val,
            }

            comp_rank = sum(norm_weights[s] * signal_values[s] for s in norm_weights if s in signal_values)
            c.composite_rank = round(comp_rank, 4)

        # Sort by composite rank descending
        all_candidates.sort(key=lambda x: x.composite_rank or x.score, reverse=True)

        # Filter out low-confidence candidates (SigLIP2 scores below 0.08 are noise)
        MIN_SCORE_THRESHOLD = 0.08
        before_filter = len(all_candidates)
        filtered_candidates = [c for c in all_candidates if c.score >= MIN_SCORE_THRESHOLD]
        if not filtered_candidates and all_candidates:
            filtered_candidates = all_candidates[:10]
        filtered_out = before_filter - len(filtered_candidates)

        # Apply per-source-file cap (max 2 clips per source file)
        max_clips = profile.get("max_clips_per_file", 2)
        capped_candidates = []
        file_counts: Dict[str, int] = {}
        for c in filtered_candidates:
            cnt = file_counts.get(c.file_path, 0)
            if cnt < max_clips:
                capped_candidates.append(c)
                file_counts[c.file_path] = cnt + 1

        elapsed = round(time.time() - t0, 3)

        telemetry_item = {
            "node": "RETRIEVAL",
            "stage": "RETRIEVAL",
            "latency_seconds": elapsed,
            "mode": mode,
            "profile": profile.get("name", profile_id),
            "scoring_signals": active_signals,
            "scoring_weights": {k: round(v, 4) for k, v in norm_weights.items()},
            "total_candidates": len(capped_candidates),
            "filtered_out_low_score": filtered_out,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "query_breakdown": query_breakdown,
            "top_candidates_preview": [
                {
                    "file": c.file_path.split("/")[-1],
                    "cosine_score": round(c.score, 3),
                    "composite_rank": c.composite_rank,
                    "nima": (c.scores or {}).get("nima_aesthetic"),
                    "granularity": c.granularity,
                    "matched_query": c.matched_query,
                }
                for c in capped_candidates[:6]
            ],
            "summary": f"Retrieved {len(capped_candidates)} candidates across {len(queries)} queries with {profile.get('name', profile_id)} scoring in {elapsed:.2f}s",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "retrieved_candidates": [c.model_dump() for c in capped_candidates],
            "agent_telemetry": current_telemetry,
        }

    return retrieval_node


# ---------------------------------------------------------------------------
# 3. DRAFTING NODE (v3: Strict Chronological Order & Intelligent Auto-Fill)
# ---------------------------------------------------------------------------

def make_drafting_node(
    llm: DirectorLLMInterface,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Drafting node."""

    def drafting_node(state: DirectorState) -> Dict[str, Any]:
        t0 = time.time()
        user_prompt = state.get("user_prompt", "")
        target_duration = state.get("target_duration", 30)
        candidates = state.get("retrieved_candidates", [])
        editor_feedback = state.get("editor_feedback", [])
        narrative = state.get("narrative_arc", "")
        profile_id = state.get("creative_profile", "journey")
        profile = state.get("creative_config") or get_creative_profile(profile_id)

        # Pre-compute segment guidance
        min_factor = profile.get("min_segments_factor", 1.0)
        min_segments = max(4, int((target_duration // 4) * min_factor))
        max_segments = max(min_segments + 4, int((target_duration // 2) * min_factor))
        emphasis = profile.get("drafter_emphasis", "")

        system_prompt = (
            "You are an Expert Film Editor assembling a video storyboard from curated candidate media.\n"
            "Rules:\n"
            f"1. TARGET DURATION: Total storyboard MUST be between {target_duration} and {int(target_duration * 1.25)} seconds.\n"
            f"2. SEGMENT COUNT: You MUST select between {min_segments} and {max_segments} segments.\n"
            "3. For images: assign duration 1.5 to 2.5 seconds (MAX 2.5s per photo).\n"
            "4. For video clips: select start_offset and end_offset spanning 2.5 to 7.0 seconds.\n"
            "5. NO DUPLICATE FILES — each file_path should appear at most once.\n"
            "6. Prefer candidates with higher composite rank.\n"
            f"7. {emphasis}\n"
            "8. Output ONLY valid JSON matching the schema."
        )

        candidates_preview = []
        for idx, c in enumerate(candidates[:35]):
            ts = c.get("creation_timestamp")
            if ts:
                try:
                    dt = datetime.datetime.fromtimestamp(ts)
                    date_str = dt.strftime("%b %d %H:%M")
                except (ValueError, OSError):
                    date_str = "unknown"
            else:
                date_str = "unknown"

            score = c.get("score", 0)
            comp_rank = c.get("composite_rank", score)
            nima_val = (c.get("scores") or {}).get("nima_aesthetic", "N/A")
            fname = c.get("file_path", "").split("/")[-1]

            candidates_preview.append(
                f"[{idx}] {fname} | Date: {date_str} | Type: {c.get('file_type')} | "
                f"Rank: {comp_rank:.3f} | NIMA: {nima_val} | Offset: {c.get('source_offset', 0):.1f}s | "
                f"Path: {c.get('file_path')}"
            )

        user_msg = (
            f"Prompt: {user_prompt}\n"
            f"Narrative Goal: {narrative}\n"
            f"Target Duration: {target_duration}s (Allowed: {target_duration}s - {int(target_duration * 1.25)}s)\n"
            f"Required Segment Count: {min_segments} - {max_segments} moments\n"
        )
        if editor_feedback:
            user_msg += f"\nPrevious Editor Feedback:\n- " + "\n- ".join(editor_feedback) + "\n"

        user_msg += "\nAvailable Candidates (sorted by quality rank):\n" + "\n".join(candidates_preview)

        storyboard_list: List[Dict[str, Any]] = []
        output: DraftingOutput = llm.structured_generate(
            system_prompt=system_prompt,
            user_prompt=user_msg,
            response_schema=DraftingOutput,
        )

        # Candidates lookup by path
        cand_by_path = {c.get("file_path"): c for c in candidates}

        for seg in output.storyboard:
            seg_dict = seg.model_dump()
            matched = cand_by_path.get(seg_dict.get("file_path"))
            if matched:
                if not seg_dict.get("creation_timestamp"):
                    seg_dict["creation_timestamp"] = matched.get("creation_timestamp")
                if not seg_dict.get("file_id"):
                    seg_dict["file_id"] = matched.get("file_id")
                if not seg_dict.get("similarity_score"):
                    seg_dict["similarity_score"] = matched.get("score")
                if not seg_dict.get("composite_rank"):
                    seg_dict["composite_rank"] = matched.get("composite_rank")
                if not seg_dict.get("scores"):
                    seg_dict["scores"] = matched.get("scores", {})
            storyboard_list.append(seg_dict)

        # Deduplicate paths
        unique_storyboard = []
        seen_paths = set()
        for s in storyboard_list:
            fp = s.get("file_path")
            if fp and fp not in seen_paths:
                seen_paths.add(fp)
                unique_storyboard.append(s)
        storyboard_list = unique_storyboard

        # -------------------------------------------------------------------
        # POST-PROCESSING 1: Auto-Fill if Under Target Duration
        # -------------------------------------------------------------------
        current_duration = sum(s.get("duration", 3.0) for s in storyboard_list)
        if current_duration < float(target_duration) and candidates:
            logger.info("Drafted duration (%.1fs) < target (%ds). Auto-filling top candidates...", current_duration, target_duration)
            for c in candidates:
                if current_duration >= float(target_duration):
                    break
                fp = c.get("file_path")
                if fp in seen_paths:
                    continue
                seen_paths.add(fp)

                is_vid = c.get("file_type") == "video"
                dur = 4.0 if is_vid else 2.5
                start_off = float(c.get("source_offset", 0.0))
                end_off = start_off + dur if is_vid else 0.0

                storyboard_list.append({
                    "file_path": fp,
                    "file_id": c.get("file_id"),
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "duration": dur,
                    "segment_type": "video_clip" if is_vid else "image",
                    "scene_id": c.get("scene_id"),
                    "retrieval_strategy": c.get("granularity", "frame"),
                    "similarity_score": c.get("score"),
                    "composite_rank": c.get("composite_rank"),
                    "scores": c.get("scores", {}),
                    "creation_timestamp": c.get("creation_timestamp"),
                    "justification": f"Auto-fill to reach target duration (Rank={c.get('composite_rank', 0):.3f})",
                })
                current_duration += dur

        # -------------------------------------------------------------------
        # POST-PROCESSING 2: Tail Trim if Over Maximum Upper Bound (+25%)
        # -------------------------------------------------------------------
        max_allowed_dur = float(target_duration) * 1.25
        if current_duration > max_allowed_dur and len(storyboard_list) > min_segments:
            logger.info("Drafted duration (%.1fs) > upper bound (%.1fs). Trimming lowest-scored segments...", current_duration, max_allowed_dur)
            while current_duration > max_allowed_dur and len(storyboard_list) > min_segments:
                # Find lowest ranked segment
                lowest_idx = min(
                    range(len(storyboard_list)),
                    key=lambda i: storyboard_list[i].get("composite_rank") or storyboard_list[i].get("similarity_score") or 0.0
                )
                removed = storyboard_list.pop(lowest_idx)
                current_duration -= removed.get("duration", 3.0)

        # -------------------------------------------------------------------
        # POST-PROCESSING 3: Force Strict Chronological Sort
        # -------------------------------------------------------------------
        storyboard_list.sort(key=lambda s: s.get("creation_timestamp") or 0)

        elapsed = round(time.time() - t0, 3)
        total_cut_dur = sum(s.get("duration", 0) for s in storyboard_list)
        telemetry_item = {
            "node": "DRAFTING",
            "stage": "DRAFTING",
            "latency_seconds": elapsed,
            "drafted_segments": len(storyboard_list),
            "total_duration": round(total_cut_dur, 1),
            "llm_telemetry": getattr(llm, "last_telemetry", {}),
            "summary": f"Drafted & chronologically sorted {len(storyboard_list)} segments ({total_cut_dur:.1f}s) in {elapsed:.2f}s",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "storyboard": storyboard_list,
            "agent_telemetry": current_telemetry,
        }

    return drafting_node


# ---------------------------------------------------------------------------
# 4. EDITOR NODE (v3: Multi-Factor Evaluation & Best-Draft Watermarking)
# ---------------------------------------------------------------------------

def make_editor_node(
    llm: DirectorLLMInterface,
    max_iterations: int = 3,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Editor critique node."""

    def editor_node(state: DirectorState) -> Dict[str, Any]:
        t0 = time.time()
        iteration_count = state.get("iteration_count", 0) + 1
        target_duration = state.get("target_duration", 30)
        storyboard = state.get("storyboard", [])
        best_storyboard = state.get("best_storyboard", [])
        best_composite = state.get("best_composite_score", 0.0)

        total_duration = sum(s.get("duration", 3.0) for s in storyboard)
        min_allowed_dur = float(target_duration)
        max_allowed_dur = float(target_duration) * 1.25

        # 1. Deterministic Evaluation Dimensions
        # A. Duration accuracy (0.0 to 1.0)
        if total_duration < min_allowed_dur:
            duration_score = max(0.0, total_duration / min_allowed_dur)
        elif total_duration > max_allowed_dur:
            duration_score = max(0.0, 1.0 - (total_duration - max_allowed_dur) / target_duration)
        else:
            duration_score = 1.0

        # B. Pacing rhythm variety
        durations = [s.get("duration", 3.0) for s in storyboard]
        pacing_std = float(np.std(durations)) if durations else 0.0
        pacing_variety_score = float(np.clip(pacing_std / 1.0, 0.2, 1.0))

        # C. Visual quality mean (NIMA)
        nima_scores = [(s.get("scores") or {}).get("nima_aesthetic", 0.5) for s in storyboard]
        visual_quality_score = float(np.mean(nima_scores)) if nima_scores else 0.5

        # D. Query relevance mean
        rel_scores = [s.get("similarity_score") or 0.10 for s in storyboard]
        query_rel_score = float(np.clip((np.mean(rel_scores) - 0.05) / 0.18, 0.0, 1.0)) if rel_scores else 0.5

        # E. Diversity (time spread & media type mix)
        types = {s.get("segment_type") for s in storyboard}
        diversity_score = 1.0 if len(types) > 1 else 0.7

        # Objective Composite Evaluation Score (0.0 - 10.0 scale)
        composite_score = round(
            10.0 * (
                0.25 * duration_score +
                0.20 * pacing_variety_score +
                0.25 * visual_quality_score +
                0.20 * query_rel_score +
                0.10 * diversity_score
            ),
            2,
        )

        feedback_notes = []
        if total_duration < min_allowed_dur:
            feedback_notes.append(f"Timeline is too short ({total_duration:.1f}s vs target {target_duration}s). Add more segments.")
        elif total_duration > max_allowed_dur:
            feedback_notes.append(f"Timeline exceeds upper bound ({total_duration:.1f}s vs max {max_allowed_dur:.1f}s). Trim segments.")

        # Best-Draft Watermarking
        if composite_score >= best_composite or not best_storyboard:
            best_storyboard = list(storyboard)
            best_composite = composite_score
            logger.info("New best storyboard watermarked at iteration %d (Score: %.2f/10)", iteration_count, best_composite)

        # If reached max iterations, approve and revert to best draft if current degraded
        if iteration_count >= max_iterations:
            logger.info("Director reached max iterations (%d). Finalizing best watermarked draft (Score: %.2f/10).", iteration_count, best_composite)
            elapsed = round(time.time() - t0, 3)
            telemetry_item = {
                "node": "EDITOR",
                "stage": "EDITING",
                "latency_seconds": elapsed,
                "approved": True,
                "iteration": iteration_count,
                "composite_score": best_composite,
                "feedback": ["Max iterations reached; approved highest-scoring storyboard."],
                "summary": f"Editor finalized best storyboard (Score: {best_composite:.1f}/10, {len(best_storyboard)} moments) in {elapsed:.2f}s",
            }
            if step_callback:
                step_callback(telemetry_item)
            current_telemetry = list(state.get("agent_telemetry", []))
            current_telemetry.append(telemetry_item)
            return {
                "approved": True,
                "storyboard": best_storyboard,
                "iteration_count": iteration_count,
                "best_storyboard": best_storyboard,
                "best_composite_score": best_composite,
                "editor_feedback": ["Approved highest-scoring storyboard."],
                "agent_telemetry": current_telemetry,
            }

        # LLM qualitative critique
        approved = (len(feedback_notes) == 0) and (composite_score >= 7.5)
        system_prompt = (
            "You are a Senior Film Editor performing quality control on a proposed storyboard.\n"
            "Review pacing, variety, and cinematic flow. Output ONLY valid JSON."
        )
        user_msg = (
            f"Target Duration: {target_duration}s (Allowed: {min_allowed_dur}s - {max_allowed_dur}s)\n"
            f"Actual Duration: {total_duration:.1f}s | Moments: {len(storyboard)}\n"
            f"Composite Quality Score: {composite_score}/10\n"
        )
        critique: EditorOutput = llm.structured_generate(
            system_prompt=system_prompt,
            user_prompt=user_msg,
            response_schema=EditorOutput,
        )
        if not critique.approved:
            approved = False
            feedback_notes.append(critique.feedback)
            feedback_notes.extend(critique.suggested_modifications)

        elapsed = round(time.time() - t0, 3)
        telemetry_item = {
            "node": "EDITOR",
            "stage": "EDITING",
            "latency_seconds": elapsed,
            "approved": approved,
            "pacing_score": critique.pacing_score,
            "composite_score": composite_score,
            "iteration": iteration_count,
            "feedback": feedback_notes,
            "llm_telemetry": getattr(llm, "last_telemetry", {}),
            "summary": f"Editor {'approved ✅' if approved else 'requested revisions ⚠️'} (Composite Score: {composite_score}/10, Duration: {total_duration:.1f}s) in {elapsed:.2f}s",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "approved": approved,
            "storyboard": storyboard if approved else (best_storyboard if best_storyboard else storyboard),
            "iteration_count": iteration_count,
            "best_storyboard": best_storyboard,
            "best_composite_score": best_composite,
            "editor_feedback": feedback_notes,
            "agent_telemetry": current_telemetry,
        }

    return editor_node


# ---------------------------------------------------------------------------
# 5. COMPILER NODE
# ---------------------------------------------------------------------------

def make_compiler_node(
    manifest: Optional[ManifestDB] = None,
    job_id: Optional[str] = None,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Compiler node that persists the approved storyboard."""

    def compiler_node(state: DirectorState) -> Dict[str, Any]:
        t0 = time.time()
        storyboard = state.get("storyboard", [])
        best_storyboard = state.get("best_storyboard")
        if best_storyboard and len(best_storyboard) > 0:
            storyboard = best_storyboard

        segments: List[TimelineSegment] = []

        for s in storyboard:
            seg = TimelineSegment(
                file_path=s.get("file_path", ""),
                file_id=s.get("file_id"),
                start_offset=s.get("start_offset", 0.0),
                end_offset=s.get("end_offset", 0.0),
                duration=s.get("duration", 3.0),
                segment_type=s.get("segment_type", "image"),
                scene_id=s.get("scene_id"),
                retrieval_strategy=s.get("retrieval_strategy", "frame"),
                similarity_score=s.get("similarity_score"),
                composite_rank=s.get("composite_rank"),
                scores=s.get("scores", {}),
                justification=s.get("justification", ""),
                creation_timestamp=s.get("creation_timestamp"),
            )
            segments.append(seg)

        # If manifest and job_id are available, save to SQLite
        if manifest and job_id:
            try:
                records = []
                for idx, seg in enumerate(segments):
                    file_id = seg.file_id
                    if file_id is None:
                        rec = manifest.lookup(seg.file_path)
                        if rec and rec.id:
                            file_id = rec.id

                    if file_id is not None:
                        records.append(
                            TimelineSegmentRecord(
                                id=None,
                                job_id=job_id,
                                position=idx,
                                file_id=file_id,
                                segment_type=seg.segment_type,
                                duration=seg.duration,
                                start_offset=seg.start_offset,
                                similarity_score=seg.composite_rank or seg.similarity_score,
                                time_bucket=idx,
                            )
                        )
                if records:
                    manifest.save_timeline(job_id, records)
                    logger.info("Compiled %d timeline segments to manifest for job %s", len(records), job_id)
            except Exception as e:
                logger.error("Failed saving timeline segments to manifest: %s", e)

        elapsed = round(time.time() - t0, 3)
        telemetry_item = {
            "node": "COMPILER",
            "stage": "COMPILING",
            "latency_seconds": elapsed,
            "compiled_segments": len(segments),
            "summary": f"Compiled {len(segments)} segments to SQLite timeline in {elapsed:.2f}s",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "storyboard": [s.model_dump() for s in segments],
            "agent_telemetry": current_telemetry,
        }

    return compiler_node
