"""
LangGraph Node implementations for Director Agent state machine.
Includes Planner, Retrieval v2, Drafting, Editor, and Compiler nodes.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel

from app.core.director.state import (
    DirectorState,
    PlannerOutput,
    DraftingOutput,
    DraftingSegmentChoice,
    EditorOutput,
    TimelineSegment,
    CandidateItem,
)
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
        import time
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
            "1. Generate 8 to 15 SHORT, CONCRETE search queries (2-4 words each).\n"
            "2. Use plain visual descriptions like 'mountain landscape', 'group of friends', 'river flowing'.\n"
            "3. Do NOT use cinematic language like 'wide angle', 'slow motion', 'time lapse', 'drone shot'.\n"
            "4. Include a mix of: landscapes, people, activities, objects, and atmosphere.\n"
            "5. Include at least 2 broad fallback queries like 'outdoor scenery' or 'people walking'.\n\n"
            "Example - Prompt: 'Mountain trek adventure with friends'\n"
            "Good queries: ['mountain landscape', 'hiking trail', 'group of friends outdoors', 'backpack gear', "
            "'river crossing', 'tent camping', 'sunrise sky', 'rocky terrain', 'valley view', 'people walking', "
            "'forest path', 'snow on mountain']\n\n"
            "Example - Prompt: 'Beach vacation family fun'\n"
            "Good queries: ['beach sand ocean', 'family playing', 'sunset over water', 'swimming', "
            "'sandcastle', 'palm trees', 'children laughing', 'boat on water', 'seafood plate', "
            "'beach umbrella', 'waves crashing', 'group photo outdoors']"
        )
        user_msg = (
            f"User Prompt: '{user_prompt}'\n"
            f"Target Video Duration: {target_duration} seconds.\n\n"
            "Generate 8-15 short, concrete visual search queries and establish a narrative mood.\n"
            "Output ONLY valid JSON."
        )

        try:
            output: PlannerOutput = llm.structured_generate(
                system_prompt=system_prompt,
                user_prompt=user_msg,
                response_schema=PlannerOutput,
            )
            queries = output.search_queries
            narrative = output.mood_or_narrative
        except Exception as e:
            logger.warning("Planner LLM generation error: %s. Using default queries.", e)
            queries = [
                user_prompt,
                f"{user_prompt} scenery",
                f"{user_prompt} people",
                f"{user_prompt} outdoors",
                "landscape scenery",
                "people outdoors",
            ]
            narrative = "Direct match highlights montage."

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
# 2. RETRIEVAL NODE (v2: Dual-Granularity Search with Score Thresholding)
# ---------------------------------------------------------------------------

def make_retrieval_node(
    embedder: EmbedderInterface,
    qdrant: QdrantVectorDB,
    collection_name: str,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Dual-Granularity Retrieval node."""

    def retrieval_node(state: DirectorState) -> Dict[str, Any]:
        import time
        t0 = time.time()
        queries = state.get("search_queries", [])
        mode = state.get("retrieval_mode", "dual")
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

        # Convert to serialized dicts sorted by relevance score
        sorted_candidates = sorted(
            candidates_map.values(), key=lambda x: x.score, reverse=True
        )

        # Filter out low-confidence candidates (SigLIP2 scores below 0.08 are noise)
        MIN_SCORE_THRESHOLD = 0.08
        before_filter = len(sorted_candidates)
        filtered_candidates = [c for c in sorted_candidates if c.score >= MIN_SCORE_THRESHOLD]
        # If all candidates filtered out, keep top 10 as fallback
        if not filtered_candidates and sorted_candidates:
            filtered_candidates = sorted_candidates[:10]
        filtered_out = before_filter - len(filtered_candidates)
        sorted_candidates = filtered_candidates

        elapsed = round(time.time() - t0, 3)

        telemetry_item = {
            "node": "RETRIEVAL",
            "stage": "RETRIEVAL",
            "latency_seconds": elapsed,
            "mode": mode,
            "total_candidates": len(sorted_candidates),
            "filtered_out_low_score": filtered_out,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "query_breakdown": query_breakdown,
            "top_candidates_preview": [
                {
                    "file": c.file_path.split("/")[-1],
                    "score": round(c.score, 3),
                    "granularity": c.granularity,
                    "matched_query": c.matched_query,
                }
                for c in sorted_candidates[:6]
            ],
            "summary": f"Retrieved {len(sorted_candidates)} candidates across {len(queries)} queries in {elapsed:.2f}s (filtered {filtered_out} noise items < {MIN_SCORE_THRESHOLD})",
        }

        if step_callback:
            try:
                step_callback(telemetry_item)
            except Exception as e:
                logger.debug("Step callback notice: %s", e)

        current_telemetry = list(state.get("agent_telemetry", []))
        current_telemetry.append(telemetry_item)

        return {
            "retrieved_candidates": [c.model_dump() for c in sorted_candidates],
            "agent_telemetry": current_telemetry,
        }

    return retrieval_node


# ---------------------------------------------------------------------------
# 3. DRAFTING NODE
# ---------------------------------------------------------------------------

def make_drafting_node(
    llm: DirectorLLMInterface,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Drafting node."""

    def drafting_node(state: DirectorState) -> Dict[str, Any]:
        import time
        import datetime
        t0 = time.time()
        user_prompt = state.get("user_prompt", "")
        target_duration = state.get("target_duration", 30)
        candidates = state.get("retrieved_candidates", [])
        editor_feedback = state.get("editor_feedback", [])
        narrative = state.get("narrative_arc", "")

        system_prompt = (
            "You are a Film Editor assembling a video storyboard from candidate media.\n"
            "Rules:\n"
            f"1. Target total duration: approximately {target_duration} seconds.\n"
            "2. For images: assign duration 2.0 to 4.0 seconds.\n"
            "3. For video clips: select start_offset and end_offset spanning 2.0 to 6.0 seconds.\n"
            "4. CRITICAL: Arrange segments in CHRONOLOGICAL order by their capture date (earliest first).\n"
            "5. Do NOT pick duplicate files — each file_path should appear at most once.\n"
            "6. Prefer candidates with higher scores (closer to 0.15 is better).\n"
            "7. Skip candidates with scores below 0.09 unless no better option exists.\n"
            "8. Mix images and video clips for visual variety.\n"
            "9. You MUST use the exact file_path values from the candidate list below.\n"
            "10. Output ONLY valid JSON matching the schema."
        )

        candidates_preview = []
        for idx, c in enumerate(candidates[:30]):
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
            score_flag = " ⚠️LOW" if score < 0.09 else ""
            fname = c.get("file_path", "").split("/")[-1]

            candidates_preview.append(
                f"[{idx}] {fname} | Date: {date_str} | Type: {c.get('file_type')} | "
                f"Score: {score:.3f}{score_flag} | Offset: {c.get('source_offset', 0):.1f}s | "
                f"Gran: {c.get('granularity')} | Path: {c.get('file_path')}"
            )

        user_msg = (
            f"Prompt: {user_prompt}\n"
            f"Narrative Goal: {narrative}\n"
            f"Target Duration: {target_duration}s\n"
        )
        if editor_feedback:
            user_msg += f"\nPrevious Editor Feedback (Fix these issues!):\n- " + "\n- ".join(editor_feedback) + "\n"

        user_msg += "\nAvailable Candidates (sorted by relevance score, pick in DATE order):\n" + "\n".join(candidates_preview)

        storyboard_list: List[Dict[str, Any]] = []
        try:
            output: DraftingOutput = llm.structured_generate(
                system_prompt=system_prompt,
                user_prompt=user_msg,
                response_schema=DraftingOutput,
            )
            for seg in output.storyboard:
                seg_dict = seg.model_dump()
                # Ensure creation_timestamp is preserved from matched candidate
                if not seg_dict.get("creation_timestamp"):
                    matched_cand = next((c for c in candidates if c.get("file_path") == seg_dict.get("file_path")), None)
                    if matched_cand:
                        seg_dict["creation_timestamp"] = matched_cand.get("creation_timestamp")
                storyboard_list.append(seg_dict)
        except Exception as e:
            logger.warning("Drafting LLM generation error: %s. Using heuristic assembly.", e)

        # Fallback heuristic: If LLM produced 0 segments, pick candidates until duration is reached
        if not storyboard_list and candidates:
            # Sort chronologically for fallback
            sorted_by_time = sorted(candidates, key=lambda c: c.get("creation_timestamp") or 0)
            accumulated_dur = 0.0
            used_paths = set()
            for c in sorted_by_time:
                if accumulated_dur >= target_duration:
                    break
                fp = c.get("file_path")
                if fp in used_paths:
                    continue
                used_paths.add(fp)

                dur = 3.0
                if c.get("file_type") == "video":
                    start_off = float(c.get("source_offset", 0.0))
                    dur = 4.0
                    end_off = start_off + dur
                else:
                    start_off = 0.0
                    end_off = 0.0

                storyboard_list.append({
                    "file_path": fp,
                    "file_id": c.get("file_id"),
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "duration": dur,
                    "segment_type": "video_clip" if c.get("file_type") == "video" else "image",
                    "scene_id": c.get("scene_id"),
                    "retrieval_strategy": c.get("granularity", "frame"),
                    "similarity_score": c.get("score"),
                    "creation_timestamp": c.get("creation_timestamp"),
                    "justification": f"High relevance match for {c.get('matched_query', 'prompt')}",
                })
                accumulated_dur += dur

        elapsed = round(time.time() - t0, 3)
        total_cut_dur = sum(s.get("duration", 0) for s in storyboard_list)
        telemetry_item = {
            "node": "DRAFTING",
            "stage": "DRAFTING",
            "latency_seconds": elapsed,
            "drafted_segments": len(storyboard_list),
            "total_duration": round(total_cut_dur, 1),
            "llm_telemetry": getattr(llm, "last_telemetry", {}),
            "summary": f"Drafted {len(storyboard_list)} segments ({total_cut_dur:.1f}s) in {elapsed:.2f}s",
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
# 4. EDITOR NODE
# ---------------------------------------------------------------------------

def make_editor_node(
    llm: DirectorLLMInterface,
    max_iterations: int = 3,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[DirectorState], Dict[str, Any]]:
    """Factory creating the Editor critique node."""

    def editor_node(state: DirectorState) -> Dict[str, Any]:
        import time
        import datetime
        t0 = time.time()
        iteration_count = state.get("iteration_count", 0) + 1
        target_duration = state.get("target_duration", 30)
        storyboard = state.get("storyboard", [])

        # Calculate current total duration
        total_duration = sum(s.get("duration", 3.0) for s in storyboard)

        # If we reached max iterations, force approve to terminate loop
        if iteration_count >= max_iterations:
            logger.info("Director reached max iterations (%d). Forcing approval.", iteration_count)
            elapsed = round(time.time() - t0, 3)
            telemetry_item = {
                "node": "EDITOR",
                "stage": "EDITING",
                "latency_seconds": elapsed,
                "approved": True,
                "iteration": iteration_count,
                "feedback": ["Max iterations reached; approving best draft."],
                "summary": f"Editor approved draft (max iterations {iteration_count}) in {elapsed:.2f}s",
            }
            if step_callback:
                step_callback(telemetry_item)
            current_telemetry = list(state.get("agent_telemetry", []))
            current_telemetry.append(telemetry_item)
            return {
                "approved": True,
                "iteration_count": iteration_count,
                "editor_feedback": ["Max iterations reached; approving best draft."],
                "agent_telemetry": current_telemetry,
            }

        # Check duration bounds (tolerance ±25%)
        min_dur = target_duration * 0.75
        max_dur = target_duration * 1.25

        feedback_notes = []
        if total_duration < min_dur:
            feedback_notes.append(
                f"Timeline is too short ({total_duration:.1f}s vs target {target_duration}s). Add more segments."
            )
        elif total_duration > max_dur:
            feedback_notes.append(
                f"Timeline is too long ({total_duration:.1f}s vs target {target_duration}s). Trim or remove segments."
            )

        # Structural duplicate checks
        seen_paths = set()
        for s in storyboard:
            fp = s.get("file_path")
            if fp and fp in seen_paths:
                feedback_notes.append(f"Duplicate file found in timeline: {fp.split('/')[-1]}. Replace with a unique moment.")
            if fp:
                seen_paths.add(fp)

        # Ask LLM for quality critique if available
        approved = len(feedback_notes) == 0
        pacing_score = 8.0
        try:
            system_prompt = (
                "You are a Senior Video Editor performing quality control on a proposed storyboard.\n"
                "You MUST check ALL of the following and REJECT if any fail:\n\n"
                "CHECKLIST:\n"
                "1. DURATION: Total storyboard duration must be within ±25% of target.\n"
                "2. DUPLICATES: No file should appear more than once. Reject if duplicates found.\n"
                "3. CHRONOLOGICAL ORDER: Segments should be ordered by capture date (earliest → latest). "
                "Reject if a segment from a later date appears before an earlier one.\n"
                "4. VARIETY: There should be a mix of images and video clips. "
                "Reject if all segments are the same type.\n"
                "5. PACING: Durations should vary (not all exactly 3.0s). "
                "At least some segments should be 2.0-2.5s and some 4.0-5.0s.\n"
                "6. BACK-TO-BACK: No two consecutive segments should be from the same source video file.\n"
                f"7. MINIMUM SEGMENTS: For a {target_duration}s video, expect at least "
                f"{max(3, target_duration // 6)} segments.\n\n"
                "If ALL checks pass, approve. Otherwise, list which checks failed and give specific fix instructions.\n"
                "Output ONLY valid JSON."
            )

            segment_details = []
            for idx, s in enumerate(storyboard):
                ts = s.get("creation_timestamp")
                if ts:
                    try:
                        dt = datetime.datetime.fromtimestamp(ts)
                        date_str = dt.strftime("%b %d %H:%M")
                    except (ValueError, OSError):
                        date_str = "unknown"
                else:
                    date_str = "unknown"

                fname = s.get("file_path", "").split("/")[-1]
                segment_details.append(
                    f"  [{idx}] {fname} | Date: {date_str} | Type: {s.get('segment_type')} | "
                    f"Duration: {s.get('duration', 3.0):.1f}s | Score: {s.get('similarity_score', 'N/A')}"
                )

            user_msg = (
                f"Target Duration: {target_duration}s | Actual Duration: {total_duration:.1f}s\n"
                f"Segment Count: {len(storyboard)}\n\n"
                "Proposed Storyboard:\n" + "\n".join(segment_details)
            )

            critique: EditorOutput = llm.structured_generate(
                system_prompt=system_prompt,
                user_prompt=user_msg,
                response_schema=EditorOutput,
            )
            pacing_score = critique.pacing_score
            if not critique.approved:
                approved = False
                feedback_notes.append(critique.feedback)
                feedback_notes.extend(critique.suggested_modifications)
        except Exception as e:
            logger.debug("Editor LLM notice: %s", e)

        elapsed = round(time.time() - t0, 3)
        telemetry_item = {
            "node": "EDITOR",
            "stage": "EDITING",
            "latency_seconds": elapsed,
            "approved": approved,
            "pacing_score": pacing_score,
            "iteration": iteration_count,
            "feedback": feedback_notes,
            "llm_telemetry": getattr(llm, "last_telemetry", {}),
            "summary": f"Editor {'approved ✅' if approved else 'requested revisions ⚠️'} (Pacing: {pacing_score}/10, Duration: {total_duration:.1f}s) in {elapsed:.2f}s",
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
            "iteration_count": iteration_count,
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
        import time
        t0 = time.time()
        storyboard = state.get("storyboard", [])
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
                                similarity_score=seg.similarity_score,
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

