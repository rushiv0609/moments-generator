"""
Heuristic (non-LLM) Curator for deterministic chronological storyboard generation.
Uses temporal bucketing, diversity sampling, and score-weighted selection.
No LLM calls — pure algorithmic curation.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# SigLIP2 score thresholds (empirically measured)
SCORE_THRESHOLD_GOOD = 0.10   # High-confidence visual match
SCORE_THRESHOLD_MIN = 0.08    # Minimum acceptable (below this is noise)

# Duration assignments
IMAGE_DURATION = 2.5           # Seconds per image
VIDEO_CLIP_DURATION = 4.0      # Default seconds per video clip
VIDEO_CLIP_MIN = 2.0
VIDEO_CLIP_MAX = 6.0


class HeuristicCurator:
    """
    Generates a storyboard from candidates using deterministic algorithms.
    No LLM required. Produces chronologically-ordered, diversity-sampled timelines.
    """

    def curate(
        self,
        candidates: List[Dict[str, Any]],
        target_duration: int = 30,
        num_buckets: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Takes retrieved candidates, returns a storyboard dict
        compatible with DirectorState.

        Args:
            candidates: List of CandidateItem dicts (must have creation_timestamp).
            target_duration: Target video duration in seconds.
            num_buckets: Number of temporal buckets. Auto-detected if None.

        Returns:
            Dict with 'storyboard' and 'narrative_arc' keys.
        """
        if not candidates:
            return {"storyboard": [], "narrative_arc": "No candidates available."}

        # Step 1: Filter out noise (SigLIP2 scores below minimum threshold)
        viable = [c for c in candidates if c.get("score", 0) >= SCORE_THRESHOLD_MIN]
        if not viable:
            viable = list(candidates[:10])  # Fallback: take top 10 by whatever score

        # Step 2: Sort by creation_timestamp (chronological order)
        viable.sort(key=lambda c: c.get("creation_timestamp") or 0)

        # Step 3: Determine temporal buckets
        timestamps = [c.get("creation_timestamp") or 0 for c in viable if c.get("creation_timestamp")]
        if timestamps and len(timestamps) >= 2:
            min_ts, max_ts = min(timestamps), max(timestamps)
            span_days = max(1, (max_ts - min_ts) / 86400)  # Span in days

            if num_buckets is None:
                # Auto: 1 bucket per half-day, minimum 3, maximum 15
                num_buckets = max(3, min(15, int(span_days * 2)))
        else:
            num_buckets = max(3, min(10, len(viable) // 2))
            min_ts, max_ts = 0, 1

        # Step 4: Assign candidates to buckets
        bucket_size = max(1.0, (max_ts - min_ts + 1) / num_buckets) if max_ts > min_ts else 1.0
        buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_buckets)}

        for c in viable:
            ts = c.get("creation_timestamp") or min_ts
            bucket_idx = min(num_buckets - 1, max(0, int((ts - min_ts) / bucket_size)))
            buckets[bucket_idx].append(c)

        # Step 5: Pick best candidate per bucket (highest score, unique file_path)
        selected: List[Dict[str, Any]] = []
        used_paths = set()
        accumulated_dur = 0.0

        for bucket_idx in range(num_buckets):
            if accumulated_dur >= target_duration:
                break

            bucket_items = sorted(buckets[bucket_idx], key=lambda c: c.get("score", 0), reverse=True)

            for c in bucket_items:
                fp = c.get("file_path", "")
                if fp in used_paths:
                    continue  # Skip duplicate files

                # Determine duration
                if c.get("file_type") == "video":
                    scene_dur = (c.get("scene_end") or 0) - (c.get("scene_start") or 0)
                    dur = max(VIDEO_CLIP_MIN, min(VIDEO_CLIP_MAX, scene_dur if scene_dur > 0 else VIDEO_CLIP_DURATION))
                    start_off = float(c.get("source_offset", c.get("scene_start", 0.0)))
                    end_off = start_off + dur
                    seg_type = "video_clip"
                else:
                    dur = IMAGE_DURATION
                    start_off = 0.0
                    end_off = 0.0
                    seg_type = "image"

                selected.append({
                    "file_path": fp,
                    "file_id": c.get("file_id"),
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "duration": dur,
                    "segment_type": seg_type,
                    "scene_id": c.get("scene_id"),
                    "retrieval_strategy": c.get("granularity", "frame"),
                    "similarity_score": c.get("score"),
                    "creation_timestamp": c.get("creation_timestamp"),
                    "justification": f"Heuristic: best in time bucket {bucket_idx} (score={c.get('score', 0):.3f})",
                })
                used_paths.add(fp)
                accumulated_dur += dur
                break  # One per bucket

        # Step 6: If still under duration, fill from remaining high-score candidates
        if accumulated_dur < target_duration:
            remaining = [c for c in viable if c.get("file_path", "") not in used_paths]
            remaining.sort(key=lambda c: c.get("score", 0), reverse=True)

            for c in remaining:
                if accumulated_dur >= target_duration:
                    break
                fp = c.get("file_path", "")
                if fp in used_paths:
                    continue

                if c.get("file_type") == "video":
                    dur = VIDEO_CLIP_DURATION
                    start_off = float(c.get("source_offset", 0.0))
                    end_off = start_off + dur
                    seg_type = "video_clip"
                else:
                    dur = IMAGE_DURATION
                    start_off = 0.0
                    end_off = 0.0
                    seg_type = "image"

                selected.append({
                    "file_path": fp,
                    "file_id": c.get("file_id"),
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "duration": dur,
                    "segment_type": seg_type,
                    "scene_id": c.get("scene_id"),
                    "retrieval_strategy": c.get("granularity", "frame"),
                    "similarity_score": c.get("score"),
                    "creation_timestamp": c.get("creation_timestamp"),
                    "justification": f"Heuristic: diversity fill (score={c.get('score', 0):.3f})",
                })
                used_paths.add(fp)
                accumulated_dur += dur

        # Step 7: Final sort by creation_timestamp to ensure chronological order
        selected.sort(key=lambda s: s.get("creation_timestamp") or 0)

        # Generate narrative summary
        if timestamps:
            try:
                start_date = datetime.datetime.fromtimestamp(min(timestamps)).strftime("%b %d")
                end_date = datetime.datetime.fromtimestamp(max(timestamps)).strftime("%b %d")
                narrative = f"Chronological journey from {start_date} to {end_date}, {len(selected)} moments across {num_buckets} time periods."
            except (ValueError, OSError):
                narrative = f"Chronological sequence of {len(selected)} curated moments."
        else:
            narrative = f"Score-ranked sequence of {len(selected)} curated moments."

        return {
            "storyboard": selected,
            "narrative_arc": narrative,
        }
