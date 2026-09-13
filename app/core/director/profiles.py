"""
Creative Profiles configuration for LangGraph Director Agent.
Provides diverse curation styles (Cinematic, Personal Moments, Balanced Journey).
"""

from typing import Dict, Any

CREATIVE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cinematic": {
        "id": "cinematic",
        "name": "Cinematic Epic",
        "rank_weights": {
            "cosine": 0.40,
            "nima": 0.30,
            "sharpness": 0.05,
            "contrast": 0.035,
            "colorfulness": 0.015,
            "epic_sim": 0.20,
            "technical": 0.10,
        },
        "segment_duration_bias": "longer",  # Prefer 3.0-5.0s photos, 4.0-8.0s video scenes
        "min_segments_factor": 0.8,
        "max_clips_per_file": 2,
        "drafter_emphasis": (
            "EMPHASIS: Maximize grand visual impact, sweeping panoramic vistas, dramatic lighting, "
            "and golden hour atmosphere. Give epic scenes breathing room with slightly longer, steady durations."
        ),
    },
    "personal": {
        "id": "personal",
        "name": "Personal Moments & Faces",
        "rank_weights": {
            "cosine": 0.50,
            "nima": 0.15,
            "sharpness": 0.125,
            "contrast": 0.0875,
            "colorfulness": 0.0375,
            "epic_sim": 0.10,
            "technical": 0.25,
        },
        "segment_duration_bias": "shorter",  # Prefer 2.0-3.0s photos, 2.5-4.0s video clips
        "min_segments_factor": 1.2,
        "max_clips_per_file": 2,
        "drafter_emphasis": (
            "EMPHASIS: Prioritize human connections, smiling faces, group bonding, candid interactions, "
            "and emotional warmth. Use brisk, energetic cuts to maintain momentum."
        ),
    },
    "journey": {
        "id": "journey",
        "name": "Balanced Journey",
        "rank_weights": {
            "cosine": 0.45,
            "nima": 0.45,
            "technical": 0.10,
            "sharpness": 0.05,
            "contrast": 0.035,
            "colorfulness": 0.015,
            "epic_sim": 0.15,
        },
        "segment_duration_bias": "mixed",
        "min_segments_factor": 1.0,
        "max_clips_per_file": 2,
        "drafter_emphasis": (
            "EMPHASIS: Deliver a complete chronological journey from start to finish. "
            "Maintain a dynamic rhythm by interleaving landscape establishing shots with action video clips and candid photos."
        ),
    },
}


def get_creative_profile(profile_id: str) -> Dict[str, Any]:
    """Retrieve creative profile by ID with fallback to 'journey'."""
    pid = profile_id.lower()
    if pid in CREATIVE_PROFILES:
        return CREATIVE_PROFILES[pid]
    # Map legacy names if passed
    if "alt_a" in pid or "scene" in pid:
        return CREATIVE_PROFILES["cinematic"]
    elif "alt_b" in pid or "frame" in pid:
        return CREATIVE_PROFILES["personal"]
    return CREATIVE_PROFILES["journey"]
