"""
State definitions and Pydantic schemas for the LangGraph Director Agent.
Supports multi-signal scoring, creative profiles, and best-draft watermarking.
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QualityScores(TypedDict, total=False):
    """Extensible quality metrics dictionary."""
    nima_aesthetic: float
    sharpness: float
    colorfulness: float
    contrast: float
    technical_composite: float
    epic_composite: float


class CreativeConfig(TypedDict, total=False):
    """Extensible creative configuration for director curation profiles."""
    profile_id: str
    rank_weights: Dict[str, float]
    segment_duration_bias: str
    min_segments_factor: float
    max_clips_per_file: int
    drafter_emphasis: str


class TimelineSegment(BaseModel):
    """A curated segment within the final video timeline."""
    file_path: str = Field(description="Absolute path to media file")
    file_id: Optional[int] = Field(default=None, description="Database ID of file in manifest")
    start_offset: float = Field(default=0.0, description="Start timestamp in seconds")
    end_offset: float = Field(default=0.0, description="End timestamp in seconds")
    duration: float = Field(default=3.0, description="Duration of this segment in seconds")
    segment_type: str = Field(default="image", description="'image' or 'video_clip'")
    scene_id: Optional[int] = Field(default=None, description="Scene ID if extracted from video")
    retrieval_strategy: str = Field(default="frame", description="'scene' or 'frame'")
    similarity_score: Optional[float] = Field(default=None, description="Embedding match score")
    composite_rank: Optional[float] = Field(default=None, description="Multi-signal candidate ranking score")
    scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores (nima, sharpness, etc.)")
    justification: str = Field(default="", description="Director reasoning for choosing this moment")
    creation_timestamp: Optional[float] = Field(default=None, description="Original capture timestamp (Unix epoch) from EXIF/metadata")


class PlannerOutput(BaseModel):
    """Structured output from the Planner node."""
    search_queries: List[str] = Field(
        description="8 to 15 short, concrete visual search queries for vector DB",
        min_length=1,
        max_length=20,
    )
    mood_or_narrative: str = Field(
        default="",
        description="Cinematic tone, mood, or storyline progression",
    )
    target_duration_seconds: int = Field(
        default=30,
        description="Target duration of the curated montage in seconds",
    )


class CandidateItem(BaseModel):
    """A media candidate retrieved from the vector database."""
    file_path: str
    file_id: Optional[int] = None
    file_type: str  # 'image' | 'video'
    score: float  # Cosine similarity score
    composite_rank: Optional[float] = None  # Multi-signal ranking score
    scores: Dict[str, float] = Field(default_factory=dict)
    source_offset: float = 0.0
    duration_seconds: Optional[float] = None
    granularity: str = "frame"
    scene_id: Optional[int] = None
    scene_start: Optional[float] = None
    scene_end: Optional[float] = None
    matched_query: str = ""
    creation_timestamp: Optional[float] = None
    vector: Optional[List[float]] = None


class DraftingSegmentChoice(BaseModel):
    """An individual segment selection made by the drafting LLM."""
    file_path: str = Field(description="File path from candidate list")
    start_offset: float = Field(default=0.0, description="Start timestamp in seconds")
    end_offset: float = Field(default=0.0, description="End timestamp in seconds")
    duration: float = Field(default=3.0, description="Duration in seconds (e.g. 2.0 to 6.0s)")
    segment_type: str = Field(default="image", description="'image' or 'video_clip'")
    scene_id: Optional[int] = Field(default=None, description="Scene ID if video candidate")
    retrieval_strategy: str = Field(default="frame", description="'scene' or 'frame'")
    similarity_score: Optional[float] = Field(default=None, description="Embedding similarity score")
    composite_rank: Optional[float] = Field(default=None, description="Multi-signal candidate ranking")
    scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores")
    justification: str = Field(default="", description="Reason for selection and sequencing")
    creation_timestamp: Optional[float] = Field(default=None, description="Original capture timestamp")


class DraftingOutput(BaseModel):
    """Structured output from the Drafting node."""
    storyboard: List[DraftingSegmentChoice] = Field(
        default_factory=list,
        description="Ordered sequence of media segments forming the video",
    )
    narrative_arc: str = Field(
        default="",
        description="Explanation of how the sequence creates a story or mood",
    )


class EditorOutput(BaseModel):
    """Structured output from the Editor node."""
    approved: bool = Field(description="True if storyboard meets quality, pacing, and duration goals")
    feedback: str = Field(description="Detailed constructive critique of pacing, duplicates, or flow")
    pacing_score: float = Field(default=7.0, description="Score from 1.0 to 10.0 on rhythm and variety")
    suggested_modifications: List[str] = Field(
        default_factory=list,
        description="Specific adjustment recommendations if rejected",
    )
    composite_score: float = Field(default=7.0, description="Objective multi-factor composite evaluation score")


class DirectorState(TypedDict, total=False):
    """State graph working memory passed between Director Agent nodes."""
    # User Inputs & Configuration
    user_prompt: str
    target_duration: int
    retrieval_mode: str  # 'scene' | 'frame' | 'dual'
    creative_profile: str  # 'cinematic' | 'personal' | 'journey'
    creative_config: Dict[str, Any]
    epic_reference_vector: Optional[List[float]]
    scoring_signals: Optional[List[str]]
    scoring_weights: Optional[Dict[str, float]]

    # Internal Working Memory
    search_queries: List[str]
    retrieved_candidates: List[Dict[str, Any]]

    # Storyboard Drafting & Critique
    storyboard: List[Dict[str, Any]]
    editor_feedback: List[str]
    narrative_arc: str

    # State Flow Controls & Best-Draft Watermarking
    iteration_count: int
    approved: bool
    best_storyboard: List[Dict[str, Any]]
    best_composite_score: float

    # Metadata & Provenance
    llm_model: str
    run_label: str
    error: Optional[str]
    agent_telemetry: List[Dict[str, Any]]
