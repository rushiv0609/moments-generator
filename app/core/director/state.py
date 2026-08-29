"""
State definitions and Pydantic schemas for the LangGraph Director Agent.
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


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
    score: float
    source_offset: float = 0.0
    duration_seconds: Optional[float] = None
    granularity: str = "frame"
    scene_id: Optional[int] = None
    scene_start: Optional[float] = None
    scene_end: Optional[float] = None
    matched_query: str = ""
    creation_timestamp: Optional[float] = None


class DraftingSegmentChoice(BaseModel):
    """An individual segment selection made by the drafting LLM."""
    file_path: str = Field(description="File path from candidate list")
    start_offset: float = Field(default=0.0, description="Start timestamp in seconds")
    end_offset: float = Field(default=0.0, description="End timestamp in seconds")
    duration: float = Field(default=3.0, description="Duration in seconds (e.g. 2.0 to 5.0s)")
    segment_type: str = Field(default="image", description="'image' or 'video_clip'")
    scene_id: Optional[int] = Field(default=None, description="Scene ID if video candidate")
    retrieval_strategy: str = Field(default="frame", description="'scene' or 'frame'")
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


class DirectorState(TypedDict):
    """State graph working memory passed between LangGraph nodes."""
    # User Inputs & Config
    user_prompt: str
    target_duration: int
    retrieval_mode: str  # 'scene' | 'frame' | 'dual'

    # Internal Working Memory
    search_queries: List[str]
    retrieved_candidates: List[Dict[str, Any]]

    # Storyboard Drafting & Critique
    storyboard: List[Dict[str, Any]]
    editor_feedback: List[str]
    narrative_arc: str

    # State Flow Controls
    iteration_count: int
    approved: bool

    # Metadata & Provenance
    llm_model: str
    run_label: str
    error: Optional[str]
    agent_telemetry: List[Dict[str, Any]]
