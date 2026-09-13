# Technical Architecture

Design decisions for the pipeline framework, state management, scoring storage, and extensibility.

---

## 1. Pipeline Framework: Strategy Pattern

### Problem

We currently use LangGraph for the director state machine. It works and is stable. However:
- We use ~5% of LangGraph's capabilities (5 nodes, 1 conditional edge)
- The `TypedDict` state is awkward for extensibility
- We want to experiment with a Pydantic-based state machine for stronger typing

### Decision

Run both frameworks behind a strategy pattern. User selects which engine to use.

```
┌─────────────────────────────────────────────┐
│                 API Layer                    │
│         POST /api/v1/director/generate      │
│              engine=langgraph|pydantic       │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  DirectorEngine   │  (Abstract interface)
         │  .run(prompt, ...) │
         └─────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼───────┐   ┌────────▼────────┐
│ LangGraphEngine│   │ PydanticEngine  │
│  (existing)    │   │  (new)          │
│  TypedDict     │   │  Pydantic model │
│  StateGraph    │   │  custom runner  │
└───────┬────────┘   └────────┬────────┘
        │                     │
        └─────────┬───────────┘
                  │
         ┌────────▼────────┐
         │  Shared Nodes   │  (Same functions for both)
         │  planner_node() │
         │  retrieval_node()│
         │  drafting_node() │
         │  editor_node()  │
         │  compiler_node()│
         └─────────────────┘
```

**Key constraint**: Node functions remain framework-agnostic. They take state (dict or Pydantic model), return partial updates. The engine handles routing and state merging.

### LangGraph Engine (existing, stable)

- `DirectorState` gets `total=False` nested TypedDicts for extensibility:

```python
class QualityScores(TypedDict, total=False):
    nima_aesthetic: float
    sharpness: float
    colorfulness: float
    contrast: float
    technical_composite: float

class DirectorState(TypedDict):
    user_prompt: str
    target_duration: int
    # ... core fields ...
    quality_scores: QualityScores
    best_storyboard: List[Dict[str, Any]]
    best_composite_score: float
```

### Pydantic Engine (new, experimental)

```python
class QualityScores(BaseModel):
    nima_aesthetic: float = 0.0
    sharpness: float = 0.0
    colorfulness: float = 0.0
    contrast: float = 0.0
    technical_composite: float = 0.0

class DirectorState(BaseModel):
    user_prompt: str
    target_duration: int = 30
    quality_scores: QualityScores = QualityScores()
    # Full Pydantic validation, serialization, defaults
```

The Pydantic engine is a simple while-loop state machine:

```python
class PydanticDirectorEngine:
    def run(self, state: DirectorState) -> DirectorState:
        current = "planner"
        while current != "END":
            updates = self.nodes[current](state.model_dump())
            state = state.model_copy(update=updates)
            current = self.route(current, state)
        return state
```

No external dependency beyond Pydantic (which we already have).

---

## 2. Scoring Storage in Qdrant

### Payload Structure

All quality scores live under a nested `scores` key:

```json
{
  "file_path": "/path/to/photo.HEIC",
  "file_id": 42,
  "file_type": "image",
  "frame_index": 0,
  "source_offset": 0.0,
  "creation_timestamp": 1720358423.0,
  "granularity": "frame",
  "scores": {
    "nima_aesthetic": 0.82,
    "sharpness": 0.91,
    "colorfulness": 0.68,
    "contrast": 0.77,
    "technical_composite": 0.79
  }
}
```

### Implementation

`VectorPoint` gets a single `scores: Dict[str, float]` field:

```python
@dataclass
class VectorPoint:
    vector: Union[np.ndarray, List[float]]
    file_path: str
    scores: Dict[str, float] = field(default_factory=dict)
    # ... existing fields ...

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "file_path": self.file_path,
            # ... existing fields ...
            "scores": self.scores,
        }
        return payload
```

`SearchResult` mirrors this:

```python
@dataclass
class SearchResult:
    score: float  # cosine similarity (from Qdrant search)
    scores: Dict[str, float] = field(default_factory=dict)  # quality scores from payload
    # ... existing fields ...
```

### Filtering

Qdrant supports nested field filtering:

```python
# Only return candidates with aesthetic score >= 0.5
models.FieldCondition(
    key="scores.nima_aesthetic",
    range=models.Range(gte=0.5),
)
```

---

## 3. NIMA Integration

### Model Loading

NIMA loads alongside SigLIP2 during embedder initialization:

```
App startup
  ├── Load SigLIP2 (MLX, ~350MB)
  └── Load NIMA (MobileNetV2, ~14MB, PyTorch MPS)
```

NIMA runs on the MPS backend (same GPU, different model). It's small enough that it doesn't compete with SigLIP2 for memory.

### Scoring Pipeline Flow

```
Frame decoded (224x224 RGB)
  │
  ├── SigLIP2 embed → 768-dim vector → Qdrant vector
  │
  ├── NIMA predict → aesthetic score 1-10 → normalize to 0.0-1.0
  │
  ├── OpenCV sharpness → Laplacian variance → normalize
  ├── OpenCV colorfulness → opponent color-space σ → normalize
  ├── OpenCV contrast → histogram spread → normalize
  │
  └── All scores → Qdrant payload under "scores" key
```

The OpenCV metrics add ~0.5ms per frame. NIMA adds ~3ms per frame. Total overhead vs current pipeline: <5ms per frame on a pipeline that takes ~15ms per frame for SigLIP2 embedding.

---

## 4. Epic Reference Comparison

### Flow

```
User provides reference images (optional)
  │
  ├── Embed each reference through SigLIP2 → N × 768-dim vectors
  ├── Compute mean → epic_reference_vector (768-dim)
  │
  └── At retrieval time:
      For each candidate:
        epic_sim = cosine(candidate_vector, epic_reference_vector)
        
        Apply diversity penalty:
          For each already-selected candidate A:
            if cosine(candidate, A) > 0.85:
              epic_sim *= 0.5  (halve the bonus for similar-looking shots)
```

### Overfitting Prevention

The epic similarity is one of four factors in the composite rank:

```
candidate_rank = 0.50 * query_cosine + 0.20 * nima + 0.15 * epic_sim + 0.15 * technical
```

At 15% weight, epic similarity acts as a tiebreaker between otherwise-equal candidates, not as the dominant signal. Combined with the 0.85 cosine diversity penalty, the system will prefer epic-looking shots but won't fill the video with 20 near-identical frames.

### Toggle

This feature is off by default. Activated when user provides reference images via the UI. The UI will have an "Epic References" panel where users can drag-drop photos they consider representative of what they want.

---

## 5. Creative Profiles for Alternative Timelines

### Current Design (being replaced)

```
Alternative A: scene-only retrieval    ← technical split
Alternative B: frame-only retrieval   ← technical split
Alternative C: dual retrieval         ← technical split
Alternative D: heuristic (no LLM)    ← deterministic baseline
```

### New Design

```
Alternative A: "Cinematic" (dual + epic-weighted)    ← creative split
Alternative B: "Personal" (dual + people-weighted)   ← creative split
Alternative C: "Journey" (dual + balanced)            ← creative split
Alternative D: Heuristic (no LLM)                    ← deterministic baseline
```

Each profile is a configuration dict:

```python
CREATIVE_PROFILES = {
    "cinematic": {
        "name": "Cinematic Epic",
        "rank_weights": {"cosine": 0.40, "nima": 0.30, "epic": 0.20, "technical": 0.10},
        "segment_duration_bias": "longer",   # prefer 4-6s segments
        "min_segments_factor": 0.7,          # fewer, longer segments
        "drafter_emphasis": "dramatic visual impact, sweeping landscapes, golden hour",
    },
    "personal": {
        "name": "Personal Moments",
        "rank_weights": {"cosine": 0.50, "nima": 0.15, "epic": 0.10, "technical": 0.25},
        "segment_duration_bias": "shorter",  # prefer 2-3s segments
        "min_segments_factor": 1.3,          # more segments, tighter cuts
        "drafter_emphasis": "people, faces, group interactions, candid moments",
    },
    "journey": {
        "name": "Balanced Journey",
        "rank_weights": {"cosine": 0.50, "nima": 0.20, "epic": 0.15, "technical": 0.15},
        "segment_duration_bias": "mixed",
        "min_segments_factor": 1.0,
        "drafter_emphasis": "chronological coverage, visual variety, mix of photos and video",
    },
}
```

The drafter system prompt is parameterized by the profile's `drafter_emphasis` field. Retrieval weights adjust candidate ranking. Segment count and duration preferences shape the storyboard structure.

---

## 6. File Map

Summary of files affected and new files:

```
app/
├── core/
│   ├── aesthetic.py           [NEW]   NIMA loader + OpenCV quality metrics
│   ├── pipeline.py            [MODIFY] Add aesthetic scoring to index loop
│   ├── embedder.py            [NO CHANGE]
│   ├── compiler/
│   │   ├── renderer.py        [MODIFY] Resolution preservation, speed clamping
│   │   └── effects.py         [MODIFY] Smart Ken Burns, smoother easing
│   └── director/
│       ├── state.py           [MODIFY] Add QualityScores, best-draft fields
│       ├── nodes.py           [MODIFY] Composite ranking, duration enforcement,
│       │                               chronological sort, best-draft tracking
│       ├── graph.py           [MODIFY] Strategy pattern, creative profiles
│       ├── engine.py          [NEW]    Abstract DirectorEngine + PydanticEngine
│       ├── profiles.py        [NEW]    Creative profile configurations
│       ├── heuristic.py       [MODIFY] Use composite scoring
│       └── llm.py             [NO CHANGE]
├── db/
│   └── qdrant.py              [MODIFY] Nested scores payload, SearchResult.scores
└── api/
    ├── routes.py              [MODIFY] engine param, epic references endpoint
    └── schemas.py             [MODIFY] New API schemas

tests/
├── test_aesthetic.py          [NEW]
├── test_engine.py             [NEW]
└── test_director.py           [MODIFY] Updated for composite scoring
```
