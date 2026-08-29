# LangGraph Director Agent: Implementation Guide

**Project:** Local AI Moments Generator
**Phase:** 9 (Curator Upgrade)

This document details the architectural upgrade from the static time-bucketing curator to the **Agentic LangGraph Director**. It replaces rigid heuristics with a reasoning LLM (e.g., Qwen 3.5 9B or Gemma 4) to iteratively plan, retrieve, and sequence video clips.

---

## 1. Architectural Overview

The LangGraph Director is a state machine consisting of specialized Python functions (Nodes) that modify a shared data structure (State). The execution flow is governed by routing logic (Edges), allowing the agent to evaluate its own work and loop back if the drafted video fails quality constraints.

### 1.1 The State Machine Diagram

```text
                             [ START ]
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │       PLANNER NODE        │
                   │ LLM breaks prompt into    │
                   │ temporal sub-queries      │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │      RETRIEVAL NODE       │
                   │ Executes Qdrant searches  │
                   │ for all sub-queries       │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐◄──────┐
                   │       DRAFTING NODE       │       │
                   │ LLM selects segments to   │       │
                   │ fit target duration       │       │
                   └─────────────┬─────────────┘       │
                                 │                     │
                                 ▼                     │
                   ┌───────────────────────────┐       │
                   │       EDITOR NODE         │       │
                   │ LLM critiques pacing,     │       │
                   │ flow, and duplicates      │       │
                   └─────────────┬─────────────┘       │
                                 │                     │
                                 ▼                     │
                      { Conditional Edge }             │
                     Does draft pass checks?           │
                     /                   \             │
               [ YES ]                  [ NO ] ────────┘
                 /                        (Provide feedback,
                ▼                          loop to redraft)
  ┌───────────────────────────┐
  │      COMPILER NODE        │
  │ Generates final EDL JSON  │
  │ and saves to Manifest DB  │
  └─────────────┬─────────────┘
                │
                ▼
             [ END ]
```

---

## 2. State Schema Definition

The `State` is the single source of truth passed between nodes. It uses Pydantic to ensure strict typing.

```python
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
import operator

class TimelineSegment(BaseModel):
    file_path: str
    start_offset: float
    duration: float
    justification: str

class DirectorState(TypedDict):
    # Inputs
    user_prompt: str
    target_duration: int
    
    # Internal Working Memory
    search_queries: List[str]
    retrieved_candidates: List[dict]
    
    # Agent Outputs
    storyboard: List[TimelineSegment]
    editor_feedback: List[str]
    
    # Loop Control
    iteration_count: Annotated[int, operator.add]
    approved: bool
```

---

## 3. Node Implementations

### 3.1 Planner Node
**Role:** Breaks the user's natural language prompt into specific visual search queries to ensure a diverse narrative arc.

*   **Input:** `"Make a video of my beach trip with friends"`
*   **LLM Task:** Return a JSON list of 3-5 visual concepts to search for.
*   **Output to State (`search_queries`):** `["arriving at the beach", "playing in the sand", "group selfie", "sunset over the ocean"]`

### 3.2 Retrieval Node (Tool Node)
**Role:** The only node that interacts with the Qdrant Vector DB.
*   **Action:** For every query in `search_queries`, use SigLIP 2 to embed the text, run a Qdrant Top-K search, and append the results to `retrieved_candidates`.
*   **Note:** This node does not use the LLM; it is pure deterministic Python logic.

### 3.3 Drafting Node
**Role:** The core "Editor". It looks at the pool of `retrieved_candidates` and pieces together a timeline.

*   **LLM Task:** Select specific items from the candidates list. Assign a duration (e.g., 3s for images, dynamic for video). Ensure the total duration equals `target_duration`.
*   **Output to State (`storyboard`):** A chronological list of `TimelineSegment` objects.
*   **Feedback Loop:** If `editor_feedback` exists in the state from a previous loop, the LLM must read it and adjust the draft accordingly.

### 3.4 Editor Node (Critique)
**Role:** The Quality Assurance checker.

*   **LLM Task:** Review the `storyboard` against specific rules:
    1.  Is the total duration within 5% of the target?
    2.  Are there duplicate shots (same file path used twice)?
    3.  Is the progression logical?
*   **Output to State:** Sets `approved = True` if perfect. If flawed, sets `approved = False` and populates `editor_feedback` with specific instructions (e.g., *"The video is 10 seconds too long. Cut the second group selfie."*).

### 3.5 Compiler Node
**Role:** Finalizes the data for FFmpeg.
*   **Action:** Converts the approved `storyboard` into the `timeline_segments` format required by the SQLite Manifest DB, marking the LangGraph execution as complete.

---

## 4. Integration with the Current Pipeline

To integrate this into your existing codebase:

1.  **Replace `app/core/curator.py`:** Delete the static time-bucketing logic.
2.  **Initialize LangGraph:** Define the `StateGraph(DirectorState)`.
3.  **Add Nodes:** Add the Python functions defined above (`graph.add_node("planner", planner_node)`, etc.).
4.  **Add Edges:** 
    *   `graph.add_edge("planner", "retrieval")`
    *   `graph.add_edge("retrieval", "drafting")`
    *   `graph.add_edge("drafting", "editor")`
5.  **Conditional Logic:**
    ```python
    def check_approval(state: DirectorState):
        if state["iteration_count"] >= 3: # Fallback to prevent infinite loops
            return "compiler"
        return "compiler" if state["approved"] else "drafting"
        
    graph.add_conditional_edges("editor", check_approval)
    ```
6.  **Execute:** When the FastAPI endpoint `/api/v1/jobs/generate` is called, invoke the graph: `graph.invoke({"user_prompt": prompt, "target_duration": duration, "iteration_count": 0})`.

---

## 5. Next Steps for Implementation

As per the project plan, you should move sequentially:
1.  **Finish Milestone 7:** Validate the Qdrant connection and batch upserts.
2.  **Finish Milestone 8:** Verify the end-to-end ingestion pipeline writes to both SQLite and Qdrant correctly.
3.  **Begin Milestone 9 (This Document):** Install `langgraph` and `langchain-ollama`, serve Qwen 3.5 9B locally, and build the `Planner Node` as the first test of LLM structured output.
