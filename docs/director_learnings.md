# Director Mode — Learnings & Institutional Knowledge

> Living document capturing everything learned building the LangGraph AI Director pipeline.
> Updated: 2026-08-22

---

## 1. SigLIP2 Embedding Score Characteristics

SigLIP 2 (base-patch16-224) produces **cosine similarity scores** in a narrow, compressed range compared to traditional CLIP models. This is critical to understand because naive thresholding will discard good results.

### Empirical Score Ranges (Pin-Bhabha Corpus, ~4,008 vectors)

| Query Type | Example | Top Score | Score @ rank 10 | Score @ rank 20 |
|------------|---------|-----------|-----------------|-----------------|
| **Strong semantic match** | `"mountain landscape"` | 0.140 | 0.133 | 0.132 |
| **Grounded object query** | `"tent camping"` | 0.135 | 0.128 | — |
| **People/face query** | `"person"` | 0.119 | 0.114 | — |
| **Partially relevant** | `"people eating food at table"` | 0.113 | 0.097 | 0.068 |
| **Completely irrelevant** | `"abstract neon cyberpunk city"` | 0.077 | 0.056 | — |

### Key Takeaways

1. **Score range is ~0.05 to ~0.15.** Not 0 to 1. A score of 0.10 is a **reasonable match**, not a weak one.
2. **The gap between relevant and irrelevant is narrow**: ~0.14 (strong match) vs ~0.07 (garbage). Only ~0.07 of dynamic range.
3. **Recommended minimum score threshold: `0.08`**. Below this, candidates are noise — the vector DB returns whatever is closest but it's semantically unrelated to the query.
4. **"Good" threshold for high-confidence picks: `0.11+`**. These almost always show visually relevant content.
5. **Score degrades gracefully with query specificity**. `"mountain"` → 0.14. `"snowy mountain at sunrise with person"` → 0.12. The more specific the query, the lower the top score, but precision improves.

### Practical Implications for Director

- **Don't use `score_threshold` in Qdrant search itself** (it returns 0 results too often). Instead, **filter in Python after retrieval** using `score >= 0.08`.
- When presenting candidates to the Drafting LLM, flag scores below 0.10 as "low-confidence".
- When the heuristic curator picks candidates, prefer `score >= 0.10` but allow down to `0.08` to fill time buckets if needed.

---

## 2. LLM Prompt Engineering for Local Models (Qwen, Gemma4)

### What Works

| Pattern | Example | Why It Works |
|---------|---------|--------------|
| **Numbered JSON fields with descriptions** | `"search_queries": ["query1", "query2"]` | Local models follow explicit field naming better than prose |
| **Few-shot examples in system prompt** | Show 1-2 complete JSON examples | Dramatically reduces schema errors vs schema-only |
| **Short, imperative system prompts** | `"Output valid JSON. No markdown."` | Long narrative prompts cause local models to generate prose instead of JSON |
| **`format="json"` flag in ChatOllama** | `ChatOllama(format="json")` | Forces JSON mode at the tokenizer level |
| **`num_predict=1024`** | Limits output length | Prevents runaway generation loops |

### What Doesn't Work

| Anti-Pattern | Failure Mode |
|-------------|-------------|
| **Cinematic prose in queries** ("Wide angle drone shot of...") | SigLIP2 can't match this; LLM wastes tokens generating creative writing |
| **Asking for justifications in drafting** | Local 4B-9B models produce empty or repetitive justifications; wastes tokens |
| **Complex nested schemas** | Qwen 3.5 9B struggles with deeply nested Pydantic models; flatten where possible |
| **`<think>` tokens in Qwen 3.5** | Adds 30-50s of thinking before output; increases latency by 3-10× |
| **System prompts over ~500 tokens** | Local models start ignoring instructions at the tail |

### Temperature Recommendations

| Node | Recommended Temp | Why |
|------|-----------------|-----|
| Planner | 0.7 | Needs creative query diversity |
| Drafting | 0.3–0.5 | Should be precise about file_path and offsets |
| Editor | 0.3 | Should be deterministic in approval/rejection |

---

## 3. Chronological Metadata Flow

### The Full Pipeline (Where Timestamps Come From → Where They Go)

```
EXIF DateTimeOriginal (photos)     ──► metadata.py:extract_image_metadata()
Filesystem st_birthtime (fallback) ──► metadata.py:extract_video_metadata()
                                          │
                                          ▼
                                    scanner.py:scan_corpus()
                                    ┌─ creation_timestamp stored in:
                                    │  1. manifest.db (files.creation_timestamp column)
                                    │  2. Qdrant payload (creation_timestamp field)
                                    │
                                    ▼
                              Qdrant SearchResult
                              .creation_timestamp ✅ Available here
                                    │
                                    ▼
                              CandidateItem ← ❌ DROPPED (field missing)
                                    │
                                    ▼
                              Drafting LLM ← ❌ Never sees dates
                                    │
                                    ▼
                              TimelineSegment ← ❌ No timestamp field
```

### Fix: Add `creation_timestamp` at Every Stage

1. **`CandidateItem`** in `state.py` — add the field
2. **Retrieval node** in `nodes.py` — pass `creation_timestamp=r.creation_timestamp`
3. **Drafting prompt** — show human-readable date per candidate
4. **`TimelineSegment`** in `state.py` — add the field for verification
5. **Compiler node** — pass it through for audit

### Qdrant Temporal Filtering

Qdrant already supports `start_timestamp` and `end_timestamp` range filters in `search()`. We're not using this in the Director pipeline at all. For multi-day trips, we should:
1. Query the manifest to find the min/max `creation_timestamp` range
2. Divide into N time buckets
3. Run separate queries per bucket with range filters

---

## 4. Retrieval Strategy Learnings

### Query Decomposition: Grounded vs Cinematic

SigLIP2 is trained on **image-alt-text pairs** from the web. It understands concrete visual descriptions, not film direction language.

| ❌ Don't Generate This | ✅ Generate This Instead |
|------------------------|-------------------------|
| "Wide angle drone shot of snow-capped mountain peaks at sunrise" | "mountain", "snowy peaks", "sunrise sky" |
| "Cinematic hiking trail winding through dense pine forest" | "hiking trail", "pine forest", "path through trees" |
| "Close up slow motion of hiker boots on rocky terrain" | "hiking boots on rocks", "rocky ground" |
| "Panoramic valley view from high altitude ridge line" | "valley view", "ridge", "panoramic landscape" |
| "Time lapse of clouds rolling over jagged mountain summits" | "clouds over mountains", "mountain summit" |

**Rule of thumb**: If a query phrase wouldn't appear as alt-text on a web image, it won't match well in SigLIP2.

### Query Volume Strategy

- Queries cost ~50ms each (embedding + Qdrant search)
- 20 queries = 1 second total
- **Use many simple queries** rather than few complex ones
- De-duplication via `candidates_map` dictionary (keyed by `file_path:offset`) already handles overlap

### Dual-Granularity Insights

- **Scene-level** (`is_scene_representative=True`): Returns the mean embedding of a detected scene. Good for "what scenes exist" but scores are lower (averaging dilutes signal). Best for video-heavy corpora.
- **Frame-level** (`granularity="frame"`): Individual frame/image embeddings. Higher scores, more precise. Best for photo-heavy corpora like Pin-Bhabha.
- **Dual mode**: Fires both. De-duplication favors the higher score. Best for mixed corpora.

---

## 5. LangGraph Topology Learnings

### Current Graph Shape

```
START → planner → retrieval → drafting → editor ─┬─► compiler → END
                                    ▲             │
                                    └─────────────┘ (if not approved)
```

### Key Observations

1. **The Planner → Retrieval edge is one-shot**. If retrieval returns bad candidates, there's no feedback loop to re-plan with different queries. The only loop is `editor → drafting`, which re-sequences the *same* candidates.

2. **Alternatives share the Planner** but run separate `retrieval → drafting → editor → compiler` for each mode (scene/frame/dual). This is an optimization — planning is the slowest LLM call.

3. **Editor max_iterations=3** means at most 3 drafting attempts. In practice with fast cloud models (Gemini/Groq), the first draft usually gets approved because the Editor prompt is too permissive.

4. **The fallback heuristic** in `make_drafting_node()` only fires when the LLM produces 0 segments. It's a crash guard, not a curation strategy.

---

## 6. Performance Benchmarks

| Backend | Planner | Retrieval | Drafting | Editor | Compiler | Total |
|---------|---------|-----------|----------|--------|----------|-------|
| Gemma 4 E4B (MLX) | ~7s | ~0.3s | ~10s | ~8s | <0.1s | ~25s |
| Qwen 3.5 9B (MLX) | ~35s | ~0.3s | ~45s | ~30s | <0.1s | ~110s |
| Gemini 2.0 Flash | ~1.2s | ~0.3s | ~1.5s | ~1.0s | <0.1s | ~4s |
| Groq Llama 3.3 70B | ~0.5s | ~0.3s | ~0.6s | ~0.4s | <0.1s | ~2s |
| Mock | <0.001s | ~0.3s | <0.001s | <0.001s | <0.1s | ~0.4s |

> [!NOTE]
> Retrieval is constant (~300ms) regardless of LLM backend because it's pure embedding + Qdrant search.

---

## 7. Known Pitfalls & Gotchas

| Pitfall | What Happens | Mitigation |
|---------|-------------|------------|
| Ollama not running | `OllamaDirectorLLM` silently falls back to Mock | `ensure_ollama_running()` auto-starts it |
| Qwen 3.5 `<think>` tokens | 30-50s thinking latency before JSON | Use `/no_think` or reduce temperature |
| SSE `event_type="completed"` vs `stage="COMPLETED"` | UI closes stream prematurely | Only close on `event_type`, not `stage` |
| `num_predict=1024` too low for long storyboards | JSON output truncated mid-object | Increase to 2048 for 60s+ videos |
| Qdrant in-memory mode | All data lost on restart | Use embedded disk mode for workspaces |
| HEIC without `pillow-heif` | Silent metadata extraction failure | Already installed as dependency |
| SigLIP2 batch size > 64 on 8GB M-series | OOM crash | Default batch_size=32 in pipeline |
