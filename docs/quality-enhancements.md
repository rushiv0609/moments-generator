# Quality Enhancements

Each enhancement is tied to a specific problem from [quality-observations.md](./quality-observations.md). Nothing here is cosmetic — every change exists because we observed a concrete failure.

---

## Enhancement 1: Multi-Signal Scoring for Candidate Ranking

**Addresses**: Problem 1 (missing epic frames)

### What We're Adding

Two new scoring signals layered on top of the existing cosine similarity:

**Signal A — NIMA Aesthetic Score (index time)**

NIMA is a lightweight neural network (~14MB, MobileNetV2 backbone) trained on 250K human-rated photographs. It outputs a score from 1–10 representing aesthetic quality: composition, focus, lighting, color harmony.

- Runs on the 224px thumbnail already decoded for SigLIP2 — no additional decode cost
- ~3ms per image on Apple Silicon MPS
- Stored in Qdrant payload as `scores.nima_aesthetic` (normalized to 0.0–1.0)

Why NIMA and not just OpenCV heuristics? OpenCV Laplacian variance measures sharpness, but a sharp photo of a trash can scores high. NIMA captures the holistic "does this look like a good photograph" signal that low-level metrics miss. We still compute sharpness/contrast/colorfulness as supplementary signals — they're fast and useful for filtering obvious throwaway frames.

**Signal B — Epic Reference Similarity (query time, optional)**

User uploads 3–5 photos they consider "epic" for their context. We:
1. Embed each reference image through SigLIP2
2. Compute the mean reference vector
3. For each candidate, compute `cosine(candidate_vector, epic_reference_vector)`

This is image-to-image comparison, which SigLIP2 is good at (unlike aesthetic assessment). It answers: "does this candidate look like the kind of shot the user values?"

**Overfitting guard**: Epic similarity is weighted at 15–20% in the composite rank, not 60%. Additionally, a visual diversity penalty applies: if candidate B's embedding is >0.85 cosine similar to an already-selected candidate A, B's composite score is penalized. This prevents the video from being 20 frames that all look alike.

### Composite Ranking Formula

```
candidate_rank = (
    0.50 * cosine_similarity     +  # Relevance to user query
    0.20 * nima_aesthetic         +  # Photo quality (NIMA)
    0.15 * epic_reference_sim     +  # User's personal epic preference (if provided)
    0.15 * technical_quality         # Sharpness + contrast + colorfulness average
)
```

If no epic references provided, the weights redistribute:
```
candidate_rank = (
    0.55 * cosine_similarity     +
    0.25 * nima_aesthetic         +
    0.20 * technical_quality
)
```

### Index-Time Score Storage in Qdrant

All quality signals stored under a nested `scores` key in the Qdrant payload:

```json
{
  "file_path": "/path/to/IMG_7071.HEIC",
  "file_type": "image",
  "scores": {
    "nima_aesthetic": 0.82,
    "sharpness": 0.91,
    "colorfulness": 0.68,
    "contrast": 0.77,
    "technical_composite": 0.79
  }
}
```

Adding a new score in the future = add one key. No schema changes anywhere.

---

## Enhancement 2: Duration Enforcement

**Addresses**: Problem 2 (inconsistent video length)

### Changes

1. **Pre-compute segment count guidance** before the LLM drafts:
   ```
   min_segments = max(5, target_duration // 4)
   max_segments = target_duration // 2
   ```
   For a 60s target: at least 15 segments, at most 30.

2. **Inject explicit count into drafter prompt**:
   ```
   "You MUST produce between {min_segments} and {max_segments} segments."
   ```

3. **Post-LLM duration enforcement**:
   - If total < target: auto-fill from remaining candidates sorted by composite rank until duration ≥ target
   - If total > target * 1.25: trim lowest-scored segments from the tail until within range

4. **Tighten editor tolerance**:
   - Old: ±25% (45s–75s for 60s target)
   - New: target to target+25% (60s–75s for 60s target). Never shorter than target.

5. **Relaxed per-segment caps** (the old 3.0s/5.0s caps were bug workarounds, no longer needed):
   - Images: 1.5s – 5.0s
   - Video clips: 2.0s – 10.0s

---

## Enhancement 3: Timeline Integrity

**Addresses**: Problem 3 (chronological breaks, editor degradation, single-dimension scoring)

### 3a: Forced Chronological Sort

After the LLM outputs a storyboard, sort all segments by `creation_timestamp` regardless of the LLM's ordering. The LLM picks *which* moments; we enforce *when* they appear. This is deterministic and eliminates the re-ordering failure mode entirely.

### 3b: Best-Draft Watermarking

Track `best_storyboard` and `best_composite_score` in state across iterations:

```
After each editor pass:
  if current_composite >= best_composite:
    best_storyboard = current_storyboard  (new best)
    best_composite = current_composite
  else:
    revert storyboard to best_storyboard  (iteration degraded, discard it)
```

This ensures the output is always the best draft seen across all iterations, not just the last one.

### 3c: Multi-Dimension Composite Score

Replace the single `pacing_score` with a deterministic composite computed from measurable signals:

| Dimension | Weight | Computation |
|:---|:---:|:---|
| Duration accuracy | 0.25 | `1.0 - max(0, target - actual) / target` (penalize under-target only) |
| Pacing rhythm | 0.20 | Standard deviation of segment durations (higher variety = better) |
| Visual quality | 0.25 | Mean `nima_aesthetic` of all selected segments |
| Query relevance | 0.20 | Mean `similarity_score` of all selected segments |
| Diversity | 0.10 | Temporal spread + media type mix |

The LLM still provides qualitative critique text, but the score that drives best-draft selection is computed, not hallucinated.

---

## Enhancement 4: Rendering Fidelity

**Addresses**: Problem 4 (resolution loss, speed artifacts, mechanical Ken Burns)

### 4a: Resolution Preservation

- Detect source resolution via `cv2.VideoCapture` properties or `ffprobe`
- Match output resolution to the dominant source resolution in the storyboard
- If majority of sources are 4K: render at 4K with bitrate 15000k
- If majority are 1080p: render at 1080p with bitrate 8000k
- Aspect ratio handling: pad to fit (letterbox) instead of crop when source and target ratios differ. Cropping cuts content; letterbox preserves it.

### 4b: Speed Clamping

When the source clip at the selected offset is shorter than the requested duration, the renderer currently truncates. With this change:

- Compute implied speed factor: `speed = source_available / requested_duration`
- If speed > 1.25x, extend the requested duration instead of speeding up
- Maximum speed-up: 1.25x (25% faster). Beyond that, the motion looks unnatural.
- Log a warning when speed clamping adjusts the planned duration

### 4c: Content-Aware Ken Burns

Replace the mechanical `index % 4` cycling with image analysis:

**Layer 1 — Aspect ratio heuristic**:
- Ultra-wide source (≥16:9) → prefer horizontal pans (`pan_left` / `pan_right`)
- Tall/portrait source → prefer vertical motion or zoom
- Square or 4:3 → prefer zoom

**Layer 2 — Saliency detection** (OpenCV built-in, no extra dependency):
- `cv2.saliency.StaticSaliencySpectralResidual` finds the attention center
- If saliency center is in the left third → `pan_left` (move toward subject)
- If in the right third → `pan_right`
- If centered → `zoom_in`

**Layer 3 — Face detection** (OpenCV DNN, optional):
- If face(s) detected → zoom toward face region
- If no face → fall back to saliency

**Layer 4 — Landscape detection** (from SigLIP2 embedding, free):
- At render time, compute cosine against "wide landscape scenic view" and "close up portrait"
- Landscape-like → `zoom_out` (reveal the grandeur)
- Portrait-like → `zoom_in` (focus on subject)

**Smoother motion**:
- Reduce zoom delta from 0.15 (15%) to 0.08 (8%)
- Replace linear interpolation with Hermite smoothstep easing: `z = 1.0 + delta * (3t² - 2t³)` where `t = on/total_frames`
- This creates gentle acceleration and deceleration instead of abrupt start/stop

---

## Enhancement 5: Creative Alternative Timelines

**Addresses**: The current alternative split (scene-only, frame-only, dual, heuristic) is a technical split, not a creative one. Users don't care about retrieval granularity — they want different creative perspectives.

### New Alternative Structure

All alternatives use dual retrieval. They differ in creative emphasis:

| Alternative | What It Prioritizes | How |
|:---|:---|:---|
| **Cinematic** | Visual impact, dramatic moments | Weights `nima_aesthetic` and `epic_reference` higher. Prefers longer segments (4–6s). Fewer cuts. |
| **Personal** | Human moments, faces, interactions | Boosts candidates with face detection signals. More segments, tighter pacing (2–3s). |
| **Journey** | Chronological coverage, balanced mix | Equal weights across all factors. Ensures every time bucket is represented. |
| **Heuristic** | Deterministic baseline | No LLM. Pure algorithm: chronological sort, score-weighted bucket sampling. |

Each alternative gets a `creative_profile` configuration that adjusts:
1. Composite ranking weights
2. Drafter system prompt emphasis
3. Min/max segment count preferences
4. Duration per segment tendencies
