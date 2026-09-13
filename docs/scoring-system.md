# Scoring System Design

How frames are scored, ranked, and selected throughout the pipeline.

---

## Overview

The pipeline uses three layers of scoring at different stages:

```
Index Time (per frame)           Query Time (per candidate)        Editor Time (per storyboard)
─────────────────────────        ──────────────────────────        ────────────────────────────
NIMA aesthetic score             Cosine similarity to query        Duration accuracy
Sharpness (Laplacian)            Epic reference similarity         Pacing rhythm
Colorfulness (opponent σ)        Composite candidate rank          Visual quality mean
Contrast (histogram spread)      Diversity penalty                 Query relevance mean
Technical composite              ───▶ Ranked candidate list        Diversity spread
───▶ Stored in Qdrant payload                                     ───▶ Composite storyboard score
```

---

## Layer 1: Index-Time Quality Scores

Computed once per frame during ingestion. Stored in Qdrant payload. Never recomputed.

### NIMA Aesthetic Score

- **What**: Neural network trained on 250K human-rated photos (AVA dataset)
- **Input**: 224x224 RGB numpy array (same as SigLIP2 input)
- **Output**: Score 1–10, normalized to 0.0–1.0
- **What it captures**: Composition, lighting, focus, color harmony — the overall "does this look like a good photograph" signal
- **What it misses**: Context relevance (a well-composed photo of food scores high even if we're making a trek video)
- **Model**: MobileNetV2 backbone, ~14MB, runs on MPS

### Sharpness (Laplacian Variance)

```python
gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = laplacian.var()
# Normalize: empirically, values range 0–2000. Clamp and divide by 2000.
```

Separates crisp focused shots from motion blur or out-of-focus frames. A key filter for video frames — many extracted frames are mid-motion and blurry.

### Colorfulness (Hasler & Süsstrunk)

```python
R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
rg = R.astype(float) - G.astype(float)
yb = 0.5 * (R.astype(float) + G.astype(float)) - B.astype(float)
colorfulness = (rg.std() + yb.std()) + 0.3 * (rg.mean()**2 + yb.mean()**2)**0.5
# Normalize: empirically, values range 0–150. Clamp and divide by 150.
```

Separates vivid scenes (sunsets, green valleys, blue lakes) from dull overcast shots. Useful for "epic" landscapes where color intensity correlates with visual impact.

### Contrast (Histogram Spread)

```python
gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
p5, p95 = np.percentile(gray, [5, 95])
contrast = (p95 - p5) / 255.0
```

Measures dynamic range. High contrast = bright highlights and dark shadows (dramatic lighting, sunset silhouettes). Low contrast = flat, washed out.

### Technical Composite

```python
technical = 0.45 * sharpness + 0.30 * colorfulness + 0.25 * contrast
```

Sharpness weighted highest because blur is the strongest quality-killer.

---

## Layer 2: Query-Time Candidate Ranking

Computed at retrieval time. Combines the stored quality scores with live similarity search results.

### Cosine Similarity

The Qdrant search result score. Measures how well the frame's SigLIP2 embedding matches the text query embedding.

- Range for SigLIP2: typically -0.05 to +0.25
- Threshold: candidates below 0.08 are treated as noise

### Epic Reference Similarity (optional)

If user has provided reference images:
```python
epic_sim = cosine(candidate_vector, mean_reference_vector)
```

If not provided: this term is zeroed out and its weight redistributed.

### Composite Candidate Rank (User-Configurable)

The composite rank evaluates candidates by combining multiple active quality and semantic signals. Users can toggle individual signals via the UI (`scoring_signals` parameter in the API):

| Signal Key | Source | Description | Default Status |
|:---|:---|:---|:---|
| `cosine` | SigLIP2 Search Score | Normalized vector similarity to text sub-query | **Locked ON** |
| `nima` | MobileNetV2 NIMA Model | Neural aesthetic score (composition, lighting) | **ON** |
| `sharpness` | OpenCV Laplacian | Clarity filter (penalizes blur & mid-motion fuzz) | **ON** |
| `contrast` | OpenCV Histogram (p95-p5) | Dynamic range / tonal richness | **ON** |
| `colorfulness` | Hasler-Süsstrunk Metric | Color variance & saturation | OFF (prevents flower bias) |
| `epic_sim` | SigLIP2 Cosine to Reference | Similarity to user-uploaded epic reference photos | Opt-in (if refs provided) |

#### Dynamic Weight Normalization

Base weights per profile for all available signals:

| Profile | Cosine | NIMA | Sharpness | Contrast | Colorfulness | Epic Sim |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cinematic Epic** | 0.40 | 0.30 | 0.050 | 0.0350 | 0.0150 | 0.20 |
| **Personal Moments** | 0.50 | 0.15 | 0.125 | 0.0875 | 0.0375 | 0.10 |
| **Balanced Journey** | 0.50 | 0.20 | 0.075 | 0.0525 | 0.0225 | 0.15 |

When a user selects an active subset of signals $S \subseteq \{\text{cosine}, \text{nima}, \text{sharpness}, \text{contrast}, \text{colorfulness}, \text{epic\_sim}\}$, the active weights are normalized dynamically:

$$w_s = \frac{\text{base\_weight}_s}{\sum_{k \in S} \text{base\_weight}_k}$$

Ensuring that $\sum_{s \in S} w_s = 1.0$ at all times regardless of which combination of signals is chosen.
If `epic_sim` is selected but no reference vector is supplied, its weight is automatically redistributed among the other active signals.

### Diversity Penalty

Applied after initial ranking, before passing candidates to the drafter:

```python
selected = []
for candidate in ranked_candidates:
    penalty = 1.0
    for already_selected in selected:
        inter_sim = cosine(candidate.vector, already_selected.vector)
        if inter_sim > 0.85:
            penalty *= 0.5  # halve the score for each near-duplicate
    candidate.adjusted_rank = candidate.rank * penalty
    selected.append(candidate)
```

This prevents the candidate list from being dominated by visually similar frames.

---

## Layer 3: Editor-Time Storyboard Scoring

Computed after the drafter outputs a storyboard. Used for best-draft watermarking.

### Duration Accuracy (0.0 – 1.0)

```python
total = sum(seg.duration for seg in storyboard)
if total >= target:
    accuracy = 1.0 - max(0, (total - target * 1.25)) / target
else:
    accuracy = total / target  # linear penalty for under-target
```

Under-target is penalized more heavily than over-target (per user directive: never shorter, up to +25% is ok).

### Pacing Rhythm (0.0 – 1.0)

```python
durations = [seg.duration for seg in storyboard]
std_dev = np.std(durations)
# Good pacing has variety. Normalize std_dev against expected range.
rhythm = min(1.0, std_dev / 1.5)
```

A storyboard where every segment is exactly 3.0s scores 0.0 (monotonous). A mix of 1.5s, 3.0s, and 5.0s segments scores higher.

### Visual Quality (0.0 – 1.0)

```python
quality = mean(seg.scores.nima_aesthetic for seg in storyboard)
```

Mean aesthetic quality of all selected segments.

### Query Relevance (0.0 – 1.0)

```python
relevance = mean(seg.similarity_score for seg in storyboard)
# Normalize: SigLIP2 scores 0.08-0.25. Map to 0.0-1.0.
relevance = (relevance - 0.08) / 0.17
```

### Diversity (0.0 – 1.0)

```python
# Temporal spread: what fraction of time buckets are represented?
bucket_coverage = len(unique_buckets) / total_buckets

# Media type mix: penalty if all same type
type_mix = 1.0 if has_photos and has_videos else 0.5

diversity = 0.6 * bucket_coverage + 0.4 * type_mix
```

### Composite Storyboard Score

```python
composite = (
    0.25 * duration_accuracy +
    0.20 * pacing_rhythm +
    0.25 * visual_quality +
    0.20 * query_relevance +
    0.10 * diversity
)
```

This score drives best-draft watermarking. The storyboard with the highest composite across all iterations is the one that gets compiled and rendered.
