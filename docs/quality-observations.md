# Quality Observations

Documented problems observed in production runs against the Pin Bhabha trek corpus (1,602 files, 4,085 vectors).

---

## Problem 1: Missing Epic Frames

**What happens**: The pipeline retrieves semantically relevant candidates (e.g., query "mountain landscape" returns mountain photos) but misses the *best* mountain photos. A blurry ground-level shot of a mountain and a stunning HDR panorama both match the query "mountain landscape" with similar cosine scores because SigLIP2 encodes *what is in the image*, not *how good it looks*.

**Root cause**: The retrieval stage ranks candidates by a single signal — cosine similarity between the query text embedding and the frame vision embedding. There is no quality signal. A frame with cosine 0.12 that's a blurry snapshot beats a frame with cosine 0.11 that's a stunning panorama.

**Evidence from session `job_gen_3734ccc9`**:
- 68 candidates retrieved across 13 queries
- Several high-quality panoramic shots from the corpus were absent from the candidate pool
- The top-ranked candidates by cosine score included multiple mediocre-quality frames that happened to match the query text well

**What "epic" means in different contexts**:

| Context | What makes a frame epic |
|:---|:---|
| Mountain trek | Grand mountain views, dramatic clouds, alpine valleys, ridge panoramas |
| Beach trip | Sunset/sunrise over water, turquoise ocean, wide shoreline views |
| Friend gathering | Candid laughter, group photos, bonding moments |
| Creative photography | Motion blur waterfalls, silhouettes, long exposure, unusual angles |

The system has no way to distinguish these or to understand that "epic" is context-dependent.

---

## Problem 2: Inconsistent Video Duration

**What happens**: The drafter generates storyboards that don't fill the target duration consistently. The same prompt with the same corpus produces wildly different durations depending on which retrieval mode is used.

**Observed data**:

| Target | Dual Mode Output | Scene Mode Output | Expected |
|:---:|:---:|:---:|:---:|
| 60s | ~30s | ~90s | 60–75s |
| 120s | ~40s | ~120s | 120–150s |

**Root cause (multiple)**:
1. The drafter's system prompt says "target approximately X seconds" but doesn't enforce minimum segment counts. The LLM sometimes produces 6 segments at 3s each = 18s total for a 60s target.
2. The editor's duration tolerance is ±25%, which means a 60s target accepts 45–75s. The lower bound (45s) is too generous — a 45s video for a 60s request feels short.
3. The editor can reject a draft for being short, but its feedback ("add more segments") is vague. The drafter has no guidance on *which* candidates to add.
4. Dual mode retrieves fewer unique candidates than scene mode because it deduplicates across both granularities, leading to fewer options for the drafter.

---

## Problem 3: Timeline Ordering and Editor Degradation

### 3a: Chronological Breaks

**What happens**: The drafter sometimes places a Day 4 photo before a Day 1 frame. The system prompt says "arrange in chronological order" but the LLM doesn't always comply, especially with 20+ candidates to sort.

**Root cause**: The LLM receives candidates sorted by score (highest first) and is asked to re-sort by date. With small local models (gemma4:e2b), long context re-ordering is unreliable. The model tends to preserve the order it received the candidates in.

### 3b: Editor Keeps Degraded Drafts

**What happens**: In a 3-iteration drafting loop:
- Iteration 1: Pacing score 6/10 (decent)
- Iteration 2: Pacing score 7/10 (improved — best so far)
- Iteration 3: Pacing score 4/10 (degraded — LLM over-corrected)

The system keeps iteration 3's output because it's the latest, not iteration 2's which was the best.

**Root cause**: The state only tracks the current storyboard. There is no watermark of the best draft seen so far. The editor's `approved` flag gates the loop exit, but the best-scoring intermediate draft is lost.

### 3c: Single-Dimension Scoring

**What happens**: The editor scores storyboards on "pacing" — a single subjective score from the LLM. This misses duration accuracy, visual quality of selected frames, and relevance to the user's query.

**Root cause**: The `EditorOutput` schema has one `pacing_score` field. There is no composite score combining measurable signals.

---

## Problem 4: Rendering Quality

### 4a: Resolution Degradation

**What happens**: Source files are 4K (3840x2160) or native camera resolution (4032x3024 for iPhone). The renderer forces everything to 1920x1080 using `scale=1920:1080:force_original_aspect_ratio=increase` then crops. This:
- Downscales 4K to 1080p (losing detail)
- Crops to 16:9 (cutting edges of 4:3 photos)
- Applies center crop regardless of subject position

### 4b: Speed-Up Artifacts

**What happens**: When a video clip's source duration at the selected offset is shorter than the requested segment duration, the renderer currently just truncates. But the inverse problem exists too: when the drafter requests 3s from a 10s clip, FFmpeg trims to 3s which is fine. However, when scene boundaries produce a 2s natural clip and the drafter assigns 3s, the renderer has no speed adjustment — it just truncates at whatever's available.

The deeper issue: the hard 5.0s cap on video clips means epic establishing shots (e.g., a 12-second slow river pan) get cut to 5s, losing their impact.

### 4c: Mechanical Ken Burns

**What happens**: Ken Burns patterns cycle mechanically: `zoom_in → pan_right → zoom_out → pan_left`, repeating. This means:
- A panoramic landscape might get `zoom_in` (should be `zoom_out` to reveal the scene)
- A portrait of a person might get `pan_right` (should zoom toward the face)
- The zoom delta (15%) is too aggressive for short durations — causes visible judder
- Linear interpolation creates abrupt start/stop — no easing

**Root cause**: `get_ken_burns_pattern(index)` uses `index % 4` to pick the pattern. No analysis of image content.
