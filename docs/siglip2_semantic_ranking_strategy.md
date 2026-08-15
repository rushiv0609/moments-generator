# Architectural Decision: Semantic Ranking Strategy over Absolute Thresholding

## Overview
During the implementation of the Semantic Playground, we observed a critical behavioral characteristic of the **SigLIP 2** Vision-Language Model: 

When comparing an image of an *arid mountain peak* against the text prompt *"snowy peak"*, the model yielded a cosine similarity score of `+0.0815`. Against the prompt *"arid rocky mountain landscape with colorful strata"*, it yielded a score of `+0.12`.

While an initial inclination might be to set a hard `MIN_SIMILARITY_THRESHOLD` at `>= 0.15` to strictly discard mismatched concepts like "snowy", doing so breaks the core functionality of a natural language photo search engine.

## The "Bag of Concepts" Effect
Contrastive models map images and text to a shared dimensional space (768-dim L2 normalized float32 vectors in our implementation). 
- The prompt `"snowy peak"` contains two dominant semantic anchors: `snowy` and `peak`. 
- The image lacks `snowy` but provides an extremely strong visual representation of a `peak`. 

Instead of strictly executing `(Has Peak) AND (Has Snowy) = FALSE`, the model performs a partial overlap calculation: `(Score for Peak) + (Score for Snow) = Partial Match (+0.0815)`.

## Why We Must Use Top-K Ranking (Not High Thresholds)

> [!WARNING]
> Setting a high absolute threshold (e.g., `0.15`) will drastically reduce **Recall** and cause the engine to return 0 results for typical, short, user-provided queries (e.g. "mountain"). 

### 1. User Query Behavior
Users rarely write mathematically optimal, highly descriptive queries like *"arid rocky mountain landscape with colorful strata"*. They type short, casual phrases. Short prompts mathematically produce lower absolute dot-product scores because they provide fewer context features to match the rich embedding of an image.

### 2. Sigmoid Loss Compression
Unlike OpenAI's CLIP which distributes similarities over a wide range (e.g. `0.2` to `0.4`), SigLIP 2 uses a Sigmoid loss function. This naturally compresses similarity distributions tightly around `0.0`. In SigLIP 2, `0.05` is a decent match, and `0.12` is an incredibly strong match. 

### 3. The Top-K Solution
Instead of discarding images that score `+0.08`, we rely entirely on **Vector Database Top-K Ranking** (via Qdrant in Milestone 7):
- **Precision:** If the user searches `"snowy peak"` and the library contains *actual* snowy peaks (scoring `+0.15`), those images will rank #1-10.
- **Recall:** The arid mountain (scoring `+0.08`) will still be returned, but it will rank lower (e.g. #45). It is always a better UX to allow a user to scroll down to loosely related context than to hit a hard wall of "0 results found".

## Implementation Changes
To properly implement this strategy in the Moments Generator:
- `app/config.py`: The `MIN_SIMILARITY_THRESHOLD` is lowered to `0.05`. This serves only as a safety net to filter out absolute garbage or distractors (`< 0.0`), not to enforce strict compositional precision.
- **UI Diagnostics:** The Semantic Playground is explicitly labeled to reflect that `+0.08` is a "Strong Match / High likelihood to rank at the top", matching the true mathematical distribution of SigLIP 2.
