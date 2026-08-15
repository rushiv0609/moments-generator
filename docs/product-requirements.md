# Product Requirements Document (PRD): Local AI Moments Generator

**Version:** 1.0  
**Status:** Finalised  
**Last Updated:** 2026-08-15

---

## 1. Product Overview

### 1.1 Problem Statement

Users possess large, unstructured collections of local photos and videos accumulated over trips, events, and years of daily life. There is no simple, offline tool that can take a natural-language description — such as *"Moments from the trek with epic mountain views"* — and automatically produce a curated highlight video composed entirely of the user's own captured media.

### 1.2 Solution

A **locally-running desktop application** that:

1. Ingests a user-specified folder of photos and videos (up to 20 GB).
2. Uses a multimodal AI embedding model to semantically understand every image and video frame.
3. Accepts a natural-language text prompt describing the desired "moments."
4. Scores, curates, and temporally balances the most relevant media.
5. Stitches the selections into a polished output video — no generative AI, no cloud, no artificial content.

### 1.3 Target Environment

| Constraint | Value |
|---|---|
| **Hardware** | Apple Silicon — Mac M5 or M5 Pro |
| **OS** | macOS (native execution, not containerised) |
| **Network** | Fully offline after initial model download |
| **Processing paradigm** | Semantic stitching of captured moments — NOT generative video synthesis |

### 1.4 Global Constraints

| Constraint | Limit | Rationale |
|---|---|---|
| Maximum corpus size | 20 GB per execution | Memory and storage budget for a laptop |
| Maximum output duration | 300 seconds (5 minutes) | Keeps output watchable; bounds rendering time |
| Latency | No strict target for MVP | Accuracy and quality over speed; optimise later |

---

## 2. User Interaction Model

### 2.1 Inputs

| Input | Type | Description |
|---|---|---|
| **Corpus path** | Directory path | Local folder (with sub-directories) containing all source media |
| **Prompt** | Free-text string | Natural-language description of the desired moments |
| **Target duration** | Integer (seconds) | Desired length of the output video (1–300 s) |
| **Output aspect ratio** | Enum | `1:1` (default), `16:9`, `9:16` |
| **Force re-index** | Boolean | If `true`, ignore cached embeddings and reprocess everything |

### 2.2 Outputs

- A single MP4 video file containing the curated moments.
- Real-time progress updates streamed to the UI during processing.

### 2.3 Error Cases

| Condition | Behaviour |
|---|---|
| Corpus path does not exist or is empty | Return error with clear message |
| No media files found in the corpus | Return error listing the supported formats |
| Zero semantic matches above the relevance threshold | Abort with `ZERO_SEMANTIC_MATCHES`; suggest broadening the prompt |
| Corrupt or unreadable file in the corpus | Skip the file, log a warning, continue processing the rest |
| Target duration exceeds 300 s | Clamp to 300 s and warn the user |

---

## 3. Media Ecosystem & Format Support

The 20 GB corpus will be messy, heterogeneous, and come from diverse devices.

### 3.1 Source Devices

Smartphones (iPhone, Android), action cameras (GoPro), drones, DSLRs, mirrorless cameras, webcams.

### 3.2 Supported File Formats

| Category | Formats |
|---|---|
| **Images** | `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.webp`, `.tiff` |
| **Videos** | `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v` |

> **Note:** HEIC/HEIF requires explicit library support (`pillow-heif`). This is validated at application startup.

### 3.3 Metadata Handling

Timestamps are critical for temporal bucketing. The system resolves creation time using this priority chain:

| Priority | Source | Library |
|---|---|---|
| 1 (highest) | EXIF `DateTimeOriginal` / `SubSecTimeOriginal` | `exifread` / `Pillow` |
| 2 | Video container metadata (`quicktime:creationdate`, `creation_time`) | `ffprobe` (JSON output) |
| 3 (fallback) | Filesystem birth time (`st_birthtime`) or modification time (`st_mtime`) | `os.stat()` |

When metadata is missing, stripped, or conflicting, the system falls back silently through this chain and never crashes.

---

## 4. Semantic Scenarios & Corpus Profiles

The scoring engine must handle vastly different corpus types and user intents.

### Scenario A: High-Action Trek

- **Corpus:** Mountains, river crossings, epic views, people hiking. Mixed action-camera (GoPro) and smartphone footage.
- **Example prompt:** *"Moments from the trek with epic mountain views"*
- **Challenge:** Differentiate a casual selfie from a frame where the environmental context (peaks, valleys) is the primary semantic focus.

### Scenario B: Friends Fun Trip

- **Corpus:** Partying, dancing, group selfies. Heavily low-light indoor, motion blur, chaotic framing.
- **Example prompt:** *"Make a video of my trip with friends that has the most bonding and beautiful moments"*
- **Challenge:** Interpret abstract emotional concepts ("bonding" → people hugging, laughing together) and prioritise human interaction over scenery.

### Scenario C: Specific Vibe / Location

- **Corpus:** Beach trips — sand, ocean, sunsets, resort pools.
- **Example prompt:** *"Give me a video from the collection of beach trips"*
- **Challenge:** Filter out transit, indoor, and irrelevant media; strictly identify visual markers of the requested environment.

### Scenario D: Multi-Year Photo Dump

- **Corpus:** 5 years of accumulated data. Extreme variance in lighting, locations, devices, and people.
- **Example prompt:** *"Find the best bonding moments with friends"*
- **Challenge:** Severe temporal skew — the time-bucketing strategy must ensure the output actually represents the full span, not just one over-photographed vacation.

---

## 5. Phased Requirements

### Phase 0 — MVP (Build First)

#### 5.1 Ingestion & Configuration
- Recursive directory scan with sub-directory support.
- Accept all formats listed in §3.2.
- HEIC/HEIF decoding validated at startup.
- Metadata extraction for timestamp resolution (§3.3).

#### 5.2 Manifest & Caching
- SQLite-based manifest tracks every file's processing state.
- Content-addressed hashing detects new, modified, and deleted files.
- Incremental re-indexing: on re-run, only process new/changed files.
- Model version tracking: if the embedding model changes, re-embed all files (but preserve extracted metadata).
- Checkpoint/resume: if the pipeline crashes, restart from the last completed stage per file.

#### 5.3 Parallel Ingestion Pipeline
- Multi-threaded frame extraction (N concurrent FFmpeg workers).
- GPU-batched embedding (dynamic batch size).
- Bounded queues with backpressure between pipeline stages.
- Per-file error isolation: corrupt files are skipped and logged, never crash the pipeline.

#### 5.4 Semantic Matching & Scoring
- Multimodal embedding model encodes both text prompts and visual content into a shared vector space.
- Cosine similarity scoring between prompt embedding and frame/image embeddings.
- Configurable minimum relevance threshold (default: 0.22, tunable).
- Zero-match abort with clear user feedback.

#### 5.5 Curation & Timeline Composition
- **Time-bucketing:** Divide the corpus time range into K equal intervals; select highest-scoring media within each bucket to ensure chronological balance.
- **Burst deduplication:** Eliminate near-duplicate burst photos (cosine distance < 0.05 or perceptual hash Hamming distance < 5).
- **Video segment merging:** Consecutive high-scoring frames from the same video within 4 seconds are merged into a single 3-second clip.
- **Static image duration:** Each selected image appears for 3 seconds.

#### 5.6 Rendering & Export
- **Aspect ratio normalisation:** Configurable canvas (`1:1`, `16:9`, `9:16`). Mismatched source media is letter/pillar-boxed — never stretched or cropped.
- **Codec:** H.264 via `h264_videotoolbox` (hardware-accelerated on macOS).
- **Frame rate:** 30 FPS.
- **Transitions:** Hard cuts for MVP. Crossfades deferred to Phase 1.
- **Audio:** Silent. All source audio tracks are stripped.
- **Output format:** MP4, saved to a configurable export directory.
- **Timeline persistence:** The curated segment list is saved to the manifest before rendering starts, enabling re-render without re-curation.

#### 5.7 API & Progress
- REST API for job submission and status.
- Server-Sent Events (SSE) for real-time pipeline progress streaming.
- Job states: `QUEUED → SCANNING → INDEXING → CURATING → RENDERING → COMPLETE | FAILED`.

#### 5.8 UI
- Minimal single-page web UI served as static assets by FastAPI.
- Form: corpus path, prompt, duration, aspect ratio.
- Progress bar consuming the SSE stream.
- Video preview and download on completion.

---

### Phase 1 — Fast Follow

#### 5.9 Iterative Generation
- Generate 2–3 candidate output videos per query.
- User can provide feedback on candidates (thumbs up/down, "more of this, less of that").

#### 5.10 Identity Management (Face Tagging)
- User uploads reference photos of people and tags them with names.
- System identifies those people across the corpus using face embeddings.
- Prompts can reference tagged people: *"Best moments with Rushi and Aanya"*.

#### 5.11 Transition Effects
- Crossfade transitions between clips using FFmpeg's `xfade` filter.
- Configurable transition duration (default: 0.5 s).

---

### Phase 2 — Polish & Expansion

#### 5.12 Audio Integration
- Overlay a background music track (user-provided or from a bundled library).
- Beat detection to align cuts with music rhythm.
- Audio ducking / volume normalisation.

#### 5.13 Image Quality Scoring
- Secondary ranking signal: lightweight image quality assessment (blur detection, exposure).
- Blurry or badly exposed frames are penalised even if semantically relevant.

---

## 6. Non-Functional Requirements

| Requirement | Target |
|---|---|
| **Privacy** | All processing is local. No data leaves the machine. No telemetry. |
| **Idempotency** | Running the same prompt + corpus + duration twice produces the same output (deterministic curation). |
| **Resilience** | Pipeline survives corrupt files, missing metadata, and mid-run crashes via manifest checkpointing. |
| **Resource usage** | Peak memory ≤ 5 GB (comfortable on 18+ GB unified memory). Peak disk for working data ≤ 2 GB (manifests + vector index). |
| **Startup time** | Model loaded and ready to accept jobs within 30 seconds of server start. |
