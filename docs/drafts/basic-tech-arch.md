# Technical Architecture & Implementation Specification: Local AI Moments Generator

## 1. System Overview & Deployment Topology

The application follows a **Decoupled Hybrid Architecture** designed to extract maximum performance from Apple Silicon (M5/M5 Pro unified memory and Neural Engine/GPU) while maintaining strict resource boundaries and data isolation[cite: 1].


```

+-------------------------------------------------------------------------------+
| macOS Host (Apple Silicon - M5 / M5 Pro)                                      |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  | Python Virtual Environment (.venv via uv)                               |  |
|  |                                                                         |  |
|  |  +---------------------+   +---------------------+   +---------------+  |  |
|  |  | FastAPI Web Server  |   | Apple MLX Engine    |   | FFmpeg Engine |  |  |
|  |  | - REST Endpoints    |---| - SigLIP 2 Vision   |   | - Decoders    |  |  |
|  |  | - SSE Event Stream  |   | - SigLIP 2 Text     |   | - Encoders    |  |  |
|  |  | - Static UI Assets  |   | - Unified Memory Ops|   | - FilterGraphs|  |  |
|  |  +---------------------+   +---------------------+   +---------------+  |  |
|  |             |                         |                      |          |  |
|  +-------------|-------------------------|----------------------|----------+  |
|                |                         |                      |             |
|                | (HTTP/gRPC: 6333)       |                      |             |
|                v                         |                      |             |
|  +----------------------------+          |                      |             |
|  | Docker Container           |          |                      |             |
|  |  +----------------------+  |          |                      |             |
|  |  | Qdrant Vector DB     |  |          |                      |             |
|  |  | - mmap Vector Index  |  |          |                      |             |
|  |  | - On-disk Payload    |  |          |                      |             |
|  |  +----------------------+  |          |                      |             |
|  +--------------|-------------+          |                      |             |
|                 |                        |                      |             |
|                 v                        v                      v             |
|  +-------------------------------------------------------------------------+  |
|  | Host File System                                                        |  |
|  | - Raw Media Corpus (<= 20 GB)                                           |  |
|  | - Qdrant Mapped Volume (./qdrant_storage)                               |  |
|  | - Render Output Directory (./exports/*.mp4)                             |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+

```

### 1.1. Process Boundaries & Execution Contexts
1. **Core Service (macOS Native):** FastAPI, Apple MLX, and FFmpeg run natively as host macOS processes within a dedicated Python virtual environment. This bypasses Docker's hypervisor layer, providing raw access to Apple Metal Performance Shaders (MPS) and the M5 Neural Engine.
2. **Persistence Service (Docker Container):** Qdrant executes in an isolated Docker container bound to `localhost:6333`. It is memory-capped and relies on host-mounted volumes for persistent vector storage.

---


## 2. AI Inference & Hardware Acceleration (Apple MLX + SigLIP 2)

### 2.1. Model Architecture

* **Vision & Text Model:** `google/siglip2-base-patch16-224` (or `siglip2-so400m` if memory headroom permits).
* **Execution Framework:** Apple MLX (`mlx`, `mlx-vlm`).
* **Precision:** Float16 (`fp16`) to maximize memory bandwidth and throughput on M5 unified memory.

### 2.2. Embedding Pipeline

1. **Model Initialization:** Loaded into unified memory once at FastAPI application startup via an MLX model manager singleton.
2. **Text Encoding:**
* Text prompt is normalized and passed through the SigLIP 2 text tokenizer.
* MLX text encoder outputs a normalized 768-dimensional float16 vector.


3. **Vision Encoding:**
* Extracted frames/images are preprocessed (scaled and normalized according to SigLIP 2 transforms).
* Images are batched dynamically (batch size = 32 or 64) and evaluated concurrently on the M5 GPU/Neural Engine.
* Output vectors are L2-normalized prior to ingestion into Qdrant.



---

## 3. Ingestion & Preprocessing Subsystem

### 3.1. Directory Traversal & Metadata Extraction

1. **Recursive Scan:** The system traverses the user-specified input folder.
2. **Format Support:**
* **Images:** `.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.tiff`
* **Videos:** `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`


3. **Timestamp Resolution Priority:**
1. EXIF Metadata (`DateTimeOriginal` / `SubSecTimeOriginal`) via `exifread` / `Pillow`.
2. Video Container Metadata (`quicktime:creationdate`, `creation_time`) via `ffprobe`.
3. Filesystem birth/modification time (`os.stat().st_birthtime` or `st_mtime`).



### 3.2. Video Frame Extraction Strategy

* Videos are sampled at a fixed rate of **1 Frame Per Second (1 FPS)**.
* FFmpeg extraction command uses hardware acceleration (`videotoolbox` where available):
```bash
ffmpeg -hwaccel videotoolbox -i input.mp4 -vf "fps=1,scale=224:224:force_original_aspect_ratio=increase,crop=224:224" -f image2pipe -pix_fmt rgb24 -vcodec rawvideo pipe:1

```


* Frames are streamed directly from the standard output pipe into MLX arrays in memory, bypassing intermediate disk I/O.

---

## 4. Curation, Pacing & Culling Algorithms

### 4.1. Temporal Time-Bucketing Algorithm

To prevent temporal clustering (e.g., thousands of photos from Day 2 dominating a 5-year corpus):

1. **Calculate Global Time Range:**

$$\Delta T_{corpus} = T_{max} - T_{min}$$


2. **Determine Bucket Count ($K$):**
Given target output duration $D_{out} \in [1, 300]$ seconds:



$$K = \min\left(10, \max\left(5, \left\lfloor \frac{D_{out}}{15} \right\rfloor\right)\right)$$


3. **Partition Intervals:**
Divide $[T_{min}, T_{max}]$ into $K$ uniform time intervals:

$$B_i = \left[ T_{min} + i \cdot \frac{\Delta T_{corpus}}{K}, \; T_{min} + (i+1) \cdot \frac{\Delta T_{corpus}}{K} \right) \quad \text{for } i \in [0, K-1]$$


4. **Target Allocation per Bucket:**
Each bucket $B_i$ receives a time quota:

$$t_{quota} = \frac{D_{out}}{K}$$


5. **Filtered Vector Query:**
For each bucket $B_i$, execute a Qdrant filtered search:
```json
{
  "filter": {
    "must": [
      {
        "key": "creation_timestamp",
        "range": {
          "gte": B_i.start,
          "lt": B_i.end
        }
      }
    ]
  },
  "limit": 50,
  "with_payload": true
}

```



### 4.2. Semantic Thresholding & Fallback

* Let $S(p, v) = \frac{p \cdot v}{\|p\|_2 \|v\|_2}$ be the cosine similarity between prompt vector $p$ and frame vector $v$.
* **Relevance Floor:** Minimum threshold $\theta_{min} = 0.22$. Any candidate scoring below $\theta_{min}$ is pruned.
* **Zero-Match Trigger:** If $\sum_{i=0}^{K-1} |\{v \in B_i \mid S(p, v) \ge \theta_{min}\}| == 0$, abort execution and return `422 Unprocessable Entity` with reason `ZERO_SEMANTIC_MATCHES`.

### 4.3. Burst & Proximity Deduplication

To eliminate repetitive burst shots and consecutive frames of the same video:

1. **Visual Similarity Pruning:** If two selected images have a cosine distance $< 0.05$ or a perceptual hash Hamming distance $< 5$, only the higher-scoring item is retained.
2. **Temporal Windowing for Video:**
* If frame $f_a$ at offset $t_a$ and frame $f_b$ at offset $t_b$ belong to the same video file:
* If $|t_a - t_b| < 4.0\text{ seconds}$, they are merged into a single segment starting at $\max(0, \min(t_a, t_b) - 0.5)$ with a duration of $3.0\text{ seconds}$.


3. **Static Image Duration:** Each selected static image is assigned a duration of $3.0\text{ seconds}$.

---

## 5. Rendering & Stitching Pipeline (FFmpeg)

### 5.1. Output Specifications

* **Container:** MP4
* **Video Codec:** `libx264` (or `h264_videotoolbox` on macOS)
* **Pixel Format:** `yuv420p`
* **Canvas Resolution:** $1080 \times 1080$ (Strict 1:1 Aspect Ratio)
* **Frame Rate:** $30\text{ FPS}$
* **Audio Track:** None (`-an`) for Phase 0.

### 5.2. Aspect Ratio Normalization Filter Graph

Every visual asset is scaled and padded to fit the $1080 \times 1080$ canvas without stretching or cropping.

**FFmpeg Filter String per Input:**

```
[in]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:color=black,setsar=1,fps=30[out]

```

### 5.3. Timeline Assembly (Complex Filter Graph)

For an assembled timeline with $N$ clips:

```bash
ffmpeg -y \
  -loop 1 -t 3.0 -i image_1.jpg \
  -ss 14.5 -t 3.5 -i video_1.mp4 \
  -loop 1 -t 3.0 -i image_2.heic \
  -filter_complex "
    [0:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v0];
    [1:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v1];
    [2:v]scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(1080-iw)/2:(1080-ih)/2:black,setsar=1,fps=30[v2];
    [v0][v1][v2]concat=n=3:v=1:a=0[v_out]
  " \
  -map "[v_out]" \
  -c:v h264_videotoolbox \
  -b:v 6000k \
  -pix_fmt yuv420p \
  ./exports/moments_output.mp4

```

---

## 6. API Architecture & Event Streaming

### 6.1. REST Endpoints

#### `POST /api/v1/jobs/generate`

Initiates a scanning, indexing, scoring, and rendering pipeline.

* **Request Body:**

```json
{
  "corpus_path": "/Users/developer/Pictures/Trek2025",
  "prompt": "Moments from the trek with epic mountain views",
  "target_duration_seconds": 120,
  "force_reindex": false
}

```

* **Response (202 Accepted):**

```json
{
  "job_id": "8f3b6c2a-9e12-4d5a-b9c1-ef4a37b12d54",
  "status": "QUEUED",
  "created_at": 1771141200
}

```

#### `GET /api/v1/jobs/{job_id}/events`

Server-Sent Events (SSE) connection streaming real-time pipeline execution progress.

* **Event Stream Format:**

```
event: progress
data: {"stage": "SCANNING", "progress_pct": 100, "details": "Found 412 files (14.2 GB)"}

event: progress
data: {"stage": "INDEXING", "progress_pct": 45.2, "details": "Embedded 186/412 files"}

event: progress
data: {"stage": "CURATING", "progress_pct": 100, "details": "Selected 34 segments across 8 time buckets"}

event: progress
data: {"stage": "RENDERING", "progress_pct": 72.0, "details": "FFmpeg rendering frame 1800/2500"}

event: complete
data: {"job_id": "8f3b6c2a-9e12-4d5a-b9c1-ef4a37b12d54", "output_path": "/Users/developer/exports/output.mp4", "duration": 119.5}

```

#### `GET /api/v1/jobs/{job_id}/download`

Serves the rendered MP4 file for local browser playback.

---


## 7. Automated Host Bootstrapper (`start.sh`)

This script initializes the environment on macOS, ensuring native execution of the AI backend alongside the containerized database.

```bash
#!/usr/bin/env bash
set -eo pipefail

echo "=== Initializing Local AI Moments Generator ==="

# 1. Dependency Validation
command -v python3 >/dev/null 2>&1 || { echo "Error: Python3 is required."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: FFmpeg is required. Install with 'brew install ffmpeg'."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Error: Docker is required."; exit 1; }

# 2. Spin up Qdrant Vector DB
echo "--- Ensuring Qdrant is running ---"
docker compose up -d qdrant

# 3. Setup Virtual Environment via uv
if [ ! -d ".venv" ]; then
    echo "--- Creating virtual environment ---"
    python3 -m venv .venv
fi

source .venv/bin/activate

if ! command -v uv >/dev/null 2>&1; then
    echo "--- Installing uv package manager ---"
    pip install --upgrade pip
    pip install uv
fi

echo "--- Syncing dependencies ---"
uv pip install -r requirements.txt

# 4. Start Server
echo "--- Starting FastAPI Server natively on macOS (Apple Silicon Accelerated) ---"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

