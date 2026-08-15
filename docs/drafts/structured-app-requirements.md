# Product Requirements Document (PRD): Local AI Moments Generator

## 1. Product Overview
**Problem Statement:** Users possess large, unstructured corpuses of local photos and videos[cite: 1]. The goal is to generate a highly relevant, curated "moments" video strictly from this local corpus based on a natural language text prompt, without relying on cloud processing or generative video models[cite: 1]. 

**Target Environment:** 
*   Must run entirely locally on Apple Silicon (specifically Mac M5 or M5 Pro processors)[cite: 1].
*   The application focuses on semantic stitching of captured moments, not artificial video generation[cite: 1].

**Global Constraints:**
*   **Maximum Corpus Size:** 20 GB per execution[cite: 1].
*   **Maximum Output Duration:** 300 seconds (5 minutes)[cite: 1].
*   **Latency:** No strict latency constraints for the MVP; accuracy and functionality are prioritized over speed[cite: 1].

---

## 2. Media Ecosystem & Device Variability
The application must handle the messy reality of raw user media. The 20 GB corpus will rarely be clean or uniform.

*   **Source Devices:** Media can originate from diverse hardware, including smartphones (iPhones, Androids), action cameras (GoPros), drones, and dedicated digital cameras (DSLRs/Mirrorless).
*   **File Formats:** The media parser must support a heterogeneous mix of formats. This includes standard images (JPEG, PNG), modern compressed images (HEIC), raw formats (if feasible for MVP), and diverse video containers/codecs (MP4, MOV, HEVC, H.264).
*   **Metadata Chaos:** The system must account for missing, stripped, or conflicting EXIF data, relying on file creation dates or heuristic sorting when standard timestamps fail.

---

## 3. Semantic Scenarios & Corpus Profiles
The semantic scoring engine must be robust enough to adapt its filtering logic based on vastly different corpus profiles and user intents. Below are the core scenarios the system must successfully process:

**Scenario A: The High-Action Trek**
*   **Corpus Profile:** Contains people, mountains, river crossings, and epic views[cite: 1]. Heavily features landscapes, high dynamic range lighting, and potentially shaky action-camera video clips.
*   **Example Prompt:** "Moments from the trek with epic mountain views"[cite: 1].
*   **Algorithmic Challenge:** The system must differentiate between a standard photo of a person and a photo where the environmental context (mountains/views) is the primary semantic focus.

**Scenario B: The Friends Fun Trip**
*   **Corpus Profile:** Contains fun moments, partying moments, and random dancing moments[cite: 1]. Heavily features low-light indoor environments, motion blur, and chaotic group shots.
*   **Example Prompt:** "Make a video of my trip with friends that has most bonding and beautiful moments"[cite: 1].
*   **Algorithmic Challenge:** The system must interpret abstract emotional concepts like "bonding" (e.g., people hugging, laughing together) and prioritize faces and human interaction over environmental scenery.

**Scenario C: The Specific Vibe/Location**
*   **Corpus Profile:** A localized collection focusing on a specific environment, such as a beach trip.
*   **Example Prompt:** "Give me a video from the collection of beach trips"[cite: 1].
*   **Algorithmic Challenge:** The system must filter out irrelevant indoor or transit media and strictly identify visual markers of the requested environment (sand, ocean, sunsets).

**Scenario D: The Multi-Year Photo Dump**
*   **Corpus Profile:** 5 years worth of accumulated data[cite: 1]. Extremely high variance in lighting, locations, devices, and people.
*   **Example Prompt:** Find the best bonding moments with friends[cite: 1].
*   **Algorithmic Challenge:** Severe temporal skewness risk. The time-bucketing strategy is critical here to ensure a 300-second output actually represents the full 5-year span rather than clustering entirely on a single highly photographed vacation.

---

## 4. Core Requirements: MVP (Phase 0)

### 4.1. Input & Configuration
*   **Corpus Ingestion:** The user will input a local directory path[cite: 1]. The system must recursively scan the directory and its sub-directories for the diverse media formats listed in Section 2[cite: 1].
*   **User Prompt:** The system must accept a natural language query[cite: 1].
*   **Target Duration:** The user will specify a desired output video length in seconds (e.g., 120s, 240s)[cite: 1].

### 4.2. Media Analysis & Scoring
*   **Semantic Matching:** The system must analyze visual contents and score relevance against the user's prompt, adapting to the diverse scenarios outlined in Section 3.
*   **Graceful Fallbacks (Zero-Match):** If the corpus yields no media meeting a minimum semantic relevance threshold, the application must immediately alert the user rather than outputting an empty video.

### 4.3. Curation & Timeline Composition
*   **Temporal Pacing (Time-Bucketing):** The requested duration must be divided into equal chronological segments (e.g., 8-10 buckets). The highest-scoring media must be selected *within each bucket* to prevent chronological skewness.
*   **De-duplication:** The system must identify burst photos or nearly identical consecutive video frames, aggressively discarding duplicates to maintain visual variety.

### 4.4. Rendering & Export
*   **Aspect Ratio Normalization:** The final output canvas must be strictly locked to **1:1 (Square)** to optimize for cross-platform playback.
*   **Media Fitting:** Mismatched aspect ratios must be pillar-boxed or letter-boxed with a solid white or black background. No stretching or cropping is permitted.
*   **Stitching & Transitions:** Selected media must be stitched chronologically with simple, non-intrusive transitions (e.g., cuts or crossfades)[cite: 1].
*   **Audio:** The MVP will be strictly silent. Raw video audio tracks must be stripped.
*   **Output Format:** The final video must be saved locally as a standard MP4 file.

---

## 5. Fast-Follow Requirements (Phase 1)

*   **Iterative Generation:** The application must generate 2 to 3 candidate output videos per query. The user must be able to provide iterative feedback on these candidates.
*   **Identity Management (Face Tagging):** 
    *   The user can add photos of people and tag them with names[cite: 1].
    *   The app has to find relevant photos with these people and generate video with them[cite: 1].

---

## 6. Polish & Expansion (Phase 2)

*   **Audio Integration:** The system will introduce audio support by overlaying a background song snippet that semantically matches the tone of the user's prompt and the generated visual content.

---

## 7. Technical Handoff Notes for LLM Agent
*   **Architecture Focus:** Design the technical architecture to fulfill **Phase 0 (MVP)** using local, offline models compatible with the M5 processor's hardware acceleration.
*   **Key Engineering Challenges to Solve:** 
    1.  Selecting lightweight, multi-modal embedding models capable of understanding abstract concepts ("bonding", "epic") across highly varied media formats.
    2.  Building efficient local media decoding/encoding pipelines that handle mixed frame rates, resolutions, and codecs without crashing.
    3.  Memory management when parsing and scoring a heterogeneous 20 GB corpus.