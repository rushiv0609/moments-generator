/**
 * Local AI Moments Generator - Living Debug UI Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initPlayground();
    initWorkspace();
    initMediaExplorer();
    initDirectorStudio();
    fetchHealth();
    fetchConfig();
    fetchDataDir();

    // Event listeners
    document.getElementById("btnRefreshHealth").addEventListener("click", () => {
        fetchHealth(true);
    });

    document.getElementById("btnRefreshData").addEventListener("click", () => {
        fetchDataDir();
    });

    document.getElementById("configSearch").addEventListener("input", (e) => {
        filterConfigTable(e.target.value);
    });

    const btnScan = document.getElementById("btnScanCorpus");
    if (btnScan) {
        btnScan.addEventListener("click", () => {
            scanCorpusUI();
        });
    }

    const btnIndexWs = document.getElementById("btnIndexWorkspace");
    if (btnIndexWs) {
        btnIndexWs.addEventListener("click", () => {
            startIndexingJob();
        });
    }

    const btnSubmitIdx = document.getElementById("btnSubmitIndex");
    if (btnSubmitIdx) {
        btnSubmitIdx.addEventListener("click", () => {
            startIndexingJob();
        });
    }

    // Auto-refresh health every 15 seconds
    setInterval(() => {
        fetchHealth(false);
        fetchWorkspace(false);
    }, 15000);
});

/**
 * Project Workspace Management & Native Finder Integration
 */
function initWorkspace() {
    fetchWorkspace(true);

    const btnBrowseWs = document.getElementById("btnBrowseWorkspace");
    if (btnBrowseWs) {
        btnBrowseWs.addEventListener("click", () => {
            browseFolder("workspacePath", "Choose or Create Project Workspace Folder");
        });
    }

    const btnBrowseCorpus = document.getElementById("btnBrowseCorpus");
    if (btnBrowseCorpus) {
        btnBrowseCorpus.addEventListener("click", () => {
            browseFolder("corpusPath", "Choose Source Photos & Videos Folder");
        });
    }

    const btnApply = document.getElementById("btnApplyWorkspace");
    if (btnApply) {
        btnApply.addEventListener("click", () => {
            applyWorkspace();
        });
    }
}

async function browseFolder(targetInputId, promptText) {
    const targetInput = document.getElementById(targetInputId);
    const currentVal = targetInput ? targetInput.value.trim() : null;

    try {
        const res = await fetch("/api/v1/workspace/select-folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: promptText, default_path: currentVal || null }),
        });
        if (res.ok) {
            const data = await res.json();
            if (data.selected_path && targetInput) {
                targetInput.value = data.selected_path;
                if (targetInputId === "workspacePath") {
                    // Automatically activate selected workspace
                    applyWorkspace(false);
                }
            }
        }
    } catch (err) {
        console.error("Failed to open Finder picker:", err);
    }
}

async function fetchWorkspace(updateInputs = false) {
    try {
        const res = await fetch("/api/v1/workspace/current");
        if (res.ok) {
            const data = await res.json();
            const wsInput = document.getElementById("workspacePath");
            const corpusInput = document.getElementById("corpusPath");
            const indexedCount = document.getElementById("wsIndexedCount");
            const vectorCount = document.getElementById("wsVectorCount");

            if (updateInputs && wsInput && data.workspace_dir) {
                wsInput.value = data.workspace_dir;
            }
            if (updateInputs && corpusInput && data.corpus_dir) {
                corpusInput.value = data.corpus_dir;
            }
            if (indexedCount) indexedCount.textContent = data.indexed_files;
            if (vectorCount) vectorCount.textContent = data.total_vectors;
        }
    } catch (err) {
        console.debug("Could not fetch current workspace:", err);
    }
}

async function applyWorkspace(showAlert = true) {
    const wsInput = document.getElementById("workspacePath");
    const corpusInput = document.getElementById("corpusPath");
    const btnApply = document.getElementById("btnApplyWorkspace");

    if (!wsInput || !wsInput.value.trim()) {
        if (showAlert) alert("Please specify a Project Workspace Directory path.");
        return;
    }

    if (btnApply) {
        btnApply.disabled = true;
        btnApply.textContent = "⏳ Activating...";
    }

    try {
        const res = await fetch("/api/v1/workspace/set", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                workspace_path: wsInput.value.trim(),
                corpus_path: corpusInput ? corpusInput.value.trim() || null : null,
            }),
        });

        if (res.ok) {
            const data = await res.json();
            if (showAlert) alert(`✅ Project Workspace Activated:\n${data.workspace_dir}`);
            fetchWorkspace(false);
            fetchHealth(true);
        } else {
            const err = await res.json();
            if (showAlert) alert(`Failed to activate workspace: ${err.detail || res.statusText}`);
        }
    } catch (err) {
        if (showAlert) alert(`Error connecting to server: ${err.message}`);
    } finally {
        if (btnApply) {
            btnApply.disabled = false;
            btnApply.textContent = "💾 Save & Switch Project";
        }
    }
}

/**
 * Milestone 8: Parallel Ingestion Pipeline Execution & SSE Progress Stream
 */
let _activeEventSource = null;

async function startIndexingJob() {
    const wsInput = document.getElementById("workspacePath");
    const corpusInput = document.getElementById("corpusPath");
    const forceReindex = document.getElementById("forceReindex") ? document.getElementById("forceReindex").checked : false;

    const btnIndexWs = document.getElementById("btnIndexWorkspace");
    const btnSubmitIdx = document.getElementById("btnSubmitIndex");
    const progressCard = document.getElementById("pipelineProgressCard");
    const stageBadge = document.getElementById("pipelineStageBadge");
    const statusText = document.getElementById("pipelineStatusText");
    const pctText = document.getElementById("pipelinePctText");
    const progressBar = document.getElementById("pipelineProgressBar");
    const logTerminal = document.getElementById("pipelineLogTerminal");

    const wsPath = wsInput ? wsInput.value.trim() : null;
    const corpusPath = corpusInput ? corpusInput.value.trim() : null;

    if (!wsPath) {
        alert("Please specify or choose a Project Workspace Directory first.");
        return;
    }

    if (btnIndexWs) btnIndexWs.disabled = true;
    if (btnSubmitIdx) btnSubmitIdx.disabled = true;

    // Show live progress card
    if (progressCard) progressCard.style.display = "block";
    if (logTerminal) logTerminal.innerHTML = `<div>[${new Date().toLocaleTimeString()}] Submitting background indexing job...</div>`;

    // Close any previous SSE connection
    if (_activeEventSource) {
        _activeEventSource.close();
        _activeEventSource = null;
    }

    try {
        const res = await fetch("/api/v1/jobs/index", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                workspace_path: wsPath,
                corpus_path: corpusPath || null,
                force_reindex: forceReindex,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const jobId = data.job_id;
        if (logTerminal) {
            logTerminal.innerHTML += `<div>[${new Date().toLocaleTimeString()}] Job ${jobId} registered. Connecting live SSE stream...</div>`;
        }

        // Open SSE connection
        let isIndexingFinished = false;
        _activeEventSource = new EventSource(`/api/v1/jobs/${jobId}/events`);

        _activeEventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.stage === "COMPLETED" || payload.event_type === "completed") {
                    isIndexingFinished = true;
                }
                updatePipelineUI(payload);
            } catch (e) {
                console.debug("SSE Parse notice:", e);
            }
        };

        _activeEventSource.addEventListener("progress", (event) => {
            const payload = JSON.parse(event.data);
            updatePipelineUI(payload);
        });

        _activeEventSource.addEventListener("completed", (event) => {
            isIndexingFinished = true;
            const payload = JSON.parse(event.data);
            updatePipelineUI(payload);
            if (_activeEventSource) {
                _activeEventSource.close();
                _activeEventSource = null;
            }
            if (btnIndexWs) btnIndexWs.disabled = false;
            if (btnSubmitIdx) btnSubmitIdx.disabled = false;
            fetchWorkspace(false);
            fetchHealth(true);
            fetchDataDir();
        });

        _activeEventSource.addEventListener("error", async (event) => {
            if (_activeEventSource) {
                _activeEventSource.close();
                _activeEventSource = null;
            }
            if (btnIndexWs) btnIndexWs.disabled = false;
            if (btnSubmitIdx) btnSubmitIdx.disabled = false;

            if (isIndexingFinished) return;

            // Double check actual server status before marking failed
            try {
                const jobResp = await fetch(`/api/v1/jobs/${jobId}`);
                if (jobResp.ok) {
                    const jobData = await jobResp.json();
                    if (jobData.status === "COMPLETED") {
                        isIndexingFinished = true;
                        updatePipelineUI({
                            stage: "COMPLETED",
                            progress_pct: 100.0,
                            message: jobData.message || "Indexing completed successfully.",
                            data: jobData.summary || {},
                        });
                        fetchWorkspace(false);
                        fetchHealth(true);
                        fetchDataDir();
                        return;
                    }
                }
            } catch (e) {}

            if (stageBadge && !isIndexingFinished) {
                stageBadge.className = "badge badge-danger";
                stageBadge.textContent = "FAILED";
            }
        });

    } catch (err) {
        alert(`Failed to start indexing job: ${err.message}`);
        if (btnIndexWs) btnIndexWs.disabled = false;
        if (btnSubmitIdx) btnSubmitIdx.disabled = false;
    }
}

function updatePipelineUI(event) {
    const stageBadge = document.getElementById("pipelineStageBadge");
    const statusText = document.getElementById("pipelineStatusText");
    const pctText = document.getElementById("pipelinePctText");
    const progressBar = document.getElementById("pipelineProgressBar");
    const logTerminal = document.getElementById("pipelineLogTerminal");
    const fpsText = document.getElementById("pipeFps");
    const countText = document.getElementById("pipeCount");
    const elapsedText = document.getElementById("pipeElapsed");

    const stage = event.stage || "RUNNING";
    const pct = event.progress_pct !== undefined ? event.progress_pct : 0.0;
    const msg = event.message || "";

    if (stageBadge) {
        stageBadge.textContent = stage;
        stageBadge.className = stage === "COMPLETED" ? "badge badge-success" : (stage === "FAILED" ? "badge badge-danger" : "badge badge-warning");
    }

    if (statusText) statusText.textContent = msg;
    if (pctText) pctText.textContent = `${pct.toFixed(0)}%`;
    if (progressBar) progressBar.style.width = `${pct}%`;

    // Update Stage Chips
    const chips = ["step-scan", "step-decode", "step-embed", "step-index"];
    chips.forEach(c => {
        const el = document.getElementById(c);
        if (el) el.style.border = "1px solid transparent";
    });

    if (stage === "SCANNING") highlightStep("step-scan");
    else if (stage === "EXTRACTING") highlightStep("step-decode");
    else if (stage === "EMBEDDING") highlightStep("step-embed");
    else if (stage === "INDEXING") highlightStep("step-index");
    else if (stage === "COMPLETED") {
        chips.forEach(c => {
            const el = document.getElementById(c);
            if (el) {
                el.style.background = "rgba(16, 185, 129, 0.15)";
                el.style.color = "#10b981";
            }
        });
    }

    if (event.data) {
        const d = event.data;
        if (fpsText && d.throughput_fps !== undefined) fpsText.textContent = d.throughput_fps;
        if (countText && d.processed_count !== undefined && d.total_count !== undefined) countText.textContent = `${d.processed_count} / ${d.total_count}`;
        if (elapsedText && d.elapsed_seconds !== undefined) elapsedText.textContent = `${d.elapsed_seconds}s`;
    }

    if (logTerminal && msg) {
        const line = document.createElement("div");
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logTerminal.appendChild(line);
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }
}

function highlightStep(stepId) {
    const el = document.getElementById(stepId);
    if (el) {
        el.style.border = "1px solid #6366f1";
        el.style.background = "rgba(99, 102, 241, 0.2)";
    }
}



/**
 * Scan Corpus UI Handler (Milestone 4)
 */
async function scanCorpusUI() {
    const wsInput = document.getElementById("workspacePath");
    const workspacePath = wsInput ? wsInput.value.trim() : null;
    const corpusPath = document.getElementById("corpusPath").value.trim();
    const forceReindex = document.getElementById("forceReindex").checked;
    const btnScan = document.getElementById("btnScanCorpus");
    const resultsBox = document.getElementById("scanResultsBox");
    const tbody = document.getElementById("scanTableBody");

    if (!corpusPath) {
        alert("Please specify a corpus directory path.");
        return;
    }

    btnScan.disabled = true;
    btnScan.textContent = "⏳ Scanning Corpus...";
    resultsBox.style.display = "block";
    tbody.innerHTML = `<tr><td colspan="4" class="text-center">Traversing directory and extracting EXIF/container metadata...</td></tr>`;

    try {
        const res = await fetch("/api/v1/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                corpus_path: corpusPath,
                workspace_path: workspacePath || null,
                force_reindex: forceReindex,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        document.getElementById("scanTotal").textContent = data.total_found;
        document.getElementById("scanPhotos").textContent = `${data.images_count} photos`;
        document.getElementById("scanVideos").textContent = `${data.videos_count} clips`;
        fetchWorkspace(false);

        tbody.innerHTML = "";
        if (!data.files || data.files.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--text-muted)">No supported media files found in directory.</td></tr>`;
        } else {
            data.files.slice(0, 100).forEach(f => {
                const tr = document.createElement("tr");
                const icon = f.file_type === "image" ? "📸" : "🎬";
                const filename = f.file_path.split("/").pop();
                let dateStr = "-";
                if (f.creation_timestamp) {
                    dateStr = new Date(f.creation_timestamp * 1000).toISOString().replace("T", " ").split(".")[0];
                }
                tr.innerHTML = `
                    <td>${icon} ${f.file_type.toUpperCase()}</td>
                    <td title="${f.file_path}"><code>${filename}</code></td>
                    <td>${dateStr} <small style="color: var(--text-muted)">(${f.timestamp_source || 'fs'})</small></td>
                    <td><span class="badge badge-accent">${f.status}</span></td>
                `;
                tbody.appendChild(tr);
            });
            if (data.files.length > 100) {
                const trMore = document.createElement("tr");
                trMore.innerHTML = `<td colspan="4" class="text-center" style="color: var(--text-muted)">... and ${data.files.length - 100} more files</td>`;
                tbody.appendChild(trMore);
            }
        }
        fetchDataDir();
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--danger)">Scan failed: ${err.message}</td></tr>`;
    } finally {
        btnScan.disabled = false;
        btnScan.textContent = "🔍 Scan Corpus (Milestone 4)";
    }
}

/**
 * Tab Navigation
 */
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(`pane-${tabId}`);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });
}

/**
 * Health Check API
 */
async function fetchHealth(showFeedback = false) {
    const pillModel = document.getElementById("pillModel");
    const pillQdrant = document.getElementById("pillQdrant");
    const pillFFmpeg = document.getElementById("pillFFmpeg");

    try {
        const res = await fetch("/api/v1/health");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // 1. Model Pill & Card
        const modelOk = data.model && data.model.details;
        updatePill(pillModel, modelOk ? "dot-healthy" : "dot-degraded", `Model: ${data.model.backend.toUpperCase()}`);
        document.getElementById("modelDetails").textContent = `${data.model.name} (${data.model.details})`;
        setTag("modelTag", modelOk ? "tag-healthy" : "tag-degraded", data.model.backend.toUpperCase());

        // 2. Qdrant Pill & Card
        const qdrantOk = data.qdrant && data.qdrant.connected;
        updatePill(pillQdrant, qdrantOk ? "dot-healthy" : "dot-unhealthy", `Qdrant: ${qdrantOk ? "Connected" : "Disconnected"}`);
        document.getElementById("qdrantDetails").textContent = `${data.qdrant.host}:${data.qdrant.port} (${data.qdrant.collections} collections)`;
        setTag("qdrantTag", qdrantOk ? "tag-healthy" : "tag-unhealthy", qdrantOk ? "ONLINE" : "OFFLINE");

        // 3. FFmpeg Pill & Card
        const ffmpegOk = data.ffmpeg && data.ffmpeg.available;
        updatePill(pillFFmpeg, ffmpegOk ? "dot-healthy" : "dot-degraded", `FFmpeg: ${ffmpegOk ? "Ready" : "Missing"}`);
        document.getElementById("ffmpegDetails").textContent = data.ffmpeg.details || (ffmpegOk ? "Available" : "Not Found");
        setTag("ffmpegTag", ffmpegOk ? "tag-healthy" : "tag-degraded", ffmpegOk ? "READY" : "OPTIONAL");

        // 4. Memory Telemetry
        if (data.system_memory && data.system_memory.total_gb) {
            const mem = data.system_memory;
            document.getElementById("memAvailable").textContent = `Available: ${mem.available_gb} GB`;
            document.getElementById("memTotal").textContent = `Total: ${mem.total_gb} GB (${mem.used_pct}% used)`;
            document.getElementById("memProgressFill").style.width = `${mem.used_pct}%`;
        }

        if (showFeedback) {
            showToast("System health updated");
        }
    } catch (err) {
        console.error("Health fetch error:", err);
        updatePill(pillModel, "dot-unhealthy", "Server Offline");
        updatePill(pillQdrant, "dot-unhealthy", "Server Offline");
        updatePill(pillFFmpeg, "dot-unhealthy", "Server Offline");
    }
}

function updatePill(pill, dotClass, labelText) {
    if (!pill) return;
    const dot = pill.querySelector(".status-dot");
    const label = pill.querySelector(".pill-label");
    if (dot) {
        dot.className = `status-dot ${dotClass}`;
    }
    if (label) {
        label.textContent = labelText;
    }
}

function setTag(elementId, tagClass, text) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.className = `status-tag ${tagClass}`;
    el.textContent = text;
}

/**
 * Config Debug API
 */
let cachedConfig = {};

async function fetchConfig() {
    const tbody = document.getElementById("configTableBody");
    try {
        const res = await fetch("/api/v1/debug/config");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        cachedConfig = await res.json();
        renderConfigTable(cachedConfig);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="3" class="text-center" style="color: var(--danger)">Error loading config: ${err.message}</td></tr>`;
    }
}

const CONFIG_DESCRIPTIONS = {
    HOST: "Bind host for FastAPI server",
    PORT: "Port for FastAPI server",
    DEBUG: "Enable debug logging and hot reloading",
    MODEL_NAME: "Hugging Face vision-language embedding model ID",
    MODEL_BACKEND: "Inference engine: 'auto' selects MLX on Apple Silicon, fallback to PyTorch MPS",
    MODEL_PRECISION: "Weight precision: 'fp16' or '8bit' (MLX 50% RAM reduction)",
    EMBEDDING_RESOLUTION: "Input image dimension for SigLIP 2 vision transformer (224x224)",
    EMBED_BATCH_SIZE: "Batch size for GPU forward pass (64 saturates M5 Pro GPU)",
    EXTRACT_WORKERS: "Parallel threads for Apple ImageIO hardware photo decoding (12 optimal)",
    INDEX_BATCH_SIZE: "Batch size for vector upserts to Qdrant",
    MIN_SIMILARITY_THRESHOLD: "Cosine similarity threshold for semantic candidate retrieval",
    MAX_OUTPUT_DURATION: "Maximum generated video length in seconds",
    DEFAULT_ASPECT_RATIO: "Default output video framing (1:1, 16:9, or 9:16)",
    IMAGE_DISPLAY_DURATION: "Display duration for photo slides in the rendered video (seconds)",
    VIDEO_SEGMENT_DURATION: "Duration for video clip segments in the rendered video (seconds)",
    QDRANT_HOST: "Hostname for local Qdrant container",
    QDRANT_PORT: "REST port for Qdrant vector database",
    QDRANT_COLLECTION: "Qdrant collection name for media embeddings",
    DATA_DIR: "Local working directory for manifest SQLite databases and cache",
    EXPORTS_DIR: "Output directory for rendered video files",
    MODELS_DIR: "Local directory for cached model weights",
    VIDEO_CODEC: "FFmpeg hardware encoder (h264_videotoolbox on macOS)",
    VIDEO_BITRATE: "Target bitrate for rendered MP4 output",
    VIDEO_FPS: "Frame rate for rendered MP4 output",
};

function renderConfigTable(configObj) {
    const tbody = document.getElementById("configTableBody");
    tbody.innerHTML = "";

    const keys = Object.keys(configObj).sort();
    keys.forEach(k => {
        const val = configObj[k];
        const desc = CONFIG_DESCRIPTIONS[k] || "Configuration parameter";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong style="color: var(--primary)">${k}</strong></td>
            <td><code>${JSON.stringify(val)}</code></td>
            <td style="color: var(--text-secondary)">${desc}</td>
        `;
        tbody.appendChild(tr);
    });
}

function filterConfigTable(query) {
    const q = query.toLowerCase().trim();
    if (!q) {
        renderConfigTable(cachedConfig);
        return;
    }
    const filtered = {};
    for (const [k, v] of Object.entries(cachedConfig)) {
        if (k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q)) {
            filtered[k] = v;
        }
    }
    renderConfigTable(filtered);
}

/**
 * Data Directory Browser API
 */
async function fetchDataDir() {
    const tbody = document.getElementById("dataTableBody");
    const rootLabel = document.getElementById("dataDirRoot");
    tbody.innerHTML = `<tr><td colspan="4" class="text-center">Scanning directory...</td></tr>`;

    try {
        const res = await fetch("/api/v1/debug/data");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        rootLabel.textContent = `Root: ${data.data_dir}`;

        if (!data.items || data.items.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--text-muted)">Directory is empty (no manifests or cache files created yet)</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        data.items.forEach(item => {
            const tr = document.createElement("tr");
            const icon = item.is_dir ? "📁" : "📄";
            const sizeStr = item.is_dir ? "-" : formatBytes(item.size_bytes);
            const dateStr = item.modified_at ? item.modified_at.replace("T", " ").split(".")[0] : "-";
            tr.innerHTML = `
                <td>${icon} ${item.is_dir ? "DIR" : "FILE"}</td>
                <td><code>${item.path}</code></td>
                <td>${sizeStr}</td>
                <td>${dateStr}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="4" class="text-center" style="color: var(--danger)">Error browsing data directory: ${err.message}</td></tr>`;
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function showToast(message) {
    console.log("[Toast]", message);
}

/**
 * =========================================================================
 * Semantic Playground Interactive Controller
 * =========================================================================
 */
let targetState = {
    A: { type: "image", file: null, text: "" },
    B: { type: "text", file: null, text: "" },
};

function initPlayground() {
    setupToggleButtons();
    setupDropzones();
    setupChips();
    setupGlobalPaste();

    const btnCalc = document.getElementById("btnCalculateSimilarity");
    if (btnCalc) {
        btnCalc.addEventListener("click", calculateSimilarity);
    }

    const btnReset = document.getElementById("btnResetPlayground");
    if (btnReset) {
        btnReset.addEventListener("click", resetPlayground);
    }

    const btnClearA = document.getElementById("btnClearImgA");
    if (btnClearA) {
        btnClearA.addEventListener("click", (e) => {
            e.stopPropagation();
            clearImageTarget("A");
        });
    }

    const btnClearB = document.getElementById("btnClearImgB");
    if (btnClearB) {
        btnClearB.addEventListener("click", (e) => {
            e.stopPropagation();
            clearImageTarget("B");
        });
    }
}

function setupToggleButtons() {
    document.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const target = btn.getAttribute("data-target");
            const type = btn.getAttribute("data-type");

            // Update state
            targetState[target].type = type;

            // Update UI toggle buttons
            const card = document.getElementById(`cardTarget${target}`);
            card.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Switch visibility
            const dropzone = document.getElementById(`dropzone${target}`);
            const textContainer = document.getElementById(`textContainer${target}`);

            if (type === "image") {
                dropzone.style.display = "block";
                textContainer.style.display = "none";
            } else {
                dropzone.style.display = "none";
                textContainer.style.display = "block";
            }
        });
    });
}

function setupDropzones() {
    ["A", "B"].forEach(t => {
        const dropArea = document.getElementById(`dropArea${t}`);
        const fileInput = document.getElementById(`fileInput${t}`);

        if (!dropArea || !fileInput) return;

        // Click to browse
        dropArea.addEventListener("click", () => {
            if (!targetState[t].file) {
                fileInput.click();
            }
        });

        // File input changed
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelected(t, e.target.files[0]);
            }
        });

        // Drag & Drop
        ["dragenter", "dragover"].forEach(eventName => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.remove("dragover");
            }, false);
        });

        dropArea.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelected(t, dt.files[0]);
            }
        }, false);
    });
}

function handleFileSelected(target, file) {
    targetState[target].file = file;

    const placeholder = document.getElementById(`placeholder${target}`);
    const previewContainer = document.getElementById(`previewContainer${target}`);
    const previewImg = document.getElementById(`previewImg${target}`);
    const previewMeta = document.getElementById(`previewMeta${target}`);

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        placeholder.style.display = "none";
        previewContainer.style.display = "flex";
        const sizeStr = formatBytes(file.size);
        previewMeta.textContent = `${file.name} (${sizeStr})`;
    };
    reader.readAsDataURL(file);
}

function clearImageTarget(target) {
    targetState[target].file = null;
    const fileInput = document.getElementById(`fileInput${target}`);
    if (fileInput) fileInput.value = "";

    const placeholder = document.getElementById(`placeholder${target}`);
    const previewContainer = document.getElementById(`previewContainer${target}`);
    const previewImg = document.getElementById(`previewImg${target}`);

    if (previewImg) previewImg.src = "";
    if (previewContainer) previewContainer.style.display = "none";
    if (placeholder) placeholder.style.display = "block";
}

function setupChips() {
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const target = chip.getAttribute("data-target");
            const text = chip.textContent.replace(/^[^\w\s]+/, "").trim(); // strip emoji
            const textarea = document.getElementById(`textInput${target}`);
            if (textarea) {
                textarea.value = `a photo of ${text}`;
            }
        });
    });
}

/**
 * Global & Dropzone Clipboard Paste Listener (Cmd+V)
 */
function setupGlobalPaste() {
    window.addEventListener("paste", (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf("image") !== -1) {
                const blob = items[i].getAsFile();
                if (blob) {
                    // Determine which target to paste into:
                    // If target A is currently 'image' and empty, use A; otherwise if B is 'image', use B; otherwise default to A
                    let target = "A";
                    if (targetState.A.type === "image" && !targetState.A.file) {
                        target = "A";
                    } else if (targetState.B.type === "image" && !targetState.B.file) {
                        target = "B";
                    } else if (targetState.A.type === "image") {
                        target = "A";
                    }

                    handleFileSelected(target, blob);
                    showToast(`Pasted image into Target ${target}`);
                    break;
                }
            }
        }
    });
}

function resetPlayground() {
    clearImageTarget("A");
    clearImageTarget("B");
    const textA = document.getElementById("textInputA");
    if (textA) textA.value = "";
    const textB = document.getElementById("textInputB");
    if (textB) textB.value = "";

    const resultPanel = document.getElementById("resultPanel");
    if (resultPanel) resultPanel.style.display = "none";
}

/**
 * Fetch embeddings and calculate Cosine Similarity
 */
async function calculateSimilarity() {
    const btnCalc = document.getElementById("btnCalculateSimilarity");
    const resultPanel = document.getElementById("resultPanel");
    const scoreVal = document.getElementById("resultScoreValue");
    const statusBadge = document.getElementById("resultStatusBadge");
    const interp = document.getElementById("resultInterpretation");
    const gaugeFill = document.getElementById("gaugeBarFill");
    const latencyEl = document.getElementById("calcLatency");
    const backendEl = document.getElementById("calcBackend");

    // Gather inputs
    const typeA = targetState.A.type;
    const typeB = targetState.B.type;

    let textA = typeA === "text" ? document.getElementById("textInputA").value.trim() : null;
    let fileA = typeA === "image" ? targetState.A.file : null;

    let textB = typeB === "text" ? document.getElementById("textInputB").value.trim() : null;
    let fileB = typeB === "image" ? targetState.B.file : null;

    if (typeA === "text" && !textA) {
        alert("Please enter a text prompt for Target A.");
        return;
    }
    if (typeA === "image" && !fileA) {
        alert("Please upload or paste an image for Target A.");
        return;
    }
    if (typeB === "text" && !textB) {
        alert("Please enter a text prompt for Target B.");
        return;
    }
    if (typeB === "image" && !fileB) {
        alert("Please upload or paste an image for Target B.");
        return;
    }

    btnCalc.disabled = true;
    btnCalc.textContent = "⏳ Computing Fused Embeddings...";
    resultPanel.style.display = "block";
    statusBadge.className = "badge badge-muted";
    statusBadge.textContent = "Processing...";
    scoreVal.textContent = "0.0000";
    interp.textContent = "Calculating dot product...";

    const startTime = performance.now();

    try {
        let vecA = null;
        let vecB = null;
        let backendName = "Apple MLX";

        // If Image-Text pair, we can compute in 1 shot or separately
        if (typeA === "image" && typeB === "text") {
            const formData = new FormData();
            formData.append("file", fileA);
            formData.append("text", textB);

            const res = await fetch("/api/v1/debug/embed", {
                method: "POST",
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            vecA = data.image_embedding;
            vecB = data.text_embedding;
            if (data.model_info && data.model_info.backend) {
                backendName = data.model_info.backend.toUpperCase();
            }
        } else if (typeA === "text" && typeB === "image") {
            const formData = new FormData();
            formData.append("file", fileB);
            formData.append("text", textA);

            const res = await fetch("/api/v1/debug/embed", {
                method: "POST",
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            vecA = data.text_embedding;
            vecB = data.image_embedding;
            if (data.model_info && data.model_info.backend) {
                backendName = data.model_info.backend.toUpperCase();
            }
        } else {
            // Text vs Text or Image vs Image: fetch separately
            const [resA, resB] = await Promise.all([
                fetchEmbed(typeA, fileA, textA),
                fetchEmbed(typeB, fileB, textB),
            ]);
            vecA = resA.vec;
            vecB = resB.vec;
            if (resA.backend) backendName = resA.backend.toUpperCase();
        }

        const endTime = performance.now();
        const latencyMs = Math.round(endTime - startTime);

        // Compute exact dot product of normalized vectors
        const sim = dotProduct(vecA, vecB);
        const simFormatted = (sim >= 0 ? "+" : "") + sim.toFixed(4);

        scoreVal.textContent = simFormatted;
        latencyEl.textContent = `${latencyMs} ms`;
        backendEl.textContent = backendName;

        // Interpret score on SigLIP 2 scale
        if (sim >= 0.08) {
            scoreVal.style.color = "#10b981"; // Emerald green
            statusBadge.className = "badge badge-success";
            statusBadge.textContent = "🟢 Strong Match";
            interp.textContent = "High semantic alignment. Concepts are closely bound.";
            interp.style.color = "#10b981";
        } else if (sim >= 0.03) {
            scoreVal.style.color = "#f59e0b"; // Amber
            statusBadge.className = "badge badge-accent";
            statusBadge.textContent = "🟡 Moderate Match";
            interp.textContent = "Partial semantic relation or broad background context.";
            interp.style.color = "#f59e0b";
        } else if (sim >= 0.0) {
            scoreVal.style.color = "var(--text-secondary)";
            statusBadge.className = "badge badge-muted";
            statusBadge.textContent = "⚪️ Weak / Neutral";
            interp.textContent = "Low correlation between targets.";
            interp.style.color = "var(--text-muted)";
        } else {
            scoreVal.style.color = "#ef4444"; // Red
            statusBadge.className = "badge badge-muted";
            statusBadge.textContent = "🔴 Distractor / Negative";
            interp.textContent = "Semantic contradiction or unrelated concepts.";
            interp.style.color = "#ef4444";
        }

        // Animate gauge fill: scale [-0.15, +0.15] to [0%, 100%]
        const minScale = -0.15;
        const maxScale = 0.15;
        const clamped = Math.max(minScale, Math.min(maxScale, sim));
        const pct = ((clamped - minScale) / (maxScale - minScale)) * 100;

        const zeroPct = ((0 - minScale) / (maxScale - minScale)) * 100;

        if (sim >= 0) {
            gaugeFill.style.left = `${zeroPct}%`;
            gaugeFill.style.width = `${Math.max(2, pct - zeroPct)}%`;
            gaugeFill.style.background = sim >= 0.08 ? "#10b981" : "#f59e0b";
        } else {
            gaugeFill.style.left = `${pct}%`;
            gaugeFill.style.width = `${zeroPct - pct}%`;
            gaugeFill.style.background = "#ef4444";
        }
    } catch (err) {
        console.error("Similarity calculation error:", err);
        scoreVal.textContent = "ERR";
        statusBadge.className = "badge badge-muted";
        statusBadge.textContent = "Calculation Failed";
        interp.textContent = err.message;
        interp.style.color = "#ef4444";
    } finally {
        btnCalc.disabled = false;
        btnCalc.textContent = "⚡ Calculate Semantic Similarity";
    }
}

async function fetchEmbed(type, file, text) {
    const formData = new FormData();
    if (type === "image" && file) {
        formData.append("file", file);
    } else if (type === "text" && text) {
        formData.append("text", text);
    }

    const res = await fetch("/api/v1/debug/embed", {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    return {
        vec: type === "image" ? data.image_embedding : data.text_embedding,
        backend: data.model_info ? data.model_info.backend : "mlx",
    };
}

function dotProduct(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

/* =========================================================================
   Media Search Explorer (Active Workspace)
   ========================================================================= */
function initMediaExplorer() {
    const inputQuery = document.getElementById("explorerQueryInput");
    const btnSearch = document.getElementById("btnRunExplorerSearch");
    const selGranularity = document.getElementById("explorerGranularity");
    const selFileType = document.getElementById("explorerFileType");
    const selTopK = document.getElementById("explorerTopK");
    const emptyState = document.getElementById("explorerEmptyState");
    const grid = document.getElementById("explorerMediaGrid");
    const resultsHeading = document.getElementById("explorerResultsHeading");
    const resultsCount = document.getElementById("explorerResultsCount");
    const statsEl = document.getElementById("explorerSearchStats");

    if (!btnSearch || !inputQuery) return;

    // Search button click
    btnSearch.addEventListener("click", () => {
        executeWorkspaceSearch();
    });

    // Enter key
    inputQuery.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            executeWorkspaceSearch();
        }
    });

    // Quick chip buttons
    document.querySelectorAll(".chip-btn").forEach((chip) => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            if (q) {
                inputQuery.value = q;
                executeWorkspaceSearch();
            }
        });
    });

    // Re-filter when dropdown changes
    if (selGranularity) selGranularity.addEventListener("change", () => executeWorkspaceSearch());
    if (selFileType) selFileType.addEventListener("change", () => executeWorkspaceSearch());
    if (selTopK) selTopK.addEventListener("change", () => executeWorkspaceSearch());

    async function executeWorkspaceSearch() {
        const query = inputQuery.value.trim();
        if (!query) {
            inputQuery.focus();
            return;
        }

        const granularity = selGranularity ? selGranularity.value : "all";
        const fileType = selFileType ? selFileType.value : "all";
        const topK = selTopK ? selTopK.value : 16;

        btnSearch.disabled = true;
        btnSearch.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border: 2px solid #fff; border-top-color: transparent; border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite;"></span> <span>Searching...</span>`;

        const startTime = performance.now();

        try {
            const params = new URLSearchParams({
                query: query,
                top_k: topK,
                granularity: granularity,
                file_type: fileType,
            });

            const res = await fetch(`/api/v1/workspace/search?${params.toString()}`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            const elapsed = Math.round(performance.now() - startTime);

            renderExplorerResults(data, elapsed);
        } catch (err) {
            console.error("Workspace search error:", err);
            emptyState.style.display = "block";
            grid.style.display = "none";
            emptyState.innerHTML = `
                <div style="font-size: 32px; margin-bottom: 8px;">⚠️</div>
                <h4 style="color: #ef4444;">Search Failed</h4>
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">${err.message}</p>
            `;
            resultsCount.textContent = "Error";
            resultsCount.className = "badge badge-danger";
        } finally {
            btnSearch.disabled = false;
            btnSearch.innerHTML = `<span>⚡ Search Media</span>`;
        }
    }

    function renderExplorerResults(data, elapsedMs) {
        const results = data.results || [];
        resultsCount.textContent = `${results.length} matches`;
        resultsCount.className = results.length > 0 ? "badge badge-success" : "badge badge-secondary";
        resultsHeading.textContent = `🎯 Results for "${data.query}"`;
        statsEl.textContent = `⚡ Search completed in ${elapsedMs}ms across ${results.length} ranked items`;

        if (results.length === 0) {
            emptyState.style.display = "block";
            grid.style.display = "none";
            emptyState.innerHTML = `
                <div style="font-size: 36px; margin-bottom: 10px;">🍃</div>
                <h4>No Matching Media Found</h4>
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                    Try a different query or switch filters to 'All Media' and 'All Granularities'.
                </p>
            `;
            return;
        }

        emptyState.style.display = "none";
        grid.style.display = "grid";
        grid.innerHTML = "";

        results.forEach((item, idx) => {
            const card = document.createElement("div");
            card.className = "explorer-card";

            // Determine badge type
            let badgeClass = "explorer-badge-photo";
            let typeLabel = "Photo";
            let isVideo = item.file_type === "video";
            let offsetSec = item.source_offset ?? (item.scene_start ?? 0.0);

            if (isVideo) {
                if (item.granularity === "scene") {
                    badgeClass = "explorer-badge-scene";
                    typeLabel = `Scene #${item.scene_id ?? 0} (${item.scene_start ?? 0}s–${item.scene_end ?? 0}s)`;
                } else {
                    badgeClass = "explorer-badge-video";
                    typeLabel = `Video @ ${offsetSec.toFixed(1)}s`;
                }
            }

            // Score badge formatting
            const scoreFormatted = item.score > 0 ? `+${item.score.toFixed(3)}` : item.score.toFixed(3);
            const thumbSrc = item.thumbnail_url || item.media_url;

            card.innerHTML = `
                <div class="explorer-thumb-wrap">
                    <img src="${thumbSrc}" class="explorer-thumb" loading="lazy" alt="${item.file_name}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\' fill=\\'%23222\\'><rect width=\\'100\\' height=\\'100\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%23888\\'>No Frame</text></svg>'">
                    <span class="explorer-rank-badge">#${idx + 1}</span>
                    <span class="explorer-score-badge" title="Cosine Similarity Score">${scoreFormatted}</span>
                    ${isVideo ? `<div style="position: absolute; width: 36px; height: 36px; background: rgba(0,0,0,0.65); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; color: #fff; pointer-events: none; border: 1px solid rgba(255,255,255,0.3);">▶️</div>` : ""}
                </div>
                <div class="explorer-card-body">
                    <div class="explorer-file-title" title="${item.file_path}">${item.file_name}</div>
                    <div class="explorer-meta-row">
                        <span class="explorer-badge-type ${badgeClass}">${typeLabel}</span>
                        <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">${item.granularity}</span>
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
                        <button type="button" class="btn-card-action btn-find-similar" data-point-id="${item.point_id}" data-file-name="${item.file_name}" style="flex: 1; padding: 4px 6px; font-size: 10px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 4px; color: #a5b4fc; cursor: pointer;">
                            ✨ Similar
                        </button>
                        ${isVideo ? `
                        <button type="button" class="btn-card-action btn-inspect-scenes" data-file-path="${item.file_path}" data-file-name="${item.file_name}" style="flex: 1; padding: 4px 6px; font-size: 10px; background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 4px; color: #d8b4fe; cursor: pointer;">
                            🎬 Scenes
                        </button>` : ""}
                    </div>
                </div>
            `;

            // Clicking thumbnail opens player or photo
            const thumbWrap = card.querySelector(".explorer-thumb-wrap");
            thumbWrap.style.cursor = "pointer";
            thumbWrap.addEventListener("click", () => {
                if (isVideo) {
                    openVideoPlayerModal(item);
                } else {
                    window.open(item.media_url, "_blank");
                }
            });

            // Find Similar button click
            const btnSim = card.querySelector(".btn-find-similar");
            if (btnSim) {
                btnSim.addEventListener("click", (e) => {
                    e.stopPropagation();
                    executeSimilarSearch(item.point_id, item.file_name);
                });
            }

            // Inspect Scenes button click
            const btnScenes = card.querySelector(".btn-inspect-scenes");
            if (btnScenes) {
                btnScenes.addEventListener("click", (e) => {
                    e.stopPropagation();
                    openSceneBreakdownModal(item.file_path, item.file_name);
                });
            }

            grid.appendChild(card);
        });
    }

    async function executeSimilarSearch(pointId, fileName) {
        btnSearch.disabled = true;
        statsEl.textContent = `Finding visually similar media to "${fileName}"...`;
        const startTime = performance.now();

        try {
            const res = await fetch(`/api/v1/workspace/similar?point_id=${encodeURIComponent(pointId)}&top_k=16`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            const elapsed = Math.round(performance.now() - startTime);

            renderExplorerResults({
                query: `Similar to: ${fileName}`,
                results: data.results,
            }, elapsed);
            resultsHeading.textContent = `✨ Visually & Semantically Similar to "${fileName}"`;
        } catch (err) {
            console.error("Similar search error:", err);
            alert(`Failed to find similar items: ${err.message}`);
        } finally {
            btnSearch.disabled = false;
        }
    }

    async function openSceneBreakdownModal(filePath, fileName) {
        // Remove existing modal if any
        const existing = document.getElementById("mediaScenesModal");
        if (existing) existing.remove();

        const modal = document.createElement("div");
        modal.id = "mediaScenesModal";
        modal.style.cssText = `
            position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(8px);
            z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 20px;
        `;

        modal.innerHTML = `
            <div style="background: #111827; border: 1px solid var(--border-color); border-radius: 16px; max-width: 900px; width: 100%; max-height: 85vh; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7);">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid var(--border-color);">
                    <div>
                        <h3 style="font-size: 16px; margin: 0; color: #fff;">🎬 PySceneDetect Visual Breakdown</h3>
                        <span style="font-size: 12px; color: var(--text-muted);">${fileName}</span>
                    </div>
                    <button id="btnCloseScenesModal" style="background: transparent; border: none; font-size: 20px; color: #9ca3af; cursor: pointer;">✕</button>
                </div>
                <div id="scenesModalBody" style="padding: 20px; overflow-y: auto; flex: 1;">
                    <div style="text-align: center; padding: 30px; color: var(--text-muted);">
                        <div class="spinner" style="width: 24px; height: 24px; border: 3px solid #fff; border-top-color: transparent; border-radius: 50%; margin: 0 auto 12px; animation: spin 0.8s linear infinite;"></div>
                        Loading PySceneDetect scene boundaries & extracted frames...
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        const closeBtn = document.getElementById("btnCloseScenesModal");
        if (closeBtn) closeBtn.addEventListener("click", () => modal.remove());
        modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });

        try {
            const res = await fetch(`/api/v1/workspace/video/scenes?file_path=${encodeURIComponent(filePath)}`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            const bodyEl = document.getElementById("scenesModalBody");
            if (!bodyEl) return;

            if (data.scenes.length === 0) {
                bodyEl.innerHTML = `
                    <div style="text-align: center; padding: 30px; color: var(--text-muted);">
                        <p>No separate scene cuts detected. Entire video was indexed as one scene (${data.total_frames} frames).</p>
                    </div>
                `;
                return;
            }

            let html = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 16px; font-size: 13px;">
                    <span>Detected <strong>${data.total_scenes} Coherent Scenes</strong> (${data.total_frames} 1-FPS frames)</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 16px;">
            `;

            data.scenes.forEach((sc) => {
                // Find all frames belonging to this scene
                const sceneFrames = data.frames.filter(f => f.scene_id === sc.scene_id);

                html += `
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="badge badge-accent" style="font-size: 11px;">Scene #${sc.scene_id}</span>
                                <strong style="font-size: 13px; color: #fff;">${sc.start_sec.toFixed(1)}s – ${sc.end_sec.toFixed(1)}s</strong>
                                <span style="font-size: 11px; color: var(--text-muted);">(${sc.duration_sec}s duration, ${sceneFrames.length} frames)</span>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-secondary btn-play-scene" data-offset="${sc.start_sec}" style="padding: 4px 10px; font-size: 11px;">
                                    ▶️ Play Scene
                                </button>
                                ${sc.point_id ? `
                                <button class="btn btn-primary btn-similar-scene" data-point-id="${sc.point_id}" style="padding: 4px 10px; font-size: 11px; background: linear-gradient(135deg, #6366f1, #8b5cf6);">
                                    ✨ Find Similar
                                </button>` : ""}
                            </div>
                        </div>

                        <!-- Scene Frames Filmstrip -->
                        <div style="display: flex; gap: 8px; overflow-x: auto; padding-bottom: 6px;">
                `;

                sceneFrames.forEach(f => {
                    html += `
                        <div class="scene-filmstrip-frame" data-offset="${f.source_offset}" style="flex-shrink: 0; width: 100px; cursor: pointer; text-align: center;">
                            <div style="aspect-ratio: 4/3; background: #000; border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color);">
                                <img src="${f.thumbnail_url}" style="width: 100%; height: 100%; object-fit: cover;" loading="lazy" alt="Frame @ ${f.source_offset}s">
                            </div>
                            <span style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); display: block; margin-top: 3px;">@ ${f.source_offset}s</span>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;
            });

            html += `</div>`;
            bodyEl.innerHTML = html;

            // Attach play scene listeners
            bodyEl.querySelectorAll(".btn-play-scene").forEach(btn => {
                btn.addEventListener("click", () => {
                    const offset = parseFloat(btn.getAttribute("data-offset") || "0");
                    modal.remove();
                    openVideoPlayerModal({
                        file_path: filePath,
                        file_name: fileName,
                        media_url: `/api/v1/media/file?path=${encodeURIComponent(filePath)}`,
                        source_offset: offset,
                        score: 1.0,
                    });
                });
            });

            // Attach filmstrip frame click listeners
            bodyEl.querySelectorAll(".scene-filmstrip-frame").forEach(el => {
                el.addEventListener("click", () => {
                    const offset = parseFloat(el.getAttribute("data-offset") || "0");
                    modal.remove();
                    openVideoPlayerModal({
                        file_path: filePath,
                        file_name: fileName,
                        media_url: `/api/v1/media/file?path=${encodeURIComponent(filePath)}`,
                        source_offset: offset,
                        score: 1.0,
                    });
                });
            });

            // Attach find similar scene listeners
            bodyEl.querySelectorAll(".btn-similar-scene").forEach(btn => {
                btn.addEventListener("click", () => {
                    const pid = btn.getAttribute("data-point-id");
                    if (pid) {
                        modal.remove();
                        executeSimilarSearch(pid, `${fileName} (Scene)`);
                    }
                });
            });

        } catch (err) {
            console.error("Failed to load scenes:", err);
            const bodyEl = document.getElementById("scenesModalBody");
            if (bodyEl) {
                bodyEl.innerHTML = `
                    <div style="text-align: center; padding: 30px; color: #ef4444;">
                        <h4>Failed to load scenes</h4>
                        <p style="font-size: 12px; color: var(--text-muted);">${err.message}</p>
                    </div>
                `;
            }
        }
    }


    function openVideoPlayerModal(item) {
        // Remove existing modal if any
        const existing = document.getElementById("mediaVideoModal");
        if (existing) existing.remove();

        const offsetSec = item.source_offset ?? (item.scene_start ?? 0.0);
        const modal = document.createElement("div");
        modal.id = "mediaVideoModal";
        modal.style.cssText = `
            position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(8px);
            z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 20px;
        `;

        modal.innerHTML = `
            <div style="background: #111827; border: 1px solid var(--border-color); border-radius: 16px; max-width: 800px; width: 100%; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7);">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--border-color);">
                    <div>
                        <h3 style="font-size: 15px; margin: 0; color: #fff;">${item.file_name}</h3>
                        <span style="font-size: 12px; color: var(--accent-color);">Matched Frame at <strong>${offsetSec.toFixed(1)}s</strong> (Score: +${item.score.toFixed(3)})</span>
                    </div>
                    <button id="btnCloseVideoModal" style="background: transparent; border: none; font-size: 20px; color: #9ca3af; cursor: pointer;">✕</button>
                </div>
                <div style="background: #000; display: flex; justify-content: center; aspect-ratio: 16/9;">
                    <video id="modalVideoPlayer" src="${item.media_url}" controls autoplay style="width: 100%; height: 100%; object-fit: contain;"></video>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        const video = document.getElementById("modalVideoPlayer");
        if (video) {
            video.addEventListener("loadedmetadata", () => {
                video.currentTime = offsetSec;
                video.play().catch(() => {});
            });
        }

        const closeBtn = document.getElementById("btnCloseVideoModal");
        if (closeBtn) {
            closeBtn.addEventListener("click", () => modal.remove());
        }

        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.remove();
        });
    }
}

// =========================================================================
// Milestone 9: LangGraph AI Director Studio Controller
// =========================================================================

let currentDirectorJobData = null;
let activeAlternativeKey = "alt_c_dual";
let directorTimerInterval = null;

function initDirectorStudio() {
    fetchDirectorModels();

    // Suggestion pills
    document.querySelectorAll(".prompt-suggestion").forEach(btn => {
        btn.addEventListener("click", () => {
            const prompt = btn.getAttribute("data-prompt");
            const promptInput = document.getElementById("directorPrompt");
            if (promptInput && prompt) {
                promptInput.value = prompt;
                promptInput.focus();
            }
        });
    });

    // Form submit
    const directorForm = document.getElementById("directorForm");
    if (directorForm) {
        directorForm.addEventListener("submit", (e) => {
            e.preventDefault();
            runDirectorAgent();
        });
    }

    // Alternative Switcher Buttons
    document.querySelectorAll(".alt-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const altKey = btn.getAttribute("data-alt");
            switchAlternativeView(altKey);
        });
    });

    // Inspector toggle and clear
    const btnToggle = document.getElementById("btnToggleInspector");
    if (btnToggle) {
        btnToggle.addEventListener("click", () => {
            const logBox = document.getElementById("inspectorLogContainer");
            const metricsRow = document.getElementById("inspectorMetricsRow");
            if (logBox) {
                const isHidden = logBox.style.display === "none";
                logBox.style.display = isHidden ? "block" : "none";
                if (metricsRow) metricsRow.style.display = isHidden ? "grid" : "none";
                btnToggle.textContent = isHidden ? "Hide Inspector" : "Show Inspector";
            }
        });
    }

    const btnClear = document.getElementById("btnClearInspector");
    if (btnClear) {
        btnClear.addEventListener("click", () => {
            resetDirectorInspector();
        });
    }

    const btnPlayAll = document.getElementById("btnPlayAllStoryboard");
    if (btnPlayAll) {
        btnPlayAll.addEventListener("click", () => {
            openFullStoryboardPlayer();
        });
    }

    const btnRender = document.getElementById("btnRenderFinalVideo");
    if (btnRender) {
        btnRender.addEventListener("click", () => {
            renderActiveStoryboardVideo();
        });
    }

    // LLM Tab switcher
    document.querySelectorAll(".llm-tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const node = btn.getAttribute("data-node");
            renderLlmPromptTab(node);
        });
    });

    initCloudApiKeysUi();
}

// Initialize Cloud Key Drawer listeners & persistence
function initCloudApiKeysUi() {
    const toggleBtn = document.getElementById("toggleCloudKeysBtn");
    const content = document.getElementById("cloudKeysContent");
    const arrow = document.getElementById("cloudKeysToggleArrow");
    const geminiInput = document.getElementById("geminiApiKeyInput");
    const groqInput = document.getElementById("groqApiKeyInput");
    const btnSaveGemini = document.getElementById("btnSaveGeminiKey");
    const btnSaveGroq = document.getElementById("btnSaveGroqKey");
    const badge = document.getElementById("cloudKeysBadge");

    if (toggleBtn && content) {
        toggleBtn.addEventListener("click", () => {
            const isHidden = content.style.display === "none";
            content.style.display = isHidden ? "grid" : "none";
            if (arrow) arrow.textContent = isHidden ? "▲ Collapse" : "▼ Expand";
        });
    }

    // Load saved keys from localStorage
    const savedGemini = localStorage.getItem("moments_gemini_api_key") || "";
    const savedGroq = localStorage.getItem("moments_groq_api_key") || "";
    if (geminiInput && savedGemini) geminiInput.value = savedGemini;
    if (groqInput && savedGroq) groqInput.value = savedGroq;

    const updateBadge = () => {
        const hasGemini = !!(geminiInput?.value || savedGemini);
        const hasGroq = !!(groqInput?.value || savedGroq);
        if (badge) {
            if (hasGemini && hasGroq) {
                badge.style.background = "rgba(16, 185, 129, 0.2)";
                badge.style.color = "#34d399";
                badge.textContent = "✨ Gemini & ⚡ Groq Configured";
            } else if (hasGemini) {
                badge.style.background = "rgba(56, 189, 248, 0.2)";
                badge.style.color = "#7dd3fc";
                badge.textContent = "✨ Gemini Active";
            } else if (hasGroq) {
                badge.style.background = "rgba(245, 158, 11, 0.2)";
                badge.style.color = "#fcd34d";
                badge.textContent = "⚡ Groq Active";
            } else {
                badge.style.background = "rgba(99, 102, 241, 0.2)";
                badge.style.color = "#a5b4fc";
                badge.textContent = "Offline / Local Mode";
            }
        }
    };
    updateBadge();

    if (btnSaveGemini && geminiInput) {
        btnSaveGemini.addEventListener("click", () => {
            const val = geminiInput.value.trim();
            localStorage.setItem("moments_gemini_api_key", val);
            btnSaveGemini.textContent = "Saved ✓";
            setTimeout(() => { btnSaveGemini.textContent = "Save"; }, 1500);
            updateBadge();
        });
    }

    if (btnSaveGroq && groqInput) {
        btnSaveGroq.addEventListener("click", () => {
            const val = groqInput.value.trim();
            localStorage.setItem("moments_groq_api_key", val);
            btnSaveGroq.textContent = "Saved ✓";
            setTimeout(() => { btnSaveGroq.textContent = "Save"; }, 1500);
            updateBadge();
        });
    }
}

async function fetchDirectorModels() {
    try {
        const resp = await fetch("/api/v1/director/models");
        if (!resp.ok) return;
        const data = await resp.json();

        const modelSelect = document.getElementById("directorModelSelect");
        const statusBadge = document.getElementById("directorOllamaStatus");
        const modelBadge = document.getElementById("directorModelBadge");

        if (statusBadge) {
            if (data.ollama_connected) {
                statusBadge.className = "badge badge-success";
                statusBadge.textContent = "🟢 Ollama Connected";
            } else {
                statusBadge.className = "badge badge-warning";
                statusBadge.textContent = "⚪ Mock / Offline Mode";
            }
        }

        if (modelBadge && data.default_model) {
            modelBadge.textContent = `🤖 LLM: ${data.default_model}`;
        }

        if (modelSelect && data.models && data.models.length > 0) {
            modelSelect.innerHTML = "";

            // Group by provider
            const groups = {
                "gemini": { label: "✨ Google Gemini Cloud (Fast & Cinematic)", items: [] },
                "groq": { label: "⚡ Groq Cloud (Sub-Second LPUs)", items: [] },
                "ollama": { label: "🏠 Local Apple Silicon (Ollama)", items: [] },
                "mock": { label: "🧪 Simulation", items: [] },
            };

            data.models.forEach(m => {
                const prov = m.provider || "ollama";
                if (groups[prov]) {
                    groups[prov].items.push(m);
                } else {
                    groups["ollama"].items.push(m);
                }
            });

            Object.keys(groups).forEach(gKey => {
                const grp = groups[gKey];
                if (grp.items.length > 0) {
                    const optgroup = document.createElement("optgroup");
                    optgroup.label = grp.label;

                    grp.items.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m.name;
                        opt.textContent = `${m.display_name} (${m.size_vram})`;
                        if (m.name === data.default_model) {
                            opt.selected = true;
                        }
                        optgroup.appendChild(opt);
                    });

                    modelSelect.appendChild(optgroup);
                }
            });
        }
    } catch (err) {
        console.warn("Failed fetching director models:", err);
    }
}

async function runDirectorAgent() {
    const prompt = document.getElementById("directorPrompt")?.value?.trim();
    if (!prompt) {
        alert("Please enter a creative director prompt!");
        return;
    }

    const duration = parseInt(document.getElementById("directorDuration")?.value || "30", 10);
    const model = document.getElementById("directorModelSelect")?.value || "gemma4:e4b-mlx";
    const mode = document.getElementById("directorRetrievalMode")?.value || "dual";
    const aspectRatio = document.getElementById("directorAspectRatio")?.value || "1:1";

    // Determine if custom API key is available
    let apiKey = null;
    if (model.startsWith("gemini")) {
        apiKey = document.getElementById("geminiApiKeyInput")?.value?.trim() || localStorage.getItem("moments_gemini_api_key") || null;
    } else if (model.startsWith("groq")) {
        apiKey = document.getElementById("groqApiKeyInput")?.value?.trim() || localStorage.getItem("moments_groq_api_key") || null;
    }

    const stateCard = document.getElementById("directorStateCard");
    const resultsContainer = document.getElementById("directorResultsContainer");
    const runBtn = document.getElementById("btnRunDirector");
    const liveMsg = document.getElementById("directorLiveMessage");
    const timerElem = document.getElementById("directorExecutionTimer");

    if (stateCard) stateCard.style.display = "block";
    if (resultsContainer) resultsContainer.style.display = "none";
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = "⏳ Directing...";
    }

    // Reset flowchart nodes and live inspector
    resetFlowchartNodes();
    resetDirectorInspector();
    updateFlowchartNode("planner", "active");

    let startTime = Date.now();
    if (directorTimerInterval) clearInterval(directorTimerInterval);
    directorTimerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        if (timerElem) timerElem.textContent = `${elapsed}s`;
    }, 100);

    try {
        const payload = {
            prompt: prompt,
            target_duration_seconds: duration,
            aspect_ratio: aspectRatio,
            model_name: model,
            retrieval_mode: mode,
            generate_alternatives: true,
        };
        if (apiKey) {
            payload.api_key = apiKey;
        }

        const resp = await fetch("/api/v1/jobs/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned ${resp.status}`);
        }

        const job = await resp.json();
        console.log("Director job created:", job.job_id);

        // Subscribe to SSE stream
        subscribeToDirectorEvents(job.job_id);

    } catch (err) {
        clearInterval(directorTimerInterval);
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.textContent = "🚀 Direct Video";
        }
        if (liveMsg) liveMsg.textContent = `❌ Error: ${err.message}`;
        alert(`Failed starting Director: ${err.message}`);
    }
}

let totalLLMLatency = 0.0;
let totalVectorLatency = 0.0;
let activeLlmInspectorTab = "PLANNER";
const recordedNodeTelemetry = {
    "PLANNER": null,
    "DRAFTING": null,
    "EDITOR": null,
};

function renderLlmPromptTab(node) {
    activeLlmInspectorTab = node;
    document.querySelectorAll(".llm-tab-btn").forEach(btn => {
        if (btn.getAttribute("data-node") === node) {
            btn.classList.add("active");
            btn.style.background = "rgba(167, 139, 250, 0.35)";
            btn.style.borderColor = "#a78bfa";
        } else {
            btn.classList.remove("active");
            btn.style.background = "";
            btn.style.borderColor = "";
        }
    });

    const pPrev = document.getElementById("inspectorPromptPreview");
    const sBadge = document.getElementById("inspectorLLMSchemaBadge");
    const nLabel = document.getElementById("inspectorLLMNodeLabel");

    if (nLabel) nLabel.textContent = `[${node} Node]`;

    const item = recordedNodeTelemetry[node];
    if (!item || !item.full_prompt) {
        if (sBadge) sBadge.textContent = "No calls recorded";
        if (pPrev) pPrev.innerHTML = `<span style="color: #64748b;">No ${node} LLM prompt/response recorded for this run.</span>`;
        return;
    }

    if (sBadge) sBadge.textContent = `${item.schema || 'JSON'} (${(item.latency_seconds || 0).toFixed(2)}s)`;
    if (pPrev) {
        pPrev.textContent = `=== [${node}] SYSTEM PROMPT ===\n${item.system_prompt || 'Default instructions'}\n\n=== USER INPUT PROMPT ===\n${item.full_prompt}\n\n=== STRUCTURED RESPONSE JSON ===\n${JSON.stringify(item.response_json || {}, null, 2)}`;
    }
}

function resetDirectorInspector() {
    totalLLMLatency = 0.0;
    totalVectorLatency = 0.0;
    recordedNodeTelemetry["PLANNER"] = null;
    recordedNodeTelemetry["DRAFTING"] = null;
    recordedNodeTelemetry["EDITOR"] = null;

    const logBox = document.getElementById("inspectorLogContainer");
    if (logBox) logBox.innerHTML = "<div style='color: #64748b;'>[Init] Directing video state machine started...</div>";
    const qList = document.getElementById("inspectorQueriesList");
    if (qList) qList.innerHTML = "<span style='color: #64748b;'>Executing visual queries...</span>";
    renderLlmPromptTab("PLANNER");
    const mLLM = document.getElementById("metricLLMLatency");
    if (mLLM) mLLM.textContent = "0.00s";
    const mVec = document.getElementById("metricVectorLatency");
    if (mVec) mVec.textContent = "0.00s";
    const mCand = document.getElementById("metricCandidatesCount");
    if (mCand) mCand.textContent = "0 items";
    const mBadge = document.getElementById("inspectorLatencyBadge");
    if (mBadge) mBadge.textContent = "⏱️ Latency: 0.0s";
}

function handleDirectorTelemetryEvent(data) {
    if (!data) return;
    const node = data.node || "AGENT";
    const summary = data.summary || "";
    const latency = parseFloat(data.latency_seconds || 0);

    const now = new Date().toTimeString().split(" ")[0];

    // Colors per node
    let badgeColor = "#94a3b8";
    let badgeBg = "rgba(148, 163, 184, 0.2)";
    if (node === "PLANNER") { badgeColor = "#c084fc"; badgeBg = "rgba(192, 132, 252, 0.2)"; totalLLMLatency += latency; }
    else if (node === "RETRIEVAL") { badgeColor = "#38bdf8"; badgeBg = "rgba(56, 189, 248, 0.2)"; totalVectorLatency += latency; }
    else if (node === "DRAFTING") { badgeColor = "#fbbf24"; badgeBg = "rgba(251, 191, 36, 0.2)"; totalLLMLatency += latency; }
    else if (node === "EDITOR") { badgeColor = "#34d399"; badgeBg = "rgba(52, 211, 153, 0.2)"; totalLLMLatency += latency; }
    else if (node === "COMPILER") { badgeColor = "#4ade80"; badgeBg = "rgba(74, 222, 128, 0.2)"; }

    // Update metrics counters
    const mLLM = document.getElementById("metricLLMLatency");
    if (mLLM) mLLM.textContent = `${totalLLMLatency.toFixed(2)}s`;
    const mVec = document.getElementById("metricVectorLatency");
    if (mVec) mVec.textContent = `${totalVectorLatency.toFixed(2)}s`;
    const mBadge = document.getElementById("inspectorLatencyBadge");
    if (mBadge) mBadge.textContent = `⏱️ Latency: ${(totalLLMLatency + totalVectorLatency).toFixed(2)}s`;

    // Append to live terminal log
    const logBox = document.getElementById("inspectorLogContainer");
    if (logBox) {
        const line = document.createElement("div");
        line.style.cssText = "margin-bottom: 4px; display: flex; gap: 6px; align-items: baseline; word-break: break-word;";
        line.innerHTML = `<span style="color: #64748b; font-size: 10px;">[${now}]</span>` +
            `<span style="background: ${badgeBg}; color: ${badgeColor}; padding: 1px 5px; border-radius: 3px; font-weight: bold; font-size: 10px;">[${node}]</span>` +
            `<span>${summary}</span>`;
        logBox.appendChild(line);
        logBox.scrollTop = logBox.scrollHeight;
    }

    // If query breakdown provided
    if (data.query_breakdown && data.query_breakdown.length > 0) {
        const qList = document.getElementById("inspectorQueriesList");
        const qCount = document.getElementById("inspectorQueryCountBadge");
        if (qCount) qCount.textContent = `${data.query_breakdown.length} Queries`;
        if (qList) {
            qList.innerHTML = "";
            data.query_breakdown.forEach((qb, i) => {
                const item = document.createElement("div");
                item.style.cssText = "background: rgba(30, 41, 59, 0.7); padding: 4px 8px; border-radius: 4px; font-size: 10.5px; border-left: 2px solid #38bdf8;";
                item.innerHTML = `<strong>Q${i+1}:</strong> "${qb.query}" <span style="color: #94a3b8; font-size: 10px;">(${qb.embed_ms}ms | 🎬 ${qb.matched_scenes} scenes, 🖼️ ${qb.matched_frames} frames)</span>`;
                qList.appendChild(item);
            });
        }
    }

    if (data.total_candidates !== undefined) {
        const mCand = document.getElementById("metricCandidatesCount");
        if (mCand) mCand.textContent = `${data.total_candidates} items`;
    }

    // If LLM telemetry provided
    if (data.llm_telemetry && data.llm_telemetry.full_prompt) {
        recordedNodeTelemetry[node] = data.llm_telemetry;
        const mModel = document.getElementById("metricLLMModel");
        if (mModel && data.llm_telemetry.model) mModel.textContent = data.llm_telemetry.model;

        // Refresh currently active tab
        renderLlmPromptTab(activeLlmInspectorTab);
    }
}

function subscribeToDirectorEvents(jobId) {
    const liveMsg = document.getElementById("directorLiveMessage");
    const pbar = document.getElementById("directorProgressBar");
    const stageBadge = document.getElementById("directorLiveStageBadge");
    const runBtn = document.getElementById("btnRunDirector");

    resetDirectorInspector();

    const eventSource = new EventSource(`/api/v1/jobs/${jobId}/events`);

    eventSource.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            console.log("[Director SSE]", payload);

            if (pbar && payload.progress_pct !== undefined) {
                pbar.style.width = `${payload.progress_pct}%`;
            }

            if (liveMsg && payload.message) {
                liveMsg.textContent = payload.message;
            }

            if (stageBadge && payload.stage) {
                stageBadge.textContent = payload.stage;
            }

            // If telemetry event
            if (payload.event_type === "telemetry" && payload.data) {
                handleDirectorTelemetryEvent(payload.data);
            }

            // Flowchart stage mapping
            const stage = (payload.stage || "").toUpperCase();
            if (stage === "PLANNING") {
                updateFlowchartNode("planner", "active");
            } else if (stage === "RETRIEVAL") {
                updateFlowchartNode("planner", "completed");
                updateFlowchartNode("retrieval", "active");
            } else if (stage === "DRAFTING") {
                updateFlowchartNode("retrieval", "completed");
                updateFlowchartNode("drafting", "active");
            } else if (stage === "EDITING") {
                updateFlowchartNode("drafting", "completed");
                updateFlowchartNode("editor", "active");
            } else if (stage === "COMPILING") {
                updateFlowchartNode("editor", "completed");
                updateFlowchartNode("compiler", "active");
            } else if (payload.event_type === "completed" || (stage === "COMPLETED" && payload.event_type !== "telemetry")) {
                updateFlowchartNode("editor", "completed");
                updateFlowchartNode("compiler", "completed");
                clearInterval(directorTimerInterval);
                eventSource.close();

                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.textContent = "🚀 Direct Video";
                }

                // Render final results
                if (payload.data) {
                    currentDirectorJobData = payload.data;
                    renderDirectorResults(payload.data);
                    // Render any remaining agent telemetry logs
                    if (payload.data.agent_telemetry) {
                        payload.data.agent_telemetry.forEach(t => handleDirectorTelemetryEvent(t));
                    }
                }
            } else if (stage === "FAILED" || payload.event_type === "error") {
                clearInterval(directorTimerInterval);
                eventSource.close();
                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.textContent = "🚀 Direct Video";
                }
                alert(`Director Job Failed: ${payload.message}`);
            }
        } catch (e) {
            console.error("Error parsing SSE event:", e);
        }
    };

    eventSource.onerror = async (err) => {
        console.warn("Director EventSource error / closed:", err);
        eventSource.close();

        // Fallback: poll job status once in case SSE closed after job completed
        try {
            const resp = await fetch(`/api/v1/jobs/${jobId}`);
            if (resp.ok) {
                const jobData = await resp.json();
                if (jobData.status === "COMPLETED" && jobData.summary) {
                    updateFlowchartNode("editor", "completed");
                    updateFlowchartNode("compiler", "completed");
                    clearInterval(directorTimerInterval);
                    if (runBtn) {
                        runBtn.disabled = false;
                        runBtn.textContent = "🚀 Direct Video";
                    }
                    currentDirectorJobData = jobData.summary;
                    renderDirectorResults(jobData.summary);
                } else if (jobData.status === "FAILED") {
                    clearInterval(directorTimerInterval);
                    if (runBtn) {
                        runBtn.disabled = false;
                        runBtn.textContent = "🚀 Direct Video";
                    }
                    alert(`Director Job Failed: ${jobData.error || jobData.message}`);
                }
            }
        } catch (e) {
            console.error("Fallback job poll failed:", e);
        }
    };
}

function resetFlowchartNodes() {
    ["planner", "retrieval", "drafting", "editor", "compiler"].forEach(name => {
        const node = document.getElementById(`node-${name}`);
        if (node) {
            node.className = "graph-node-card";
        }
    });
}

function updateFlowchartNode(name, status) {
    const node = document.getElementById(`node-${name}`);
    if (node) {
        node.classList.remove("active", "completed");
        node.classList.add(status);
    }
}

function renderDirectorResults(data) {
    if (!data) return;
    currentDirectorJobData = data;
    console.log("[Director UI] Rendering Results:", data);

    const resultsContainer = document.getElementById("directorResultsContainer");
    if (!resultsContainer) return;
    resultsContainer.style.display = "block";

    // Narrative & Sub-queries
    const narrativeElem = document.getElementById("directorNarrativeArcText");
    if (narrativeElem) {
        narrativeElem.textContent = `"${data.narrative_arc || 'Cinematic moments sequence tailored to prompt.'}"`;
    }

    const queriesContainer = document.getElementById("directorSubQueriesList");
    if (queriesContainer && data.search_queries) {
        queriesContainer.innerHTML = "";
        data.search_queries.forEach((q, idx) => {
            const pill = document.createElement("div");
            pill.style.cssText = "display: flex; align-items: center; gap: 8px; font-size: 12px; background: rgba(99, 102, 241, 0.1); padding: 5px 10px; border-radius: 6px; border-left: 2px solid #6366f1;";
            pill.innerHTML = `<span>🔍 [Query ${idx + 1}]</span> <strong style="color: #c7d2fe;">${q}</strong>`;
            queriesContainer.appendChild(pill);
        });
    }

    // Editor Critique
    const feedbackBox = document.getElementById("editorFeedbackBox");
    if (feedbackBox) {
        if (data.editor_feedback && data.editor_feedback.length > 0) {
            feedbackBox.innerHTML = "<strong>Editorial Feedback:</strong><ul style='margin-left: 15px; margin-top: 4px;'>" +
                data.editor_feedback.map(f => `<li>${f}</li>`).join("") + "</ul>";
            feedbackBox.style.background = "rgba(245, 158, 11, 0.1)";
            feedbackBox.style.borderLeftColor = "#f59e0b";
            feedbackBox.style.color = "#fef3c7";
        } else {
            feedbackBox.innerHTML = "✅ Storyboard passed all duration, pacing, and visual diversity checks!";
            feedbackBox.style.background = "rgba(16, 185, 129, 0.1)";
            feedbackBox.style.borderLeftColor = "#10b981";
            feedbackBox.style.color = "#a7f3d0";
        }
    }

    // Default display Alternative C
    activeAlternativeKey = (data.alternatives && data.alternatives["alt_c_dual"]) ? "alt_c_dual" : "alt_a_scene";
    switchAlternativeView(activeAlternativeKey);
}

function switchAlternativeView(altKey) {
    if (!currentDirectorJobData) return;
    activeAlternativeKey = altKey;

    // Update active tab buttons
    document.querySelectorAll(".alt-btn").forEach(btn => {
        if (btn.getAttribute("data-alt") === altKey) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    let currentAltState = null;
    if (currentDirectorJobData.alternatives && currentDirectorJobData.alternatives[altKey]) {
        currentAltState = currentDirectorJobData.alternatives[altKey];
    } else {
        currentAltState = currentDirectorJobData;
    }

    // Synchronize telemetry items for LLM inspector
    if (currentAltState && currentAltState.agent_telemetry) {
        currentAltState.agent_telemetry.forEach(t => {
            if (t.node && t.llm_telemetry && t.llm_telemetry.full_prompt) {
                recordedNodeTelemetry[t.node] = t.llm_telemetry;
            }
        });
        renderLlmPromptTab(activeLlmInspectorTab);
    }

    const storyboard = currentAltState.storyboard || [];
    const targetDur = currentDirectorJobData.target_duration || 30;
    const totalDur = storyboard.reduce((acc, s) => acc + (s.duration || 0), 0);

    // Update summary banner
    const durElem = document.getElementById("storyboardDuration");
    const targetElem = document.getElementById("storyboardTargetDur");
    const segCountElem = document.getElementById("storyboardSegmentsCount");
    const approvalElem = document.getElementById("storyboardApprovalBadge");

    if (durElem) durElem.textContent = `${totalDur.toFixed(1)}s`;
    if (targetElem) targetElem.textContent = `${targetDur}`;
    if (segCountElem) segCountElem.textContent = `${storyboard.length}`;
    if (approvalElem) {
        const iter = currentAltState.iteration_count || 1;
        approvalElem.textContent = currentAltState.approved ? `✅ Approved (Iter ${iter})` : `⚠️ Auto-Compiled`;
    }

    // Render Filmstrip Grid
    const filmstrip = document.getElementById("storyboardFilmstrip");
    if (!filmstrip) return;
    filmstrip.innerHTML = "";

    if (storyboard.length === 0) {
        filmstrip.innerHTML = `<div style="grid-column: 1 / -1; padding: 20px; text-align: center; color: var(--text-muted);">No segments drafted for this alternative.</div>`;
        return;
    }

    storyboard.forEach((seg, idx) => {
        const card = document.createElement("div");
        card.className = "storyboard-card";

        const fileName = seg.file_path ? seg.file_path.split("/").pop() : "Unknown";
        const thumbUrl = `/api/v1/media/thumbnail?path=${encodeURIComponent(seg.file_path || '')}&offset=${seg.start_offset || 0}`;
        const mediaUrl = `/api/v1/media/file?path=${encodeURIComponent(seg.file_path || '')}`;
        const typeBadge = seg.segment_type === "video_clip" ? "🎥 Video Clip" : "📸 Photo";
        const stratBadge = seg.retrieval_strategy === "scene" ? "Scene" : "Frame";

        let dateBadge = "";
        if (seg.creation_timestamp) {
            try {
                const dt = new Date(seg.creation_timestamp * 1000);
                const dStr = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                const tStr = dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
                dateBadge = `<span style="font-size: 9px; color: #38bdf8; background: rgba(56, 189, 248, 0.15); padding: 1px 4px; border-radius: 3px;">📅 ${dStr} ${tStr}</span>`;
            } catch (e) {}
        }

        const scoreStr = (seg.similarity_score !== undefined && seg.similarity_score !== null)
            ? `<span style="font-size: 9px; color: #a7f3d0; background: rgba(16, 185, 129, 0.15); padding: 1px 4px; border-radius: 3px;">★ ${(seg.similarity_score).toFixed(3)}</span>`
            : "";

        card.innerHTML = `
            <div class="storyboard-thumb-container">
                <img src="${thumbUrl}" alt="${fileName}" class="storyboard-thumb" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\' fill=\\'%23333\\'><rect width=\\'100\\' height=\\'100\\'/></svg>'">
                <span class="storyboard-badge-pos">#${idx + 1}</span>
                <span class="storyboard-badge-dur">${(seg.duration || 3.0).toFixed(1)}s</span>
            </div>
            <div class="storyboard-content">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 4px;">
                    <strong style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; font-size: 11px;" title="${fileName}">${fileName}</strong>
                    <span style="font-size: 9px; padding: 1px 4px; border-radius: 3px; background: rgba(99, 102, 241, 0.2); color: #a5b4fc;">${stratBadge}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                    ${dateBadge || `<span style="font-size: 9px; color: var(--text-muted);">${typeBadge}</span>`}
                    ${scoreStr}
                </div>
                <div class="storyboard-justification" title="${seg.justification || ''}" style="margin-top: 4px;">
                    "${seg.justification || 'Key highlight matching narrative intent.'}"
                </div>
            </div>
        `;

        // Click to view
        card.addEventListener("click", () => {
            openDirectorMediaModal({
                file_path: seg.file_path,
                file_name: fileName,
                media_url: mediaUrl,
                thumb_url: thumbUrl,
                is_video: seg.segment_type === "video_clip",
                start_offset: seg.start_offset || 0,
                duration: seg.duration || 3.0,
                end_offset: seg.end_offset || ((seg.start_offset || 0) + (seg.duration || 3.0)),
                score: seg.similarity_score || 0.9,
                justification: seg.justification || "",
                creation_timestamp: seg.creation_timestamp,
            });
        });

        filmstrip.appendChild(card);
    });
}

async function openDirectorMediaModal(item) {
    const existing = document.getElementById("directorMediaModal");
    if (existing) existing.remove();

    const modal = document.createElement("div");
    modal.id = "directorMediaModal";
    modal.style.cssText = `
        position: fixed; inset: 0; background: rgba(0, 0, 0, 0.92); backdrop-filter: blur(14px);
        z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 20px;
    `;

    const clipUrl = `/api/v1/media/clip?path=${encodeURIComponent(item.file_path || '')}&start_offset=${item.start_offset || 0}&duration=${item.duration || 3.0}`;
    const startSec = item.start_offset || 0;
    const endSec = item.end_offset || (startSec + item.duration);

    modal.innerHTML = `
        <div style="background: #0f172a; border: 1px solid var(--border-color); border-radius: 16px; max-width: 960px; width: 100%; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.9); display: flex; flex-direction: column;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--border-color); background: rgba(255,255,255,0.03);">
                <div>
                    <h3 style="font-size: 15px; margin: 0; color: #fff; display: flex; align-items: center; gap: 8px;">
                        <span id="modalTypeIcon">${item.is_video ? '🎥' : '📸'}</span>
                        <span>${item.file_name}</span>
                        <span id="livePhotoBadge" style="display: none; font-size: 10px; background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 1px 6px; border-radius: 10px; font-weight: bold; cursor: pointer;">
                            🔴 LIVE PHOTO
                        </span>
                    </h3>
                    <span style="font-size: 12px; color: #94a3b8;" id="modalTimeSubtitle">
                        ${item.is_video ? `Curated Clip: <strong>${startSec.toFixed(1)}s ➔ ${endSec.toFixed(1)}s</strong> (${item.duration.toFixed(1)}s)` : `Photo Moment • Duration: ${item.duration.toFixed(1)}s`}
                    </span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    ${item.is_video ? `
                    <button type="button" id="btnToggleLoopMode" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px; background: rgba(99, 102, 241, 0.2); color: #c7d2fe;">
                        🔁 Loop Clip
                    </button>
                    <button type="button" id="btnPlayFullVideo" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;">
                        ▶ Full Video
                    </button>
                    ` : `
                    <button type="button" id="btnToggleLivePhotoMotion" class="btn btn-secondary" style="display: none; padding: 4px 10px; font-size: 11px; background: rgba(239, 68, 68, 0.2); color: #fca5a5;">
                        ▶ Play Motion
                    </button>
                    `}
                    <button id="btnCloseDirectorModal" style="background: transparent; border: none; font-size: 22px; color: #9ca3af; cursor: pointer; padding: 4px 8px;">✕</button>
                </div>
            </div>

            <!-- Media Viewport -->
            <div id="modalMediaViewport" style="background: #000; display: flex; justify-content: center; align-items: center; min-height: 380px; max-height: 70vh; position: relative;">
                ${item.is_video ? `
                <video id="directorModalVideo" src="${item.media_url}" controls playsinline preload="auto" style="width: 100%; height: 100%; max-height: 70vh; object-fit: contain;"></video>
                <img id="directorModalFallbackImg" src="${clipUrl}" style="display: none; width: 100%; height: 100%; max-height: 70vh; object-fit: contain;" alt="${item.file_name}">
                ` : `
                <img id="directorModalPhoto" src="${item.media_url}" alt="${item.file_name}" style="width: 100%; height: 100%; max-height: 70vh; object-fit: contain;">
                <video id="directorModalLiveVideo" playsinline loop style="display: none; width: 100%; height: 100%; max-height: 70vh; object-fit: contain;"></video>
                `}
            </div>

            <!-- Storyboard Details Footer -->
            <div style="padding: 12px 20px; background: rgba(15, 23, 42, 0.8); border-top: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; font-size: 12px;">
                <div style="color: #cbd5e1; max-width: 80%;">
                    ${item.justification ? `<strong style="color: #a5b4fc;">Director Rationale:</strong> "${item.justification}"` : `<span style="color: #64748b;">Selected moment from narrative sequence.</span>`}
                </div>
                ${item.score ? `<span style="color: #34d399; font-weight: bold; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 4px;">★ ${(item.score).toFixed(3)} match</span>` : ''}
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    let isLoopingClip = true;

    // Check Live Photo info asynchronously
    try {
        const infoResp = await fetch(`/api/v1/media/info?path=${encodeURIComponent(item.file_path || '')}`);
        if (infoResp.ok) {
            const info = await infoResp.json();
            if (info.has_live_photo) {
                const liveBadge = document.getElementById("livePhotoBadge");
                const liveBtn = document.getElementById("btnToggleLivePhotoMotion");
                const liveVideo = document.getElementById("directorModalLiveVideo");
                const photoImg = document.getElementById("directorModalPhoto");

                if (liveBadge) liveBadge.style.display = "inline-block";
                if (liveBtn) {
                    liveBtn.style.display = "inline-block";
                    let isPlayingMotion = false;

                    const toggleMotion = () => {
                        isPlayingMotion = !isPlayingMotion;
                        if (isPlayingMotion && liveVideo && photoImg) {
                            photoImg.style.display = "none";
                            liveVideo.style.display = "block";
                            liveVideo.src = info.live_photo_url || info.media_url;
                            liveVideo.play().catch(() => {});
                            liveBtn.textContent = "📸 Show Photo";
                        } else if (liveVideo && photoImg) {
                            liveVideo.pause();
                            liveVideo.style.display = "none";
                            photoImg.style.display = "block";
                            liveBtn.textContent = "▶ Play Motion";
                        }
                    };

                    liveBtn.addEventListener("click", toggleMotion);
                    if (liveBadge) liveBadge.addEventListener("click", toggleMotion);
                }
            }
        }
    } catch (e) {
        console.debug("Live photo check:", e);
    }

    if (item.is_video) {
        const video = document.getElementById("directorModalVideo");
        const fallbackImg = document.getElementById("directorModalFallbackImg");
        const btnLoop = document.getElementById("btnToggleLoopMode");
        const btnFull = document.getElementById("btnPlayFullVideo");

        if (btnLoop) {
            btnLoop.addEventListener("click", () => {
                isLoopingClip = true;
                btnLoop.style.background = "rgba(99, 102, 241, 0.4)";
                if (btnFull) btnFull.style.background = "";
                if (video) {
                    video.currentTime = startSec;
                    video.play().catch(() => {});
                }
            });
        }

        if (btnFull) {
            btnFull.addEventListener("click", () => {
                isLoopingClip = false;
                if (btnLoop) btnLoop.style.background = "";
                btnFull.style.background = "rgba(99, 102, 241, 0.4)";
                if (video) {
                    video.currentTime = 0;
                    video.play().catch(() => {});
                }
            });
        }

        if (video) {
            video.addEventListener("loadedmetadata", () => {
                video.currentTime = startSec;
                video.play().catch(() => {});
            });

            video.addEventListener("timeupdate", () => {
                if (isLoopingClip && endSec > startSec) {
                    if (video.currentTime >= endSec) {
                        video.currentTime = startSec;
                    }
                }
            });

            // If browser video decoder fails (e.g. unsupported QuickTime HEVC in older browser), switch seamlessly to animated clip
            video.addEventListener("error", () => {
                console.warn("Native video decode failed, switching to high-def animated clip stream...");
                video.style.display = "none";
                if (fallbackImg) {
                    fallbackImg.style.display = "block";
                }
            });
        }
    }

    const closeBtn = document.getElementById("btnCloseDirectorModal");
    if (closeBtn) {
        closeBtn.addEventListener("click", () => modal.remove());
    }

    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.remove();
    });
}

// Full Timeline Storyboard Cinema Player
function openFullStoryboardPlayer() {
    if (!currentDirectorJobData) {
        alert("No active storyboard generated yet. Run Direct Video first!");
        return;
    }

    let currentAltState = null;
    if (currentDirectorJobData.alternatives && currentDirectorJobData.alternatives[activeAlternativeKey]) {
        currentAltState = currentDirectorJobData.alternatives[activeAlternativeKey];
    } else {
        currentAltState = currentDirectorJobData;
    }

    const storyboard = currentAltState.storyboard || [];
    if (storyboard.length === 0) {
        alert("Storyboard is empty.");
        return;
    }

    const existing = document.getElementById("directorCinemaModal");
    if (existing) existing.remove();

    const totalDur = storyboard.reduce((acc, s) => acc + (s.duration || 0), 0);

    const modal = document.createElement("div");
    modal.id = "directorCinemaModal";
    modal.style.cssText = `
        position: fixed; inset: 0; background: rgba(0, 0, 0, 0.95); backdrop-filter: blur(16px);
        z-index: 10000; display: flex; flex-direction: column; align-items: center; justify-content: space-between; padding: 20px;
    `;

    modal.innerHTML = `
        <!-- Top Bar -->
        <div style="width: 100%; max-width: 1100px; display: flex; justify-content: space-between; align-items: center; color: #fff;">
            <div>
                <h3 style="margin: 0; font-size: 16px; display: flex; align-items: center; gap: 8px;">
                    <span>🎬 Full Storyboard Cinema</span>
                    <span style="font-size: 11px; padding: 2px 8px; border-radius: 4px; background: rgba(16, 185, 129, 0.2); color: #34d399;">
                        ${activeAlternativeKey.toUpperCase()} Cut (${totalDur.toFixed(1)}s)
                    </span>
                </h3>
                <span id="cinemaClipTitle" style="font-size: 12px; color: #94a3b8;">Loading timeline...</span>
            </div>
            <button id="btnCloseCinemaModal" style="background: transparent; border: none; font-size: 26px; color: #cbd5e1; cursor: pointer; padding: 4px 10px;">✕</button>
        </div>

        <!-- Cinema Viewport -->
        <div style="width: 100%; max-width: 1100px; flex: 1; display: flex; justify-content: center; align-items: center; position: relative; margin: 15px 0;">
            <img id="cinemaPhotoElem" style="max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 8px; display: none; box-shadow: 0 20px 40px rgba(0,0,0,0.8);">
            <video id="cinemaVideoElem" playsinline preload="auto" style="max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 8px; display: none; box-shadow: 0 20px 40px rgba(0,0,0,0.8);"></video>
            
            <div id="cinemaOverlayBadge" style="position: absolute; bottom: 20px; left: 30px; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(8px); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); color: #fff; font-size: 12px;">
                Clip 1 of ${storyboard.length}
            </div>
        </div>

        <!-- Bottom Controls & Timeline Bar -->
        <div style="width: 100%; max-width: 1100px; background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border-color); border-radius: 12px; padding: 14px 20px; display: flex; flex-direction: column; gap: 10px;">
            <!-- Progress Bar -->
            <div style="height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden; position: relative; cursor: pointer;" id="cinemaProgressBar">
                <div id="cinemaProgressFill" style="height: 100%; width: 0%; background: linear-gradient(90deg, #38bdf8, #10b981); transition: width 0.1s linear;"></div>
            </div>

            <!-- Control Buttons -->
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; gap: 10px; align-items: center;">
                    <button type="button" id="btnCinemaPrev" class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;">⏮ Prev</button>
                    <button type="button" id="btnCinemaPlayPause" class="btn btn-primary" style="padding: 6px 16px; font-size: 13px; font-weight: bold;">⏸ Pause</button>
                    <button type="button" id="btnCinemaNext" class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;">Next ⏭</button>
                </div>
                <div style="font-size: 12px; color: #94a3b8;">
                    <span id="cinemaTimeElapsed">0.0s</span> / <strong>${totalDur.toFixed(1)}s</strong>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    let currentIndex = 0;
    let isPlaying = true;
    let segmentTimer = null;
    let overallTimer = null;
    let elapsedSeconds = 0.0;

    const photoElem = document.getElementById("cinemaPhotoElem");
    const videoElem = document.getElementById("cinemaVideoElem");
    const titleElem = document.getElementById("cinemaClipTitle");
    const badgeElem = document.getElementById("cinemaOverlayBadge");
    const progressFill = document.getElementById("cinemaProgressFill");
    const timeElapsedElem = document.getElementById("cinemaTimeElapsed");
    const btnPlayPause = document.getElementById("btnCinemaPlayPause");

    function playSegment(index) {
        if (index < 0 || index >= storyboard.length) {
            // Reached end of cut
            currentIndex = 0;
            elapsedSeconds = 0.0;
            if (progressFill) progressFill.style.width = "100%";
            if (btnPlayPause) btnPlayPause.textContent = "🔄 Replay";
            isPlaying = false;
            return;
        }

        currentIndex = index;
        const seg = storyboard[currentIndex];
        const fileName = seg.file_path ? seg.file_path.split("/").pop() : "Moment";
        const isVideo = seg.segment_type === "video_clip";
        const segDur = seg.duration || 3.0;

        if (titleElem) {
            titleElem.textContent = `[${currentIndex + 1}/${storyboard.length}] ${fileName} • ${seg.justification || ''}`;
        }
        if (badgeElem) {
            badgeElem.textContent = `Moment ${currentIndex + 1} of ${storyboard.length} (${segDur.toFixed(1)}s)`;
        }

        if (segmentTimer) clearTimeout(segmentTimer);

        if (isVideo && videoElem) {
            if (photoElem) photoElem.style.display = "none";
            videoElem.style.display = "block";
            const startOff = seg.start_offset || 0;
            videoElem.src = `/api/v1/media/file?path=${encodeURIComponent(seg.file_path || '')}`;
            videoElem.currentTime = startOff;
            videoElem.play().catch(() => {});

            videoElem.onerror = () => {
                // Fallback to animated clip
                videoElem.style.display = "none";
                if (photoElem) {
                    photoElem.style.display = "block";
                    photoElem.src = `/api/v1/media/clip?path=${encodeURIComponent(seg.file_path || '')}&start_offset=${startOff}&duration=${segDur}`;
                }
            };

            segmentTimer = setTimeout(() => {
                if (isPlaying) {
                    playSegment(currentIndex + 1);
                }
            }, segDur * 1000);

        } else if (photoElem) {
            if (videoElem) {
                videoElem.pause();
                videoElem.style.display = "none";
            }
            photoElem.style.display = "block";
            photoElem.src = `/api/v1/media/file?path=${encodeURIComponent(seg.file_path || '')}`;

            segmentTimer = setTimeout(() => {
                if (isPlaying) {
                    playSegment(currentIndex + 1);
                }
            }, segDur * 1000);
        }
    }

    // Play/Pause button
    if (btnPlayPause) {
        btnPlayPause.addEventListener("click", () => {
            if (isPlaying) {
                isPlaying = false;
                btnPlayPause.textContent = "▶ Play";
                if (segmentTimer) clearTimeout(segmentTimer);
                if (videoElem) videoElem.pause();
            } else {
                isPlaying = true;
                btnPlayPause.textContent = "⏸ Pause";
                playSegment(currentIndex);
            }
        });
    }

    // Prev / Next buttons
    const btnPrev = document.getElementById("btnCinemaPrev");
    const btnNext = document.getElementById("btnCinemaNext");
    if (btnPrev) {
        btnPrev.addEventListener("click", () => {
            playSegment(Math.max(0, currentIndex - 1));
        });
    }
    if (btnNext) {
        btnNext.addEventListener("click", () => {
            playSegment(Math.min(storyboard.length - 1, currentIndex + 1));
        });
    }

    // Progress bar loop
    overallTimer = setInterval(() => {
        if (isPlaying && totalDur > 0) {
            elapsedSeconds = Math.min(totalDur, elapsedSeconds + 0.1);
            if (timeElapsedElem) timeElapsedElem.textContent = `${elapsedSeconds.toFixed(1)}s`;
            if (progressFill) progressFill.style.width = `${(elapsedSeconds / totalDur) * 100}%`;
        }
    }, 100);

    const closeBtn = document.getElementById("btnCloseCinemaModal");
    const cleanup = () => {
        if (segmentTimer) clearTimeout(segmentTimer);
        if (overallTimer) clearInterval(overallTimer);
        if (videoElem) videoElem.pause();
        modal.remove();
    };

    if (closeBtn) closeBtn.addEventListener("click", cleanup);
    modal.addEventListener("click", (e) => {
        if (e.target === modal) cleanup();
    });

    // Start playback
    playSegment(0);
}

// Final Video Rendering Client Handler
async function renderActiveStoryboardVideo() {
    console.log("[Director UI] renderActiveStoryboardVideo triggered. activeAlternativeKey:", activeAlternativeKey);

    const btnRender = document.getElementById("btnRenderFinalVideo");
    const renderCard = document.getElementById("finalRenderCard");
    const progressBox = document.getElementById("renderProgressBox");
    const progressBarFill = document.getElementById("renderProgressBarFill");
    const progressText = document.getElementById("renderProgressText");
    const progressPct = document.getElementById("renderProgressPct");
    const videoPlayer = document.getElementById("finalRenderVideoPlayer");
    const downloadBtn = document.getElementById("btnDownloadRenderedMp4");
    const renderMeta = document.getElementById("finalRenderMeta");
    const renderBadge = document.getElementById("finalRenderBadge");

    if (renderCard) {
        renderCard.style.display = "block";
        renderCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    if (progressBox) progressBox.style.display = "block";

    if (!currentDirectorJobData) {
        if (progressText) progressText.textContent = "⚠️ Please generate a storyboard first using 'Direct Video'.";
        if (renderBadge) {
            renderBadge.className = "badge badge-warning";
            renderBadge.textContent = "No Storyboard";
        }
        alert("No active storyboard available to render. Please click '🚀 Direct Video' first!");
        return;
    }

    let currentAltState = null;
    if (currentDirectorJobData.alternatives && currentDirectorJobData.alternatives[activeAlternativeKey]) {
        currentAltState = currentDirectorJobData.alternatives[activeAlternativeKey];
    } else {
        currentAltState = currentDirectorJobData;
    }

    const storyboard = currentAltState.storyboard || [];
    if (storyboard.length === 0) {
        if (progressText) progressText.textContent = "⚠️ Storyboard contains 0 segments. Cannot render.";
        alert("Storyboard is empty. Nothing to render.");
        return;
    }

    if (btnRender) {
        btnRender.disabled = true;
        btnRender.textContent = "⏳ Rendering MP4...";
    }
    if (renderBadge) {
        renderBadge.className = "badge badge-warning";
        renderBadge.textContent = "Compiling & Encoding...";
    }

    // Simulated progress ticker while FFmpeg encodes
    let pct = 5;
    if (progressBarFill) progressBarFill.style.width = "5%";
    if (progressPct) progressPct.textContent = "5%";
    if (progressText) progressText.textContent = "Initializing FFmpeg filter graph & hardware encoder...";

    const progressInterval = setInterval(() => {
        if (pct < 90) {
            pct += Math.floor(Math.random() * 8) + 4;
            if (pct > 90) pct = 90;
            if (progressBarFill) progressBarFill.style.width = `${pct}%`;
            if (progressPct) progressPct.textContent = `${pct}%`;
            if (progressText) {
                if (pct < 35) progressText.textContent = "Applying Ken Burns animations to photo moments...";
                else if (pct < 70) progressText.textContent = "Normalizing video clips and trimming scene timestamps...";
                else progressText.textContent = "Compositing cross-dissolve transitions and encoding H.264 stream...";
            }
        }
    }, 450);

    try {
        const payload = {
            storyboard: storyboard,
            job_id: currentDirectorJobData.job_id || "director_cut",
            aspect_ratio: document.getElementById("directorAspectRatio")?.value || "16:9",
            fps: 30,
            transition_duration: 0.5,
        };

        const resp = await fetch("/api/v1/director/render", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        clearInterval(progressInterval);

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned status ${resp.status}`);
        }

        const data = await resp.json();
        console.log("[Director UI] Video render successful:", data);

        if (progressBarFill) progressBarFill.style.width = "100%";
        if (progressPct) progressPct.textContent = "100%";
        if (progressText) progressText.textContent = "Video rendered successfully!";
        setTimeout(() => {
            if (progressBox) progressBox.style.display = "none";
        }, 1200);

        if (renderBadge) {
            renderBadge.className = "badge badge-success";
            renderBadge.textContent = "✨ Render Complete";
        }

        if (renderMeta) {
            const szMb = (data.file_size_bytes / (1024 * 1024)).toFixed(2);
            renderMeta.textContent = `${data.resolution} • ${data.duration_seconds}s • ${szMb} MB • ${data.total_segments} moments stitched`;
        }

        if (videoPlayer) {
            // Append cache buster to reload stream freshly
            videoPlayer.src = `${data.stream_url}?t=${Date.now()}`;
            videoPlayer.load();
            videoPlayer.play().catch(() => {});
        }

        if (downloadBtn) {
            downloadBtn.href = data.download_url;
            downloadBtn.setAttribute("download", data.file_name);
        }

    } catch (err) {
        clearInterval(progressInterval);
        console.error("Rendering failed:", err);
        if (progressBarFill) {
            progressBarFill.style.width = "100%";
            progressBarFill.style.backgroundColor = "#ef4444";
        }
        if (progressPct) progressPct.textContent = "Error";
        if (progressText) {
            progressText.style.color = "#f87171";
            progressText.textContent = `❌ Rendering error: ${err.message}`;
        }
        if (renderBadge) {
            renderBadge.className = "badge badge-danger";
            renderBadge.textContent = "Render Failed";
        }
        alert(`Rendering failed: ${err.message}`);
    } finally {
        if (btnRender) {
            btnRender.disabled = false;
            btnRender.textContent = "✨ ⚡ Render MP4";
        }
    }
}



