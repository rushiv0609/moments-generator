/**
 * Local AI Moments Generator - Living Debug UI Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initPlayground();
    initWorkspace();
    initMediaExplorer();
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
        _activeEventSource = new EventSource(`/api/v1/jobs/${jobId}/events`);

        _activeEventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
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

        _activeEventSource.addEventListener("error", (event) => {
            if (_activeEventSource) {
                _activeEventSource.close();
                _activeEventSource = null;
            }
            if (btnIndexWs) btnIndexWs.disabled = false;
            if (btnSubmitIdx) btnSubmitIdx.disabled = false;
            if (stageBadge) {
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

            if (item.file_type === "video") {
                if (item.granularity === "scene") {
                    badgeClass = "explorer-badge-scene";
                    typeLabel = `Scene #${item.scene_id ?? 0} (${item.scene_start ?? 0}s–${item.scene_end ?? 0}s)`;
                } else {
                    badgeClass = "explorer-badge-video";
                    typeLabel = `Video @ ${item.source_offset}s`;
                }
            }

            // Score badge formatting
            const scoreFormatted = item.score > 0 ? `+${item.score.toFixed(3)}` : item.score.toFixed(3);

            card.innerHTML = `
                <div class="explorer-thumb-wrap">
                    <img src="${item.media_url}" class="explorer-thumb" loading="lazy" alt="${item.file_name}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\' fill=\\'%23555\\'><rect width=\\'100\\' height=\\'100\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%23999\\'>Preview</text></svg>'">
                    <span class="explorer-rank-badge">#${idx + 1}</span>
                    <span class="explorer-score-badge" title="Cosine Similarity Score">${scoreFormatted}</span>
                </div>
                <div class="explorer-card-body">
                    <div class="explorer-file-title" title="${item.file_path}">${item.file_name}</div>
                    <div class="explorer-meta-row">
                        <span class="explorer-badge-type ${badgeClass}">${typeLabel}</span>
                        <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">${item.granularity}</span>
                    </div>
                </div>
            `;

            // Clicking card opens full image in new browser tab
            card.style.cursor = "pointer";
            card.addEventListener("click", () => {
                window.open(item.media_url, "_blank");
            });

            grid.appendChild(card);
        });
    }
}


