/**
 * Local AI Moments Generator - Living Debug UI Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
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

    // Auto-refresh health every 15 seconds
    setInterval(() => {
        fetchHealth(false);
    }, 15000);
});

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
