/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Connects to Tosu WebSocket v2 / v1, parses active .osu file in-browser,
 * and runs model.onnx via ONNX Runtime Web.
 */

let ortSession = null;
let lastBeatmapId = null;
let isPredicting = false;

// DOM Elements
const levelDisplay = document.getElementById("level-display");
const titleDisplay = document.getElementById("title-display");
const diffDisplay = document.getElementById("diff-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

function updateStatus(text, dotClass = "idle") {
  if (statusText) statusText.textContent = text;
  if (statusDot) statusDot.className = `status-dot ${dotClass}`.trim();
  console.log(`[O2Jam Predictor] ${text}`);
}

function getHost() {
  return window.location.host || "127.0.0.1:24050";
}

// ── 1. ONNX Runtime Initialization ──────────────────────────────────────────
async function initONNX() {
  updateStatus("Loading ONNX model...", "loading");
  try {
    if (typeof ort !== "undefined") {
      ort.env.wasm.wasmPaths = "./";
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }

    ortSession = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["wasm"],
    });

    updateStatus("Model ready. Waiting for Tosu...", "idle");
  } catch (err) {
    console.error("[O2Jam Predictor] Failed to load ONNX model:", err);
    updateStatus(`Model error: ${err.message}`, "warning");
  }
}

// ── 2. Parse .osu Text & Build Discrete (T, 7) Grid ─────────────────────────
function parseOsuBeatmap(osuText) {
  const lines = osuText.split(/\r?\n/);
  let mode = null;
  let circleSize = null;
  let title = "";
  let version = "";
  let isHitObjects = false;
  const notes = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    if (line === "[HitObjects]") {
      isHitObjects = true;
      continue;
    }

    if (!isHitObjects) {
      if (line.startsWith("Mode:")) {
        mode = line.split(":", 1)[1].trim();
      } else if (line.startsWith("CircleSize:")) {
        circleSize = line.split(":", 1)[1].trim();
      } else if (line.startsWith("Title:")) {
        title = line.split(":", 1)[1].trim();
      } else if (line.startsWith("Version:")) {
        version = line.split(":", 1)[1].trim();
      }
    } else {
      if (line.startsWith("[")) break;
      const parts = line.split(",");
      if (parts.length >= 4) {
        const x = parseInt(parts[0], 10);
        const t = parseInt(parts[2], 10);
        const nType = parseInt(parts[3], 10);
        const col = Math.min(Math.max(Math.floor((x * 7) / 512), 0), 6);
        let endT = t;
        if (nType & 128 && parts.length > 5) {
          const lnEnd = parseInt(parts[5].split(":")[0], 10);
          if (!isNaN(lnEnd)) endT = lnEnd;
        }
        notes.push({ startT: t, endT: endT, col: col });
      }
    }
  }

  const csNum = parseInt(circleSize, 10);
  const is7KMania = (mode === "3" || mode === 3) && csNum === 7;

  if (!is7KMania) {
    return {
      is7KMania: false,
      mode: mode ?? "0",
      circleSize: csNum || circleSize || "?",
      title: title || "Unknown",
      version: version || "",
    };
  }

  if (notes.length < 5) {
    return { is7KMania: true, error: "Too few notes (< 5)", title, version };
  }

  notes.sort((a, b) => a.startT - b.startT || a.endT - b.endT);
  const firstT = notes[0].startT;
  let maxEndT = firstT;
  for (let i = 0; i < notes.length; i++) {
    if (notes[i].endT > maxEndT) maxEndT = notes[i].endT;
  }

  const durationMs = Math.max(0, maxEndT - firstT);
  const stepMs = 20;
  let tFrames = Math.floor(durationMs / stepMs) + 1;
  tFrames = Math.min(tFrames, 25000);

  const grid = new Uint8Array(tFrames * 7);
  for (let i = 0; i < notes.length; i++) {
    const note = notes[i];
    const startFrame = Math.floor((note.startT - firstT) / stepMs);
    let endFrame = Math.floor((note.endT - firstT) / stepMs);

    if (startFrame >= tFrames) continue;
    endFrame = Math.min(endFrame, tFrames - 1);
    const col = note.col;

    if (endFrame <= startFrame) {
      grid[startFrame * 7 + col] = 1;
    } else {
      grid[startFrame * 7 + col] = 2;
      for (let f = startFrame + 1; f < endFrame; f++) {
        grid[f * 7 + col] = 3;
      }
      grid[endFrame * 7 + col] = 4;
    }
  }

  return {
    is7KMania: true,
    grid,
    tFrames,
    title,
    version,
    noteCount: notes.length,
    durationMs,
  };
}

// ── 3. ONNX Model Inference ─────────────────────────────────────────────────
async function runPrediction(parsed) {
  if (!ortSession || !parsed.grid) return null;

  const tensor = new ort.Tensor("uint8", parsed.grid, [1, parsed.tFrames, 7]);
  const t0 = performance.now();
  const results = await ortSession.run({ note_grid: tensor });
  const latencyMs = (performance.now() - t0).toFixed(1);

  const level = results.predicted_level.data[0];
  return { level, latencyMs };
}

// ── 4. Fetch Active Beatmap from Tosu ────────────────────────────────────────
async function fetchActiveBeatmap() {
  const host = getHost();
  const endpoint = `http://${host}/files/beatmap/file?_=${Date.now()}`;
  const resp = await fetch(endpoint, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status} fetching beatmap file`);
  }
  return await resp.text();
}

// ── 5. Main Recompute Handler ───────────────────────────────────────────────
async function processCurrentBeatmap(metaHint) {
  if (isPredicting) return;
  isPredicting = true;
  updateStatus("Reading map file...", "warning");

  try {
    const osuText = await fetchActiveBeatmap();
    const parsed = parseOsuBeatmap(osuText);

    const title = parsed.title || metaHint?.title || "Unknown Title";
    const diff = parsed.version || metaHint?.version || "";

    titleDisplay.textContent = title;
    diffDisplay.textContent = diff || "7K Mania";

    if (!parsed.is7KMania) {
      levelDisplay.textContent = "--";
      updateStatus(`Non-7K (${parsed.circleSize}K / Mode ${parsed.mode})`, "idle");
      return;
    }

    if (parsed.error) {
      levelDisplay.textContent = "--";
      updateStatus(parsed.error, "warning");
      return;
    }

    updateStatus("Predicting level...", "warning");
    const predResult = await runPrediction(parsed);

    if (predResult) {
      const roundedLevel = Math.round(predResult.level);
      levelDisplay.textContent = roundedLevel;
      levelDisplay.classList.remove("pulse");
      void levelDisplay.offsetWidth;
      levelDisplay.classList.add("pulse");

      updateStatus(`7K Mania (${predResult.latencyMs}ms)`, "");
      console.log(`[O2Jam Predictor] Predicted Level: ${predResult.level.toFixed(2)} (Round: ${roundedLevel})`);
    }
  } catch (err) {
    console.error("[O2Jam Predictor] Error processing beatmap:", err);
    updateStatus(`Error: ${err.message}`, "warning");
  } finally {
    isPredicting = false;
  }
}

// ── 6. Tosu WebSocket Connections (v2 + legacy fallback) ─────────────────────
function connectToTosu() {
  const host = getHost();
  const counterPath = window.COUNTER_PATH || window.location.pathname.replace(/^\/|\/$/g, "");
  const v2Url = `ws://${host}/websocket/v2?l=${encodeURI(counterPath)}`;

  console.log(`[O2Jam Predictor] Connecting to Tosu WebSocket: ${v2Url}`);
  const ws = new WebSocket(v2Url);

  ws.onopen = () => {
    updateStatus("Connected to Tosu v2", "idle");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data?.error || data?.message?.error) return;

      // Detect beatmap identity in v2 payload
      const bm = data?.beatmap;
      if (!bm) return;

      const md5 = (bm?.checksum || bm?.md5 || "").toLowerCase();
      const id = bm?.id || 0;
      const title = bm?.title || "";
      const version = bm?.version || "";
      const mapKey = md5 ? `hash:${md5}` : `id:${id}|${title}:${version}`;

      if (!mapKey || mapKey === lastBeatmapId) {
        return;
      }

      console.log(`[O2Jam Predictor] Beatmap change detected: ${title} [${version}] (${mapKey})`);
      lastBeatmapId = mapKey;
      processCurrentBeatmap({ title, version });
    } catch (err) {
      console.error("[O2Jam Predictor] Error handling WS message:", err);
    }
  };

  ws.onclose = () => {
    updateStatus("Tosu disconnected. Retrying...", "warning");
    setTimeout(connectToTosu, 1500);
  };

  ws.onerror = () => {
    ws.close();
  };
}

// Start
async function main() {
  await initONNX();
  connectToTosu();
}

main();
