/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Connects to Tosu WebSocket v2, parses active .osu file in-browser,
 * supports HT/DT/custom playback rate scaling, and runs model.onnx via ONNX Runtime Web.
 */

let ortSession = null;
let lastProcessedKey = null;
let isPredicting = false;
let currentRateInfo = { rate: 1.0, label: "", type: "" };

// DOM Elements
const levelDisplay = document.getElementById("level-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const rateBadge = document.getElementById("rate-badge");

function updateStatus(text, dotClass = "idle") {
  if (statusText) statusText.textContent = text;
  if (statusDot) statusDot.className = `status-dot ${dotClass}`.trim();
  console.log(`[O2Jam Predictor] ${text}`);
}

function updateRateBadge(rateInfo) {
  if (!rateBadge) return;
  if (rateInfo && Math.abs(rateInfo.rate - 1.0) > 0.01 && rateInfo.label) {
    rateBadge.textContent = rateInfo.label;
    rateBadge.className = `rate-badge visible ${rateInfo.type}`.trim();
  } else {
    rateBadge.textContent = "";
    rateBadge.className = "rate-badge";
  }
}

function getHost() {
  return window.location.host || "127.0.0.1:24050";
}

function getCounterFolderName() {
  if (typeof window.COUNTER_PATH === "string" && window.COUNTER_PATH.trim()) {
    return window.COUNTER_PATH.trim();
  }
  const path = window.location.pathname.replace(/^\/|\/$/g, "");
  return path || "o2jam_level_predictor";
}

// ── 1. ONNX Runtime Initialization ──────────────────────────────────────────
async function initONNX() {
  updateStatus("Loading model...", "loading");
  try {
    if (typeof ort !== "undefined") {
      ort.env.wasm.wasmPaths = "./";
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }

    ortSession = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["wasm"],
    });

    updateStatus("Ready. Waiting for map...", "idle");
  } catch (err) {
    console.error("[O2Jam Predictor] Failed to load ONNX model:", err);
    updateStatus("Model error", "warning");
  }
}

// ── 2. Parse .osu Text & Build Discrete (T, 7) Grid with Rate Scaling ───────
function parseOsuBeatmap(osuText, rate = 1.0) {
  const lines = osuText.split(/\r?\n/);
  let mode = null;
  let circleSize = null;
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
      const colonIdx = line.indexOf(":");
      if (colonIdx !== -1) {
        const key = line.slice(0, colonIdx).trim();
        const val = line.slice(colonIdx + 1).trim();
        if (key === "Mode") mode = val;
        else if (key === "CircleSize") circleSize = val;
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
      circleSize: csNum || circleSize || "?",
    };
  }

  if (notes.length < 5) {
    return { is7KMania: true, error: "Too few notes (< 5)" };
  }

  notes.sort((a, b) => a.startT - b.startT || a.endT - b.endT);
  const firstT = notes[0].startT;
  let maxEndT = firstT;
  for (let i = 0; i < notes.length; i++) {
    if (notes[i].endT > maxEndT) maxEndT = notes[i].endT;
  }

  const r = Number.isFinite(rate) && rate > 0 ? rate : 1.0;
  const durationMs = Math.max(0, maxEndT - firstT);
  const effectiveDurationMs = durationMs / r;

  // 20ms time bins in effective playback time
  let tFrames = Math.floor(effectiveDurationMs / 20) + 1;
  tFrames = Math.min(Math.max(tFrames, 1), 25000);

  const grid = new Uint8Array(tFrames * 7);
  for (let i = 0; i < notes.length; i++) {
    const note = notes[i];
    const startFrame = Math.floor((note.startT - firstT) / (20 * r));
    let endFrame = Math.floor((note.endT - firstT) / (20 * r));

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
    throw new Error(`HTTP ${resp.status}`);
  }
  return await resp.text();
}

// ── 5. Rate & Mod Extraction ────────────────────────────────────────────────
function extractRateAndMods(data) {
  const isPlaying = Number(data?.state?.number ?? data?.status?.number ?? 0) === 2;

  // 1. Direct playbackRate if given by tosu during active play
  const rateCandidate = Number(data?.gameplay?.playbackRate);
  if (isPlaying && Number.isFinite(rateCandidate) && rateCandidate > 0 && Math.abs(rateCandidate - 1.0) > 0.01) {
    const formatted = rateCandidate.toFixed(2).replace(/\.?0+$/, "");
    return { rate: rateCandidate, label: `${formatted}x`, type: "rate" };
  }

  // 2. Mod candidates: prioritize menu.mods when in song select / menus
  const mods = isPlaying
    ? (data?.play?.mods || data?.gameplay?.mods || data?.menu?.mods)
    : (data?.menu?.mods || data?.play?.mods || data?.gameplay?.mods);

  const name = typeof mods?.name === "string" ? mods.name.toUpperCase() : "";
  const num = Number(mods?.number ?? mods?.num ?? 0);

  // Check DT / NC
  if (name.includes("NC") || (num & 512) !== 0) {
    return { rate: 1.5, label: "NC", type: "dt" };
  }
  if (name.includes("DT") || (num & 64) !== 0) {
    return { rate: 1.5, label: "DT", type: "dt" };
  }

  // Check HT / DC
  if (name.includes("DC")) {
    return { rate: 0.75, label: "DC", type: "ht" };
  }
  if (name.includes("HT") || (num & 256) !== 0) {
    return { rate: 0.75, label: "HT", type: "ht" };
  }

  // 3. Lazer mod array inspection
  const modArray = Array.isArray(mods?.array) ? mods.array : (Array.isArray(mods) ? mods : []);
  for (const m of modArray) {
    const acronym = String(m?.acronym || m || "").toUpperCase();
    const speedChange = Number(m?.settings?.speed_change);
    if (acronym === "NC") return { rate: 1.5, label: "NC", type: "dt" };
    if (acronym === "DT") return { rate: 1.5, label: "DT", type: "dt" };
    if (acronym === "DC") return { rate: 0.75, label: "DC", type: "ht" };
    if (acronym === "HT") return { rate: 0.75, label: "HT", type: "ht" };
    if (Number.isFinite(speedChange) && speedChange > 0 && Math.abs(speedChange - 1.0) > 0.01) {
      return { rate: speedChange, label: `${speedChange}x`, type: "rate" };
    }
  }

  return { rate: 1.0, label: "", type: "" };
}

// ── 6. Main Recompute Handler ───────────────────────────────────────────────
async function processCurrentBeatmap() {
  if (isPredicting) return;
  isPredicting = true;
  updateStatus("Reading map...", "loading");

  try {
    const osuText = await fetchActiveBeatmap();
    const parsed = parseOsuBeatmap(osuText, currentRateInfo.rate);

    if (!parsed.is7KMania) {
      levelDisplay.textContent = "--";
      updateRateBadge({ rate: 1.0, label: "", type: "" });
      updateStatus(`Non-7K (${parsed.circleSize}K)`, "idle");
      return;
    }

    if (parsed.error) {
      levelDisplay.textContent = "--";
      updateRateBadge({ rate: 1.0, label: "", type: "" });
      updateStatus(parsed.error, "warning");
      return;
    }

    // Sync rate badge (DT / HT / None)
    updateRateBadge(currentRateInfo);

    updateStatus("Predicting...", "loading");
    const predResult = await runPrediction(parsed);

    if (predResult) {
      const roundedLevel = Math.round(predResult.level);
      levelDisplay.textContent = roundedLevel;
      levelDisplay.classList.remove("pulse");
      void levelDisplay.offsetWidth;
      levelDisplay.classList.add("pulse");

      const rateInfoText = currentRateInfo.label ? ` [${currentRateInfo.label}]` : "";
      updateStatus(`7K${rateInfoText} (${predResult.latencyMs}ms)`, "");
      console.log(`[O2Jam Predictor] Level: ${predResult.level.toFixed(2)} (Round: ${roundedLevel}) @ rate ${currentRateInfo.rate}`);
    }
  } catch (err) {
    console.error("[O2Jam Predictor] Error processing beatmap:", err);
    updateStatus("Error reading map", "warning");
  } finally {
    isPredicting = false;
  }
}

// ── 7. Tosu WebSocket Connections ───────────────────────────────────────────
function connectToTosu() {
  const host = getHost();
  const counterPath = getCounterFolderName();
  const v2Url = `ws://${host}/websocket/v2?l=${encodeURI(counterPath)}`;

  console.log(`[O2Jam Predictor] Connecting to Tosu WebSocket: ${v2Url}`);
  const ws = new WebSocket(v2Url);

  ws.onopen = () => {
    updateStatus("Connected to Tosu", "idle");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data?.error || data?.message?.error) return;

      const bm = data?.beatmap;
      if (!bm) return;

      const md5 = (bm?.checksum || bm?.md5 || "").toLowerCase();
      const id = bm?.id || 0;
      const title = bm?.title || "";
      const version = bm?.version || "";

      const rateInfo = extractRateAndMods(data);
      const processKey = `${md5 || id}:${title}:${version}@${rateInfo.rate.toFixed(2)}`;

      if (processKey === lastProcessedKey) {
        return;
      }

      lastProcessedKey = processKey;
      currentRateInfo = rateInfo;

      processCurrentBeatmap();
    } catch (err) {
      console.error("[O2Jam Predictor] Error handling WS message:", err);
    }
  };

  ws.onclose = () => {
    updateStatus("Tosu disconnected", "warning");
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
