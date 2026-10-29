/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Directly runs model.onnx via ONNX Runtime Web in the browser/overlay sandbox.
 */

let ortSession = null;
let currentMapIdentifier = null;
let isPredicting = false;

// DOM Elements
const levelDisplay = document.getElementById("level-display");
const titleDisplay = document.getElementById("title-display");
const diffDisplay = document.getElementById("diff-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const overlayContainer = document.getElementById("overlay-container");

// Initialize ONNX Runtime Session
async function initONNX() {
  try {
    statusText.textContent = "Loading ONNX model...";
    // Configure WASM paths if needed
    if (typeof ort !== "undefined") {
      ort.env.wasm.numThreads = 2;
      ort.env.wasm.simd = true;
    }
    ortSession = await ort.InferenceSession.create("./model.onnx", {
      executionProviders: ["wasm"],
    });
    statusText.textContent = "Model loaded. Ready.";
    statusDot.className = "status-dot idle";
    console.log("[O2Jam Predictor] ONNX model initialized successfully.");
  } catch (err) {
    console.error("[O2Jam Predictor] Failed to load ONNX model:", err);
    statusText.textContent = "ONNX load error";
    statusDot.className = "status-dot warning";
  }
}

// Parse raw .osu beatmap text into a 20ms (T, 7) Uint8 discrete grid
function parseBeatmapToGrid(osuText) {
  const lines = osuText.split(/\r?\n/);
  let isHitObjects = false;
  const notes = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "[HitObjects]") {
      isHitObjects = true;
      continue;
    }
    if (!isHitObjects) continue;
    if (!line || line.startsWith("[")) break;

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

  if (notes.length < 5) return null;

  // Sort notes by start timestamp
  notes.sort((a, b) => a.startT - b.startT || a.endT - b.endT);

  const firstT = notes[0].startT;
  let maxEndT = firstT;
  for (let i = 0; i < notes.length; i++) {
    if (notes[i].endT > maxEndT) maxEndT = notes[i].endT;
  }

  const durationMs = Math.max(0, maxEndT - firstT);
  const stepMs = 20;
  let tFrames = Math.floor(durationMs / stepMs) + 1;
  const maxFramesCap = 25000;
  tFrames = Math.min(tFrames, maxFramesCap);

  // Flat (T * 7) uint8 buffer
  const grid = new Uint8Array(tFrames * 7);

  for (let i = 0; i < notes.length; i++) {
    const note = notes[i];
    const startFrame = Math.floor((note.startT - firstT) / stepMs);
    let endFrame = Math.floor((note.endT - firstT) / stepMs);

    if (startFrame >= tFrames) continue;
    endFrame = Math.min(endFrame, tFrames - 1);

    const col = note.col;

    if (endFrame <= startFrame) {
      grid[startFrame * 7 + col] = 1; // Tap note
    } else {
      grid[startFrame * 7 + col] = 2; // LN Head
      for (let f = startFrame + 1; f < endFrame; f++) {
        grid[f * 7 + col] = 3; // LN Body
      }
      grid[endFrame * 7 + col] = 4; // LN Tail
    }
  }

  return { grid, tFrames, noteCount: notes.length };
}

// Execute ONNX model inference
async function predictLevel(parsedData) {
  if (!ortSession || !parsedData) return null;

  const { grid, tFrames } = parsedData;
  const tensor = new ort.Tensor("uint8", grid, [1, tFrames, 7]);

  const t0 = performance.now();
  const results = await ortSession.run({ note_grid: tensor });
  const latencyMs = (performance.now() - t0).toFixed(1);

  const predLevel = results.predicted_level.data[0];
  return { level: predLevel, latencyMs };
}

// Fetch raw .osu file content from Tosu API
async function fetchCurrentBeatmapFile() {
  const url = `http://127.0.0.1:24050/files/beatmap/file?t=${Date.now()}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: Failed to fetch beatmap file`);
  }
  return await response.text();
}

// Update UI with predicted level
function displayResult(level, latencyMs, title, diff) {
  levelDisplay.textContent = Math.round(level);
  levelDisplay.classList.remove("pulse");
  void levelDisplay.offsetWidth; // Trigger reflow for animation
  levelDisplay.classList.add("pulse");

  titleDisplay.textContent = title || "Unknown Title";
  diffDisplay.textContent = diff || "7K Mania";
  statusText.textContent = `7K Mania (${latencyMs}ms)`;
  statusDot.className = "status-dot";
}

// Connect to Tosu WebSocket
function connectWebSocket() {
  const wsUrl = "ws://127.0.0.1:24050/ws";
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusText.textContent = "Connected to Tosu";
    statusDot.className = "status-dot idle";
    console.log("[O2Jam Predictor] WebSocket connected to Tosu.");
  };

  ws.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      handleGameData(data);
    } catch (err) {
      console.error("[O2Jam Predictor] Error parsing WS data:", err);
    }
  };

  ws.onclose = () => {
    statusText.textContent = "Disconnected (Retrying...)";
    statusDot.className = "status-dot warning";
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

// Process beatmap updates from Tosu
async function handleGameData(data) {
  // Gosumemory / Tosu format inspection
  const bm = data?.menu?.bm || data?.beatmap;
  if (!bm) return;

  const mode = data?.menu?.bm?.stats?.mode ?? data?.gameplay?.gameMode ?? 3;
  const cs = data?.menu?.bm?.stats?.CS ?? data?.beatmap?.stats?.circleSize ?? 7;
  const title = bm?.metadata?.title || bm?.title || "Unknown Title";
  const diff = bm?.metadata?.difficulty || bm?.version || "";
  const mapId = bm?.id || bm?.path?.file || `${title}_${diff}`;

  // Check 7K Mania condition
  const is7KMania = (mode === 3 || mode === "mania") && Math.round(cs) === 7;

  if (!is7KMania) {
    if (currentMapIdentifier !== "non_7k") {
      currentMapIdentifier = "non_7k";
      levelDisplay.textContent = "--";
      titleDisplay.textContent = title;
      diffDisplay.textContent = `${cs}K / Non-7K`;
      statusText.textContent = "Waiting for 7K map...";
      statusDot.className = "status-dot idle";
    }
    return;
  }

  // Detect map change
  if (currentMapIdentifier === mapId || isPredicting) {
    return;
  }

  currentMapIdentifier = mapId;
  isPredicting = true;
  statusText.textContent = "Predicting level...";
  statusDot.className = "status-dot warning";

  try {
    const osuContent = await fetchCurrentBeatmapFile();
    const parsed = parseBeatmapToGrid(osuContent);

    if (parsed) {
      const result = await predictLevel(parsed);
      if (result) {
        displayResult(result.level, result.latencyMs, title, diff);
      }
    } else {
      statusText.textContent = "Parse failed (< 5 notes)";
    }
  } catch (err) {
    console.error("[O2Jam Predictor] Error during prediction:", err);
    statusText.textContent = "Error fetching map";
    statusDot.className = "status-dot warning";
  } finally {
    isPredicting = false;
  }
}

// Start application
async function start() {
  await initONNX();
  connectWebSocket();
}

start();
