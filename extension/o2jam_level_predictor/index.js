/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Minimalist HUD Overlay displaying ML-predicted O2Jam 7K difficulty rating.
 */

// ── Default Settings ────────────────────────────────────────────────────────
const settings = {
  showSongTitle: false,
  showDecimal: false,
  hideNon7K: false,
  overlayScale: 1.0,
  overlayOpacity: 0.85,
  accentColor: "#ffc800",
};

// ── Runtime State ───────────────────────────────────────────────────────────
let ortSession = null;
let lastProcessedKey = null;
let isPredicting = false;
let currentRateInfo = { rate: 1.0, label: "", type: "" };
let currentRawLevel = null;
let currentSongTitle = "";
let isCurrentMap7K = true;
let commandsSocket = null;

// ── DOM Elements ────────────────────────────────────────────────────────────
const overlayContainer = document.getElementById("overlay-container");
const songInfoEl = document.getElementById("song-info");
const songTitleEl = document.getElementById("song-title");
const levelDisplay = document.getElementById("level-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const rateBadge = document.getElementById("rate-badge");

// ── Helpers ─────────────────────────────────────────────────────────────────
function getHost() {
  return window.location.host || "127.0.0.1:24050";
}

function getCounterFolderName() {
  if (typeof window.COUNTER_PATH === "string" && window.COUNTER_PATH.trim()) {
    return window.COUNTER_PATH.trim();
  }
  let path = (window.location.pathname || "").replace(/^\/+|\/+$/g, "");
  if (path.endsWith("index.html")) {
    path = path.slice(0, -10).replace(/\/+$/, "");
  }
  const segments = path.split("/").filter(Boolean);
  return segments[0] || "o2jam_level_predictor";
}

function hexToRgba(hex, alpha) {
  let c = String(hex || "").replace("#", "");
  if (c.length === 3) c = c.split("").map((x) => x + x).join("");
  const num = parseInt(c, 16);
  if (isNaN(num)) return `rgba(255, 200, 0, ${alpha})`;
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
}

function parseBoolean(val, fallback) {
  if (typeof val === "boolean") return val;
  if (typeof val === "number") return val !== 0;
  if (typeof val === "string") {
    const s = val.trim().toLowerCase();
    if (s === "true" || s === "1") return true;
    if (s === "false" || s === "0") return false;
  }
  return fallback;
}

function updateStatus(text, dotClass = "idle") {
  if (statusText) statusText.textContent = text;
  if (statusDot) statusDot.className = `status-dot ${dotClass}`.trim();
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

// ── 1. Settings Synchronization ─────────────────────────────────────────────
function applySettings(newSettings) {
  if (!newSettings) return;

  const raw = {};
  if (Array.isArray(newSettings)) {
    for (const item of newSettings) {
      if (item && item.uniqueID) {
        raw[item.uniqueID] = item.value;
      }
    }
  } else if (typeof newSettings === "object") {
    const target = newSettings.settings || newSettings.message || newSettings;
    if (Array.isArray(target)) {
      for (const item of target) {
        if (item && item.uniqueID) raw[item.uniqueID] = item.value;
      }
    } else if (typeof target === "object") {
      Object.assign(raw, target);
    }
  }

  if ("showSongTitle" in raw) settings.showSongTitle = parseBoolean(raw.showSongTitle, settings.showSongTitle);
  if ("showDecimal" in raw) settings.showDecimal = parseBoolean(raw.showDecimal, settings.showDecimal);
  if ("hideNon7K" in raw) settings.hideNon7K = parseBoolean(raw.hideNon7K, settings.hideNon7K);
  if ("overlayScale" in raw) {
    const num = Number(raw.overlayScale);
    if (Number.isFinite(num) && num > 0) settings.overlayScale = num;
  }
  if ("overlayOpacity" in raw) {
    const num = Number(raw.overlayOpacity);
    if (Number.isFinite(num) && num >= 0 && num <= 1) settings.overlayOpacity = num;
  }
  if ("accentColor" in raw && typeof raw.accentColor === "string" && raw.accentColor.startsWith("#")) {
    settings.accentColor = raw.accentColor;
  }

  // Update CSS Custom Properties
  const root = document.documentElement.style;
  root.setProperty("--accent-color", settings.accentColor);
  root.setProperty("--accent-color-glow", hexToRgba(settings.accentColor, 0.45));
  root.setProperty("--bg-opacity", settings.overlayOpacity);
  root.setProperty("--overlay-scale", settings.overlayScale);

  // Song Title Visibility
  if (songInfoEl) {
    songInfoEl.style.display = settings.showSongTitle && currentSongTitle ? "block" : "none";
  }

  // Decimal Precision formatting
  if (typeof currentRawLevel === "number" && levelDisplay && isCurrentMap7K) {
    levelDisplay.textContent = settings.showDecimal
      ? currentRawLevel.toFixed(1)
      : Math.round(currentRawLevel);
  }

  // Non-7K Overlay Visibility
  if (overlayContainer) {
    overlayContainer.style.display = (settings.hideNon7K && !isCurrentMap7K) ? "none" : "flex";
  }
}

async function loadInitialDefaults() {
  try {
    const resp = await fetch("./settings.json", { cache: "no-store" });
    if (resp.ok) {
      const json = await resp.json();
      applySettings(json);
    }
  } catch (err) {
    console.warn("[O2Jam Predictor] Could not load local settings.json:", err);
  }
}

function requestSettings() {
  if (commandsSocket && commandsSocket.readyState === WebSocket.OPEN) {
    const counterPath = getCounterFolderName();
    commandsSocket.send(`getSettings:${encodeURI(counterPath)}`);
  }
}

function connectCommandsSocket() {
  const host = getHost();
  const counterPath = getCounterFolderName();
  const url = `ws://${host}/websocket/commands?l=${encodeURI(counterPath)}`;

  try {
    commandsSocket = new WebSocket(url);

    commandsSocket.onopen = () => {
      requestSettings();
    };

    commandsSocket.onmessage = (event) => {
      try {
        const raw = event.data;
        if (typeof raw !== "string") return;

        let packet;
        if (raw.startsWith("getSettings:") || raw.startsWith("setSettings:")) {
          const colonIdx = raw.indexOf(":");
          packet = { command: raw.slice(0, colonIdx), message: JSON.parse(raw.slice(colonIdx + 1)) };
        } else {
          packet = JSON.parse(raw);
        }

        if (Array.isArray(packet)) {
          applySettings(packet);
        } else if (packet && typeof packet === "object") {
          if (packet.message) {
            applySettings(packet.message);
          } else if (!packet.command && !packet.error) {
            applySettings(packet);
          }
        }
      } catch (err) {
        console.error("[O2Jam Predictor] Error parsing commands message:", err);
      }
    };

    commandsSocket.onclose = () => {
      setTimeout(connectCommandsSocket, 2500);
    };

    commandsSocket.onerror = () => {
      commandsSocket.close();
    };
  } catch {
    setTimeout(connectCommandsSocket, 2500);
  }
}

// ── 2. ONNX Runtime Initialization ──────────────────────────────────────────
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

// ── 3. Parse .osu Text & Build Discrete (T, 7) Grid ──────────────────────────
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
      const csNum = parseInt(circleSize, 10);
      const is7K = (mode === "3" || mode === 3) && csNum === 7;
      if (!is7K) {
        return { is7KMania: false, circleSize: csNum || circleSize || "?" };
      }
      isHitObjects = true;
      continue;
    }

    if (!isHitObjects) {
      if (line.startsWith("Mode:")) mode = line.slice(5).trim();
      else if (line.startsWith("CircleSize:")) circleSize = line.slice(11).trim();
    } else {
      if (line.startsWith("[")) break;
      const parts = line.split(",");
      if (parts.length >= 4) {
        const x = parseInt(parts[0], 10);
        const t = parseInt(parts[2], 10);
        const nType = parseInt(parts[3], 10);
        const col = Math.min(Math.max(Math.floor((x * 7) / 512), 0), 6);
        let endT = t;
        if ((nType & 128) && parts.length > 5) {
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
    return { is7KMania: false, circleSize: csNum || circleSize || "?" };
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

  return { is7KMania: true, grid, tFrames };
}

// ── 4. ONNX Model Inference ─────────────────────────────────────────────────
async function runPrediction(parsed) {
  if (!ortSession || !parsed.grid) return null;

  const tensor = new ort.Tensor("uint8", parsed.grid, [1, parsed.tFrames, 7]);
  const t0 = performance.now();
  const results = await ortSession.run({ note_grid: tensor });
  const latencyMs = (performance.now() - t0).toFixed(1);

  const level = results.predicted_level.data[0];
  return { level, latencyMs };
}

// ── 5. Fetch Active Beatmap from Tosu ────────────────────────────────────────
async function fetchActiveBeatmap() {
  const host = getHost();
  const endpoint = `http://${host}/files/beatmap/file?_=${Date.now()}`;
  const resp = await fetch(endpoint, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return await resp.text();
}

// ── 6. Rate & Mod Extraction ────────────────────────────────────────────────
function extractRateAndMods(data) {
  const isPlaying = Number(data?.state?.number ?? data?.status?.number ?? 0) === 2;

  const rateCandidate = Number(data?.gameplay?.playbackRate);
  if (isPlaying && Number.isFinite(rateCandidate) && rateCandidate > 0 && Math.abs(rateCandidate - 1.0) > 0.01) {
    const formatted = rateCandidate.toFixed(2).replace(/\.?0+$/, "");
    return { rate: rateCandidate, label: `${formatted}x`, type: "rate" };
  }

  const mods = isPlaying
    ? (data?.play?.mods || data?.gameplay?.mods || data?.menu?.mods)
    : (data?.menu?.mods || data?.play?.mods || data?.gameplay?.mods);

  const name = typeof mods?.name === "string" ? mods.name.toUpperCase() : "";
  const num = Number(mods?.number ?? mods?.num ?? 0);

  if (name.includes("NC") || (num & 512) !== 0) {
    return { rate: 1.5, label: "NC", type: "dt" };
  }
  if (name.includes("DT") || (num & 64) !== 0) {
    return { rate: 1.5, label: "DT", type: "dt" };
  }
  if (name.includes("DC")) {
    return { rate: 0.75, label: "DC", type: "ht" };
  }
  if (name.includes("HT") || (num & 256) !== 0) {
    return { rate: 0.75, label: "HT", type: "ht" };
  }

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

// ── 7. Recompute Beatmap Difficulty ─────────────────────────────────────────
async function processCurrentBeatmap() {
  if (isPredicting) return;
  isPredicting = true;
  updateStatus("Reading map...", "loading");

  try {
    const osuText = await fetchActiveBeatmap();
    const parsed = parseOsuBeatmap(osuText, currentRateInfo.rate);

    if (!parsed.is7KMania) {
      isCurrentMap7K = false;
      currentRawLevel = null;
      levelDisplay.textContent = "--";
      updateRateBadge({ rate: 1.0, label: "", type: "" });
      updateStatus(`Non-7K (${parsed.circleSize}K)`, "idle");

      if (settings.hideNon7K && overlayContainer) {
        overlayContainer.style.display = "none";
      }
      return;
    }

    isCurrentMap7K = true;
    if (overlayContainer) {
      overlayContainer.style.display = "flex";
    }

    if (parsed.error) {
      currentRawLevel = null;
      levelDisplay.textContent = "--";
      updateRateBadge({ rate: 1.0, label: "", type: "" });
      updateStatus(parsed.error, "warning");
      return;
    }

    updateRateBadge(currentRateInfo);
    updateStatus("Predicting...", "loading");

    const predResult = await runPrediction(parsed);

    if (predResult) {
      currentRawLevel = predResult.level;

      levelDisplay.textContent = settings.showDecimal
        ? currentRawLevel.toFixed(1)
        : Math.round(currentRawLevel);

      levelDisplay.classList.remove("pulse");
      void levelDisplay.offsetWidth;
      levelDisplay.classList.add("pulse");

      const rateInfoText = currentRateInfo.label ? ` [${currentRateInfo.label}]` : "";
      updateStatus(`7K${rateInfoText} (${predResult.latencyMs}ms)`, "");
    }
  } catch (err) {
    console.error("[O2Jam Predictor] Error processing beatmap:", err);
    updateStatus("Error reading map", "warning");
  } finally {
    isPredicting = false;
  }
}

// ── 8. Tosu WebSocket Stream (V2) ──────────────────────────────────────────
function connectToTosu() {
  const host = getHost();
  const counterPath = getCounterFolderName();
  const v2Url = `ws://${host}/websocket/v2?l=${encodeURI(counterPath)}`;

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

      // Update Song Title if present
      const title = bm?.title || "";
      const version = bm?.version || "";
      if (title || version) {
        currentSongTitle = version ? `${title} [${version}]` : title;
        if (songTitleEl) songTitleEl.textContent = currentSongTitle;
        if (songInfoEl) {
          songInfoEl.style.display = settings.showSongTitle && currentSongTitle ? "block" : "none";
        }
      }

      // Check map / rate changes
      const md5 = (bm?.checksum || bm?.md5 || "").toLowerCase();
      const id = bm?.id || 0;
      const rateInfo = extractRateAndMods(data);
      const processKey = `${md5 || id}:${title}:${version}@${rateInfo.rate.toFixed(2)}`;

      if (processKey !== lastProcessedKey) {
        lastProcessedKey = processKey;
        currentRateInfo = rateInfo;
        processCurrentBeatmap();
      }
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

// ── 9. Bootstrap ───────────────────────────────────────────────────────────
async function main() {
  await loadInitialDefaults();
  connectCommandsSocket();
  await initONNX();
  connectToTosu();

  window.addEventListener("focus", () => {
    requestSettings();
  });
}

main();
