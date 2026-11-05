/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Modern HUD Overlay with fixed-width difficulty graph, editor/gameplay progress tracking & Tosu native settings.
 */

// ── Default Settings ────────────────────────────────────────────────────────
const settings = {
  showGraph: true,
  showSongTitle: true,
  showDecimal: false,
  showTimeLabels: true,
  hideNon7K: false,
  graphHeight: 80,
  overlayScale: 1.0,
  overlayOpacity: 0.85,
  accentColor: "#ff5e7e",
};

// ── Runtime State ───────────────────────────────────────────────────────────
let ortSession = null;
let lastProcessedKey = null;
let isPredicting = false;
let currentRateInfo = { rate: 1.0, label: "", type: "" };
let currentRawLevel = null;
let currentStrainCurve = null;
let currentMapTiming = { firstSec: 0, durationSec: 0 };
let currentSongTitle = "";
let isCurrentMap7K = true;
let commandsSocket = null;
let currentProgressRatio = null;
let currentLiveSec = null;

// ── DOM Elements ────────────────────────────────────────────────────────────
const overlayContainer = document.getElementById("overlay-container");
const songInfoEl = document.getElementById("song-info");
const songTitleEl = document.getElementById("song-title");
const levelDisplay = document.getElementById("level-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const rateBadge = document.getElementById("rate-badge");
const graphBoxEl = document.getElementById("graph-box");
const strainCanvas = document.getElementById("strain-canvas");
const timeRowEl = document.getElementById("time-row");
const timeCurrEl = document.getElementById("time-curr");
const timePeakEl = document.getElementById("time-peak");
const timeTotalEl = document.getElementById("time-total");

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
  if (isNaN(num)) return `rgba(255, 94, 126, ${alpha})`;
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

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
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

  // Parse each known setting with type safety
  if ("showGraph" in raw) settings.showGraph = parseBoolean(raw.showGraph, settings.showGraph);
  if ("showSongTitle" in raw) settings.showSongTitle = parseBoolean(raw.showSongTitle, settings.showSongTitle);
  if ("showDecimal" in raw) settings.showDecimal = parseBoolean(raw.showDecimal, settings.showDecimal);
  if ("showTimeLabels" in raw) settings.showTimeLabels = parseBoolean(raw.showTimeLabels, settings.showTimeLabels);
  if ("hideNon7K" in raw) settings.hideNon7K = parseBoolean(raw.hideNon7K, settings.hideNon7K);
  if ("graphHeight" in raw) {
    const num = Number(raw.graphHeight);
    if (Number.isFinite(num) && num >= 30) settings.graphHeight = num;
  }
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
  root.setProperty("--graph-height", `${Math.max(40, settings.graphHeight)}px`);

  // Graph Box & Time Row Visibility
  if (graphBoxEl) {
    graphBoxEl.style.display = settings.showGraph ? "flex" : "none";
  }
  if (timeRowEl) {
    timeRowEl.style.display = settings.showTimeLabels ? "flex" : "none";
  }

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
    if (settings.hideNon7K && !isCurrentMap7K) {
      overlayContainer.style.display = "none";
    } else {
      overlayContainer.style.display = "flex";
    }
  }

  // Redraw graph with new dimensions and colors
  if (currentStrainCurve && settings.showGraph) {
    renderDifficultyGraph(currentStrainCurve, currentProgressRatio, currentLiveSec);
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
      console.log(`[O2Jam Predictor] Connected to Tosu Commands WS: ${url}`);
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

        console.log("[O2Jam Predictor] Received settings update from Tosu:", packet);

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

// ── 2. Difficulty Graph Canvas Renderer ─────────────────────────────────────
function traceSmoothCurve(ctx, points) {
  if (points.length < 2) return;
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i === 0 ? 0 : i - 1];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2 < points.length ? i + 2 : i + 1];

    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
}

function getInterpolatedCurveY(points, curX) {
  if (points.length === 0) return 0;
  if (curX <= points[0].x) return points[0].y;
  if (curX >= points[points.length - 1].x) return points[points.length - 1].y;

  for (let i = 0; i < points.length - 1; i++) {
    if (curX >= points[i].x && curX <= points[i + 1].x) {
      const span = points[i + 1].x - points[i].x;
      const t = span > 0 ? (curX - points[i].x) / span : 0;
      return points[i].y + t * (points[i + 1].y - points[i].y);
    }
  }
  return points[points.length - 1].y;
}

function renderDifficultyGraph(strainCurve, progressRatio = null, liveSec = null) {
  if (!strainCanvas || !settings.showGraph || !strainCurve || strainCurve.length === 0) {
    return;
  }

  // Crisp 2x internal rendering for retina / 1440p displays
  const targetW = 584;
  const targetH = Math.round(settings.graphHeight * 2);
  if (strainCanvas.width !== targetW || strainCanvas.height !== targetH) {
    strainCanvas.width = targetW;
    strainCanvas.height = targetH;
  }

  const ctx = strainCanvas.getContext("2d");
  const w = strainCanvas.width;
  const h = strainCanvas.height;

  ctx.clearRect(0, 0, w, h);

  // Layout bounds & padding
  const padX = 10;
  const padTop = 14;
  const padBottom = 10;
  const drawW = w - 2 * padX;
  const drawH = h - padTop - padBottom;

  // Find peak strain
  let maxStrain = 0;
  let peakIdx = 0;
  for (let i = 0; i < strainCurve.length; i++) {
    if (strainCurve[i] > maxStrain) {
      maxStrain = strainCurve[i];
      peakIdx = i;
    }
  }
  if (maxStrain <= 0.001) maxStrain = 1.0;

  const len = strainCurve.length;
  const points = [];
  for (let i = 0; i < len; i++) {
    const normX = len > 1 ? i / (len - 1) : 0;
    const x = padX + normX * drawW;
    const normY = Math.min(1.0, Math.max(0.0, strainCurve[i] / maxStrain));
    const y = (h - padBottom) - normY * drawH;
    points.push({ x, y });
  }

  // 1. Soft Background Area Gradient (Full Chart)
  const baseGrad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
  baseGrad.addColorStop(0, hexToRgba(settings.accentColor, 0.22));
  baseGrad.addColorStop(1, hexToRgba(settings.accentColor, 0.01));

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x, h - padBottom + 2);
  ctx.lineTo(points[0].x, points[0].y);
  traceSmoothCurve(ctx, points);
  ctx.lineTo(points[len - 1].x, h - padBottom + 2);
  ctx.closePath();
  ctx.fillStyle = baseGrad;
  ctx.fill();
  ctx.restore();

  // 2. Played Progress Area Fill (Clipped to Current Playhead)
  const hasProgress = progressRatio !== null && progressRatio >= 0 && progressRatio <= 1;
  const progressX = hasProgress ? padX + progressRatio * drawW : null;

  if (hasProgress && progressX > padX) {
    const playedGrad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
    playedGrad.addColorStop(0, hexToRgba(settings.accentColor, 0.48));
    playedGrad.addColorStop(1, hexToRgba(settings.accentColor, 0.06));

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, progressX, h);
    ctx.clip();

    ctx.beginPath();
    ctx.moveTo(points[0].x, h - padBottom + 2);
    ctx.lineTo(points[0].x, points[0].y);
    traceSmoothCurve(ctx, points);
    ctx.lineTo(points[len - 1].x, h - padBottom + 2);
    ctx.closePath();
    ctx.fillStyle = playedGrad;
    ctx.fill();
    ctx.restore();
  }

  // 3. Smooth Curve Stroke
  ctx.save();
  ctx.beginPath();
  traceSmoothCurve(ctx, points);
  ctx.strokeStyle = settings.accentColor;
  ctx.lineWidth = 4.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.restore();

  // 4. Peak Difficulty Spike Beacon
  const peakPt = points[peakIdx];
  if (peakPt) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(peakPt.x, peakPt.y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = settings.accentColor;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();
  }

  // 5. Active Playhead Cursor (Editor & In-game)
  if (hasProgress && progressX !== null) {
    const progressY = getInterpolatedCurveY(points, progressX);

    // Dashed Vertical Guide Line
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(progressX, padTop - 4);
    ctx.lineTo(progressX, h - padBottom + 4);
    ctx.stroke();
    ctx.restore();

    // White Cursor Dot on Top of Curve
    ctx.save();
    ctx.beginPath();
    ctx.arc(progressX, progressY, 6.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = settings.accentColor;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = settings.accentColor;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();
  }

  // 6. Update Time Labels
  if (timeRowEl && settings.showTimeLabels) {
    const displayLiveSec = liveSec !== null ? liveSec : 0;
    if (timeCurrEl) timeCurrEl.textContent = formatTime(displayLiveSec);

    // Calculate exact osu! editor timestamp for the peak spike
    const peakEditorSec = currentMapTiming.firstSec + peakIdx * currentRateInfo.rate;
    if (timePeakEl) {
      const peakLevelStr = currentRawLevel ? ` · Lv.${Math.round(currentRawLevel)}` : "";
      timePeakEl.textContent = `★ Peak ${formatTime(peakEditorSec)}${peakLevelStr}`;
    }

    const totalSec = currentMapTiming.firstSec + currentMapTiming.durationSec * currentRateInfo.rate;
    if (timeTotalEl) timeTotalEl.textContent = formatTime(totalSec);
  }
}

// ── 3. ONNX Runtime Initialization ──────────────────────────────────────────
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

// ── 4. Parse .osu Text & Build Discrete (T, 7) Grid ──────────────────────────
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
    firstT,
    durationMs,
  };
}

// ── 5. ONNX Model Inference (Level & Strain Curve) ──────────────────────────
async function runPrediction(parsed) {
  if (!ortSession || !parsed.grid) return null;

  const tensor = new ort.Tensor("uint8", parsed.grid, [1, parsed.tFrames, 7]);
  const t0 = performance.now();
  const results = await ortSession.run({ note_grid: tensor });
  const latencyMs = (performance.now() - t0).toFixed(1);

  const level = results.predicted_level.data[0];
  const strainCurve = results.strain_curve ? results.strain_curve.data : null;

  return { level, strainCurve, latencyMs };
}

// ── 6. Fetch Active Beatmap from Tosu ────────────────────────────────────────
async function fetchActiveBeatmap() {
  const host = getHost();
  const endpoint = `http://${host}/files/beatmap/file?_=${Date.now()}`;
  const resp = await fetch(endpoint, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return await resp.text();
}

// ── 7. Rate & Mod Extraction ────────────────────────────────────────────────
function extractRateAndMods(data) {
  const isPlaying = Number(data?.state?.number ?? data?.status?.number ?? 0) === 2;

  // Direct playbackRate if given by tosu during active play
  const rateCandidate = Number(data?.gameplay?.playbackRate);
  if (isPlaying && Number.isFinite(rateCandidate) && rateCandidate > 0 && Math.abs(rateCandidate - 1.0) > 0.01) {
    const formatted = rateCandidate.toFixed(2).replace(/\.?0+$/, "");
    return { rate: rateCandidate, label: `${formatted}x`, type: "rate" };
  }

  // Mod candidates: prioritize menu.mods in song select / menus
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

// ── 8. Robust Current Song Time Extraction (Editor, Play & Menu) ─────────────
function extractCurrentSongTimeMs(data) {
  const candidates = [
    data?.beatmap?.time?.live,
    data?.beatmap?.time?.current,
    data?.menu?.bm?.time?.current,
    data?.play?.time?.current,
    data?.play?.time,
  ];

  for (const value of candidates) {
    const num = Number(value);
    if (Number.isFinite(num)) {
      return num;
    }
  }

  return null;
}

// ── 9. Recompute Beatmap Difficulty ─────────────────────────────────────────
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
      currentStrainCurve = null;
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
      currentStrainCurve = null;
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
      currentStrainCurve = predResult.strainCurve;
      currentMapTiming = {
        firstSec: parsed.firstT / 1000,
        durationSec: (parsed.durationMs / 1000) / currentRateInfo.rate,
      };

      levelDisplay.textContent = settings.showDecimal
        ? currentRawLevel.toFixed(1)
        : Math.round(currentRawLevel);

      levelDisplay.classList.remove("pulse");
      void levelDisplay.offsetWidth;
      levelDisplay.classList.add("pulse");

      if (settings.showGraph) {
        renderDifficultyGraph(currentStrainCurve, currentProgressRatio, currentLiveSec);
      }

      const rateInfoText = currentRateInfo.label ? ` [${currentRateInfo.label}]` : "";
      updateStatus(`7K${rateInfoText} (${predResult.latencyMs}ms)`, "");
      console.log(`[O2Jam Predictor] Level: ${predResult.level.toFixed(2)} @ rate ${currentRateInfo.rate}`);
    }
  } catch (err) {
    console.error("[O2Jam Predictor] Error processing beatmap:", err);
    updateStatus("Error reading map", "warning");
  } finally {
    isPredicting = false;
  }
}

// ── 10. Tosu WebSocket Stream (V2) ──────────────────────────────────────────
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

      // Extract Song Title
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

      // Live Scrubber / Playhead Tracking in Editor, Gameplay, or Menus
      const liveTimeMs = extractCurrentSongTimeMs(data);
      if (typeof liveTimeMs === "number" && currentStrainCurve && currentMapTiming.durationSec > 0) {
        const liveSec = liveTimeMs / 1000;
        const elapsedSec = (liveTimeMs - (currentMapTiming.firstSec * 1000)) / (1000 * currentRateInfo.rate);
        const progressRatio = Math.max(0, Math.min(1, elapsedSec / currentMapTiming.durationSec));

        currentProgressRatio = progressRatio;
        currentLiveSec = liveSec;

        if (settings.showGraph) {
          renderDifficultyGraph(currentStrainCurve, progressRatio, liveSec);
        }
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

// ── 11. Bootstrap ───────────────────────────────────────────────────────────
async function main() {
  await loadInitialDefaults();
  connectCommandsSocket();
  await initONNX();
  connectToTosu();

  // On window focus, ask Tosu for latest settings from values.json
  window.addEventListener("focus", () => {
    requestSettings();
  });
}

main();
