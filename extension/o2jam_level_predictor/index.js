/**
 * O2Jam 7K Level Predictor - Tosu Static Extension
 * Connects to Tosu WebSocket v2, parses active .osu file in-browser,
 * supports HT/DT/custom playback rate scaling, runs model.onnx via ONNX Runtime Web,
 * and manages user level feedback with Tosu disk settings persistence.
 */

let ortSession = null;
let lastProcessedKey = null;
let isPredicting = false;

// Active Beatmap Metadata & State
const currentMap = {
  md5: "",
  beatmap_id: 0,
  beatmapset_id: 0,
  title: "",
  artist: "",
  version: "",
  creator: "",
  predictedLevel: null,
  rate: 1.0,
  modText: "",
  noteCount: 0,
};

// Persistent Suggestions Store (key -> record)
const suggestionsStore = new Map();

// DOM Elements
const levelDisplay = document.getElementById("level-display");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const rateBadge = document.getElementById("rate-badge");
const feedbackForm = document.getElementById("feedback-form");
const suggestInput = document.getElementById("suggest-input");
const suggestBtn = document.getElementById("suggest-btn");

function updateStatus(text, dotClass = "idle") {
  if (statusText) statusText.textContent = text;
  if (statusDot) statusDot.className = `status-dot ${dotClass}`.trim();
  console.log(`[O2Jam Predictor] ${text}`);
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
  // 1. Direct playbackRate if given by tosu
  const rateCandidate = Number(data?.gameplay?.playbackRate || data?.playbackRate);
  if (Number.isFinite(rateCandidate) && rateCandidate > 0 && Math.abs(rateCandidate - 1.0) > 0.01) {
    const formatted = rateCandidate.toFixed(2).replace(/\.?0+$/, "");
    return { rate: rateCandidate, modText: `${formatted}x` };
  }

  // 2. Mod objects (gameplay / play / menu)
  const mods = data?.play?.mods || data?.gameplay?.mods || data?.menu?.mods;
  const name = typeof mods?.name === "string" ? mods.name.toUpperCase() : "";
  const num = Number(mods?.number ?? mods?.num ?? 0);

  if (name.includes("NC") || (num & 512) !== 0) {
    return { rate: 1.5, modText: "NC (1.5x)" };
  }
  if (name.includes("DT") || (num & 64) !== 0) {
    return { rate: 1.5, modText: "DT (1.5x)" };
  }
  if (name.includes("DC")) {
    return { rate: 0.75, modText: "DC (0.75x)" };
  }
  if (name.includes("HT") || (num & 256) !== 0) {
    return { rate: 0.75, modText: "HT (0.75x)" };
  }

  // 3. Lazer mod array inspection
  const modArray = Array.isArray(mods?.array) ? mods.array : (Array.isArray(mods) ? mods : []);
  for (const m of modArray) {
    const acronym = String(m?.acronym || m || "").toUpperCase();
    const speedChange = Number(m?.settings?.speed_change);
    if (Number.isFinite(speedChange) && speedChange > 0) {
      return { rate: speedChange, modText: `${speedChange}x` };
    }
    if (acronym === "NC") return { rate: 1.5, modText: "NC (1.5x)" };
    if (acronym === "DT") return { rate: 1.5, modText: "DT (1.5x)" };
    if (acronym === "DC") return { rate: 0.75, modText: "DC (0.75x)" };
    if (acronym === "HT") return { rate: 0.75, modText: "HT (0.75x)" };
  }

  return { rate: 1.0, modText: "" };
}

// ── 6. Persistence: Tosu Settings API + LocalStorage ────────────────────────
function getSuggestionKey(md5, rate) {
  if (!md5) return "";
  const r = Number(rate) || 1.0;
  return Math.abs(r - 1.0) < 0.01 ? md5 : `${md5}@${r.toFixed(2)}`;
}

async function loadStoredSuggestions() {
  // 1. Load from localStorage first (instant)
  try {
    const local = localStorage.getItem("o2jam_user_suggestions");
    if (local) {
      const parsed = JSON.parse(local);
      for (const [k, v] of Object.entries(parsed)) {
        suggestionsStore.set(k, v);
      }
    }
  } catch (e) {
    console.warn("[O2Jam Predictor] LocalStorage load error:", e);
  }

  // 2. Load from Tosu settings endpoint
  try {
    const folder = getCounterFolderName();
    const res = await fetch(`http://${getHost()}/api/counters/settings/${encodeURIComponent(folder)}`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      const raw = data?.values?.userSuggestions;
      if (typeof raw === "string") {
        const parsed = JSON.parse(raw);
        for (const [k, v] of Object.entries(parsed)) {
          suggestionsStore.set(k, v);
        }
      } else if (raw && typeof raw === "object") {
        for (const [k, v] of Object.entries(raw)) {
          suggestionsStore.set(k, v);
        }
      }
    }
  } catch (e) {
    // Non-blocking if Tosu settings endpoint is unavailable
  }
}

async function saveStoredSuggestions() {
  const obj = Object.fromEntries(suggestionsStore);
  const serialized = JSON.stringify(obj);

  // 1. Sync to localStorage
  try {
    localStorage.setItem("o2jam_user_suggestions", serialized);
  } catch (e) {
    console.warn("[O2Jam Predictor] LocalStorage save error:", e);
  }

  // 2. Sync to Tosu settings API (disk persistence)
  try {
    const folder = getCounterFolderName();
    let existingValues = {};
    try {
      const getRes = await fetch(`http://${getHost()}/api/counters/settings/${encodeURIComponent(folder)}`, {
        cache: "no-store",
      });
      if (getRes.ok) {
        const getJson = await getRes.json();
        existingValues = getJson?.values || {};
      }
    } catch {}

    // Preserve any existing settings keys so future UI settings will not be wiped
    const payload = Object.entries(existingValues)
      .filter(([k]) => k !== "userSuggestions")
      .map(([uniqueID, value]) => ({ uniqueID, value }));

    payload.push({ uniqueID: "userSuggestions", value: serialized });

    await fetch(`http://${getHost()}/api/counters/settings/${encodeURIComponent(folder)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    console.log(`[O2Jam Predictor] Successfully saved ${suggestionsStore.size} suggestions to Tosu disk settings`);
  } catch (e) {
    console.error("[O2Jam Predictor] Failed to save to Tosu settings:", e);
  }
}

function syncSuggestionUI() {
  if (!suggestInput || !suggestBtn) return;
  const key = getSuggestionKey(currentMap.md5, currentMap.rate);
  const existing = suggestionsStore.get(key) || (Math.abs(currentMap.rate - 1.0) < 0.01 ? null : suggestionsStore.get(currentMap.md5));

  if (existing && existing.suggested) {
    suggestInput.value = existing.suggested;
    suggestBtn.textContent = "Saved";
    suggestBtn.classList.add("saved");
  } else {
    suggestInput.value = "";
    suggestBtn.textContent = "Save";
    suggestBtn.classList.remove("saved");
  }
}

// ── 7. Main Recompute Handler ───────────────────────────────────────────────
async function processCurrentBeatmap() {
  if (isPredicting) return;
  isPredicting = true;
  updateStatus("Reading map...", "loading");

  try {
    const osuText = await fetchActiveBeatmap();
    const parsed = parseOsuBeatmap(osuText, currentMap.rate);

    if (!parsed.is7KMania) {
      levelDisplay.textContent = "--";
      if (rateBadge) rateBadge.hidden = true;
      updateStatus(`Non-7K (${parsed.circleSize}K)`, "idle");
      syncSuggestionUI();
      return;
    }

    if (parsed.error) {
      levelDisplay.textContent = "--";
      if (rateBadge) rateBadge.hidden = true;
      updateStatus(parsed.error, "warning");
      syncSuggestionUI();
      return;
    }

    // Update rate badge
    if (rateBadge) {
      if (Math.abs(currentMap.rate - 1.0) > 0.01) {
        rateBadge.textContent = currentMap.modText || `${currentMap.rate.toFixed(2)}x`;
        rateBadge.hidden = false;
      } else {
        rateBadge.hidden = true;
      }
    }

    currentMap.noteCount = parsed.noteCount;
    updateStatus("Predicting...", "loading");
    const predResult = await runPrediction(parsed);

    if (predResult) {
      const roundedLevel = Math.round(predResult.level);
      currentMap.predictedLevel = roundedLevel;

      levelDisplay.textContent = roundedLevel;
      levelDisplay.classList.remove("pulse");
      void levelDisplay.offsetWidth;
      levelDisplay.classList.add("pulse");

      const rateInfo = Math.abs(currentMap.rate - 1.0) > 0.01 ? ` [${currentMap.modText}]` : "";
      updateStatus(`7K${rateInfo} (${predResult.latencyMs}ms)`, "");
      console.log(`[O2Jam Predictor] Level: ${predResult.level.toFixed(2)} (Round: ${roundedLevel}) @ rate ${currentMap.rate}`);
    }

    syncSuggestionUI();
  } catch (err) {
    console.error("[O2Jam Predictor] Error processing beatmap:", err);
    updateStatus("Error reading map", "warning");
  } finally {
    isPredicting = false;
  }
}

// ── 8. User Feedback Event Handlers ─────────────────────────────────────────
if (feedbackForm) {
  feedbackForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentMap.md5) return;

    const rawVal = suggestInput.value.trim();
    const numVal = parseInt(rawVal, 10);

    // Sanitize: only integer between 10 and 185
    if (isNaN(numVal) || numVal < 10 || numVal > 185) {
      suggestInput.classList.add("invalid");
      setTimeout(() => suggestInput.classList.remove("invalid"), 400);
      return;
    }

    const key = getSuggestionKey(currentMap.md5, currentMap.rate);
    const record = {
      beatmap_id: currentMap.beatmap_id || 0,
      beatmapset_id: currentMap.beatmapset_id || 0,
      md5: currentMap.md5,
      title: currentMap.title,
      artist: currentMap.artist,
      version: currentMap.version,
      creator: currentMap.creator,
      predicted: currentMap.predictedLevel,
      suggested: numVal,
      rate: currentMap.rate,
      modText: currentMap.modText,
      osu_link: currentMap.beatmap_id
        ? `https://osu.ppy.sh/b/${currentMap.beatmap_id}`
        : currentMap.beatmapset_id
        ? `https://osu.ppy.sh/beatmapsets/${currentMap.beatmapset_id}`
        : "",
      noteCount: currentMap.noteCount,
      updatedAt: new Date().toISOString(),
    };

    suggestionsStore.set(key, record);
    suggestBtn.textContent = "Saved!";
    suggestBtn.classList.add("saved");

    await saveStoredSuggestions();
  });
}

if (suggestInput) {
  suggestInput.addEventListener("input", () => {
    suggestBtn.textContent = "Save";
    suggestBtn.classList.remove("saved");
  });
}

// ── 9. Tosu WebSocket Connections ───────────────────────────────────────────
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
      const setId = bm?.setId || bm?.set_id || 0;
      const title = bm?.title || "";
      const artist = bm?.artist || "";
      const version = bm?.version || "";
      const creator = bm?.creator || "";

      const { rate, modText } = extractRateAndMods(data);

      const processKey = `${md5 || id}:${title}:${version}@${rate.toFixed(2)}`;

      if (processKey === lastProcessedKey) {
        return;
      }

      lastProcessedKey = processKey;

      // Update current map state
      currentMap.md5 = md5;
      currentMap.beatmap_id = id;
      currentMap.beatmapset_id = setId;
      currentMap.title = title;
      currentMap.artist = artist;
      currentMap.version = version;
      currentMap.creator = creator;
      currentMap.rate = rate;
      currentMap.modText = modText;

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
  await loadStoredSuggestions();
  await initONNX();
  connectToTosu();
}

main();
