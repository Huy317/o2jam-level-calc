# Tosu Extension & Real-Time In-Game Overlay

## 1. Overview
The **O2Jam 7K Level Predictor** includes an in-game HUD overlay designed for [Tosu](https://github.com/KotRikD/tosu) (the standard open-source osu! memory reader).

It renders a sleek glassmorphic HUD card displaying the predicted O2Jam level and updates in real-time as you switch maps in song select or gameplay.

---

## 2. Features

- **Sub-15ms Live Evaluation:** Runs `model.onnx` entirely in-browser via [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) using SIMD-accelerated WebAssembly.
- **Difficulty Strain Timeline Graph:** Renders an instantaneous difficulty strain sparkline curve across song duration with peak spike indicator and real-time live playhead tracking.
- **Tosu Native Settings Support:** Fully configurable from the Tosu Web Dashboard via native `settings.json` and `/websocket/commands` bidirectional synchronization (zero overlay refresh required).
- **100% Offline Bundle:** Includes all required `.wasm` and `.mjs` binaries locally. No internet connection or external CDN required.
- **Real-Time Speed & Mod Scaling (DT / HT / Custom):**
  - **Double Time / Nightcore (1.5x DT/NC):** Compresses time intervals by $1.5\times$, evaluating physical note density and burst speeds to predict the accelerated difficulty rating. Displays a soft red/rose `DT` badge.
  - **Half Time / Daycore (0.75x HT/DC):** Expands time intervals by $0.75\times$ to reflect lowered density. Displays an emerald green `HT` badge.
  - **Custom Rates:** Supports osu!lazer fractional rate multipliers (e.g. `1.2x`, `1.4x`).
  - **Instant Reset:** Automatically clears badges and restores baseline NoMod ratings when mods are toggled off.
- **Mouse-Transparent In-Game Overlay:** Designed specifically for game overlays (`WS_EX_TRANSPARENT`), allowing players to hit notes without mouse clicks ever stealing focus from the game.

---

## 3. Configuration & Tosu Settings

The overlay supports native Tosu settings configurable through the Tosu Web Manager:

| Option | Type | Default | Description |
|---|---|---|---|
| `showGraph` | Checkbox | `true` | Display instantaneous difficulty strain timeline graph below the level badge. |
| `showSongTitle` | Checkbox | `true` | Display beatmap title and difficulty name. |
| `showDecimal` | Checkbox | `false` | Show decimal level precision (e.g. `Lv. 94.2` instead of `Lv. 94`). |
| `hideNon7K` | Checkbox | `false` | Hide overlay widget entirely when playing non-7K maps. |
| `graphHeight` | Number | `32` | Height of the difficulty strain timeline graph in pixels (16 to 80). |
| `overlayScale` | Number | `1.0` | Scale multiplier for the entire overlay widget (0.5 to 2.0). |
| `overlayOpacity` | Number | `0.85` | Background opacity for the glass card (0.1 to 1.0). |
| `accentColor` | Color | `#ff9f43` | Accent color for level number, rate badge, and graph glow. |

---

## 4. Installation

### Quick Install (ZIP):
1. Download `o2jam_level_predictor.zip` from the Releases page.
2. Extract the archive into your Tosu `static/` directory:
   ```
   tosu/
   ├── static/
   │   └── o2jam_level_predictor/
   │       ├── index.html
   │       ├── index.js
   │       ├── style.css
   │       ├── model.onnx
   │       ├── ort.min.js
   │       ├── settings.json
   │       └── ...
   ```
3. Open or reload your Tosu dashboard / overlay URL:
   ```
   http://127.0.0.1:24050/o2jam_level_predictor/
   ```
4. If using Tosu In-Game Overlay, enable `o2jam_level_predictor` in Tosu overlay manager.

---

## 5. Technical Architecture: WebSocket v2 & WASM Discretization

```mermaid
flowchart TD
    A[osu! Game Memory] -->|Tosu WebSocket v2| B[index.js]
    B -->|Check md5 & mods| C{Map / Mod Changed?}
    C -- Yes --> D[Fetch active .osu file via HTTP]
    D --> E[Scale timing by playbackRate]
    E --> F[Discretize into 20ms uint8 grid]
    F --> G[ONNX Runtime Web - Wasm SIMD]
    G --> H[Update DOM: Level, Badge, & Canvas Graph]
    C -- No --> I[Update Live Playhead Cursor]
```

### Mod & Rate Extraction:
The extension checks both gameplay and song-select states:
- In gameplay (`state.number == 2`): Reads `gameplay.playbackRate` and `play.mods`.
- In song select: Prioritizes `menu.mods` to instantly reflect user mod clicks.

### Temporal Discretization:
Given raw note timestamp $t$ and first note $t_0$, the frame index is scaled directly:
$$\text{frame} = \left\lfloor \frac{t - t_0}{20 \times \text{rate}} \right\rfloor$$
Notes naturally compress together on DT ($1.5\times$), directly exciting the CNN's dilated receptive fields without requiring custom mod heuristics.
