# O2Jam 7K Level Predictor & Tosu! Overlay

An end-to-end machine learning pipeline that predicts the human-assigned **O2Jam difficulty level (Lv. 1 to 180+)** of 7-key osu!mania charts directly from raw note data. Includes a lightweight, real-time in-game overlay extension for [tosu!](https://github.com/tosuapp/tosu) powered by **ONNX Runtime Web (WebAssembly)**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![ONNX](https://img.shields.io/badge/ONNX-WebAssembly-green)
![osu!mania](https://img.shields.io/badge/osu!mania-7K-orange)
![tosu!](https://img.shields.io/badge/tosu!-v2_Compatible-purple)

---

## Features

- **Accurate Difficulty Estimation:** Trained on 2,557 unique O2Jam 7K charts across official and community releases (DMJAM / O2EZ) with a median prediction error $\le 4.7$ levels ($R = 0.964$).
- **Sub-10ms Real-Time Inference:** Operates on variable-length charts in milliseconds without needing to trim or pad.
- **Zero Python Runtime for Players:** The tosu! extension runs 100% inside your browser or OBS via WebAssembly — no Python, PyTorch, or GPU setup needed for end users.
- **Standalone CLI:** Test any `.osu` file from your command line in ~30ms.

---

## Quick Start: Tosu In-Game Overlay

The in-game overlay automatically reads the currently selected or playing beatmap from `tosu!`, parses the 7K note layout in-browser, and displays the estimated O2Jam level.

```
┌─────────────────────────┐
│  Level. 68              │
│  ● 7K (6.2ms)           │
└─────────────────────────┘
```

### Installation
1. Copy the `extension/o2jam_level_predictor` folder directly into your tosu `static` directory:
   ```text
   tosu/
   └── static/
       └── o2jam_level_predictor/
           ├── index.html
           ├── index.js
           ├── style.css
           ├── model.onnx
           └── ... (WASM files)
   ```
2. Start **osu!** and **tosu!**.
3. Open the overlay:
   - **In your browser:** `http://127.0.0.1:24050/o2jam_level_predictor`
   - **In OBS:** Add a **Browser Source** pointing to `http://127.0.0.1:24050/o2jam_level_predictor` (Recommended dimensions: `170 x 65`).

---

## Quick Start: CLI Prediction

You can test any `.osu` chart instantly from your terminal:

```bash
# 1. Install lightweight runtime dependencies
pip install onnxruntime numpy

# 2. Run prediction on any 7K chart
python predict.py "path/to/chart.osu"
```

Example output:
```text
=== O2Jam 7K Level Prediction ===
Title:    Halcyon
Version:  [O2Jam] [D] [142]
Notes:    3899 | Duration: 150.15s (7508 frames @ 20ms)
Latency:  14.23 ms
---------------------------------------------
Predicted O2Jam Level: Lv. 138.4 (Rounded: 138)
---------------------------------------------
```

---

## Scripts & Pipeline Overview

This repository contains the complete pipeline from raw `.osu` parsing to neural network training and ONNX export.

| File / Folder | Purpose |
| :--- | :--- |
| **[`predict.py`](predict.py)** | Standalone command-line tool. Loads `model.onnx` via ONNX Runtime and outputs predicted O2Jam level for any given `.osu` file. |
| **[`step1_inspect.py`](step1_inspect.py)** | Scans your osu! `Songs/` folder (read-only), verifies 7K mania mode (`CircleSize: 7`, `Mode: 3`), extracts O2Jam level tags `[XX]`, and checks for duplicates. |
| **[`step2_build_dataset.py`](step2_build_dataset.py)** | Converts hit objects from valid charts into 20ms discrete time grids `(T, 7)`. Performs MD5 note deduplication (preferring DMJAM over O2EZ) and exports packed arrays into `data/o2jam_dataset.npz` and `data/dataset_metadata.json`. |
| **[`step3_train.py`](step3_train.py)** | PyTorch training script with Automatic Mixed Precision (AMP), Cosine Annealing, Residual 1D-CNN architecture, validation tracking, and ONNX model export. |
| **[`analyze_errors.py`](analyze_errors.py)** | Evaluates the trained checkpoint across level tiers (Lv. 20-50, 50-80, 80-110, 110-140, 140-180), calculates error percentiles, and prints the top outlier charts. |
| **[`extension/o2jam_level_predictor/`](extension/o2jam_level_predictor/)** | The tosu! overlay web extension. Connects to tosu's WebSocket v2, fetches `.osu` text on map switch, and executes `model.onnx` via WebAssembly. |

---

## Technical Architecture

```text
[.osu File] 
     │
     ▼
[Discrete 20ms Grid (T, 7)]  (0=Empty, 1=Tap, 2=LN Head, 3=LN Body, 4=LN Tail)
     │
     ▼
[One-Hot Embedding] (35 channels)
     │
     ▼
[4x Dilated Conv1D Residual Blocks] (Dilation = 1, 2, 4, 8 | Channels = 128)
     │
     ▼
[Multi-Scale Temporal Aggregation] (Global Avg + Global Max + Attention Pooling -> 384 dim)
     │
     ▼
[3-Layer MLP Head with Dropout]
     │
     ▼
[Predicted O2Jam Level (Float)]
```

### 1. Representation
Rather than treating beatmaps as audio waveforms or unstructured text, charts are parsed into a discrete 2D time grid at **20ms resolution (50 FPS)** across the 7 columns:
- `0`: Empty
- `1`: Tap Note
- `2`: Long Note (LN) Head
- `3`: Long Note (LN) Body (Hold)
- `4`: Long Note (LN) Tail (Release)

### 2. Network Backbone
- **Input Projection:** 5 states $\times$ 7 columns are one-hot encoded into 35 channels.
- **Residual Conv1D Blocks:** 4 residual blocks with dilated convolutions ($d \in \{1, 2, 4, 8\}$) capture local patterns, jacks, chordstreams, and roll patterns.
- **Multi-Scale Pooling:** Combines **Global Average Pooling** (overall density), **Global Max Pooling** (peak burst density), and **Attention Pooling** (salient climax sections) into a unified representation.
- **MLP Head:** Fully connected layers `384 -> 128 -> 64 -> 1` with LeakyReLU activations and dropout.

---

## Training & Reproduction

### 1. Requirements
```bash
pip install -r requirements.txt
```
*(For GPU acceleration, ensure a CUDA-compatible build of PyTorch is installed).*

### 2. Build Dataset
Ensure your osu! songs folder contains `[_O2Jam_]*` beatmap folders, then run:
```bash
python step2_build_dataset.py
```
This generates:
- `data/o2jam_dataset.npz` (packed zero-copy array, ~12 MB)
- `data/dataset_metadata.json` (metadata for all 2,557 unique charts)

### 3. Train Model & Export to ONNX
```bash
python step3_train.py
```
Training completes in ~4 minutes on an RTX 4070 SUPER for 30 epochs. It evaluates the validation split at every epoch and automatically exports the best checkpoint to `model.onnx`.

### 4. Analyze Performance
```bash
python analyze_errors.py
```

---

## Accuracy & Benchmarks

Evaluated on an independent 15% validation split (384 charts):

| Metric | Result |
| :--- | :--- |
| **Validation MAE** | **6.68 levels** |
| **Validation RMSE** | **9.87 levels** |
| **Pearson Correlation ($R$)** | **0.964** |
| **Median Error (P50)** | **$\le$ 4.72 levels** |
| **75th Percentile (P75)** | **$\le$ 9.12 levels** |

### Latency Benchmark
- **GPU (PyTorch CUDA FP16):** `1.23 ms`
- **CPU (ONNX Runtime C++):** `4.55 ms` (30s segment) / `20.23 ms` (full 2.5 min song)
- **Browser (ONNX Runtime WASM):** `~12 - 18 ms`

---

## License & Credits

- Built for the **osu!mania** and **O2Jam** rhythm game communities.
- Compatible with [tosu!](https://github.com/tosuapp/tosu) by KotRik.
