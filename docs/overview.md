# Project Overview: O2Jam 7K Level Calculator

## 1. What is This Project?
**O2Jam 7K Level Calculator** is an end-to-end machine learning system that predicts the human-assigned **O2Jam difficulty level** (ranging from **Lv. 4 to Lv. 180**) for any 7-key osu!mania or O2Jam beatmap directly from note timings and hold durations.

In addition to the training pipeline, it includes an **instant real-time HUD overlay extension** for [Tosu](https://github.com/KotRikD/tosu) that evaluates charts live inside osu! using ONNX Runtime Web with native settings support.

---

## 2. High-Level Architecture & Workflow

```mermaid
flowchart LR
    A[osu! Converts .osu<br/>+ NSJAM .ojn<br/>+ Private .ojn] --> B[Binary Parsing<br/>& Deduplication]
    B --> C[Discrete (T, 7)<br/>20ms Grid (50 FPS)]
    C --> D[60 Kinematic Channels<br/>+ Dilated ResNet Backbone]
    D --> E[Sustain-Damped Multi-Scale<br/>+ Peak-Relative Active Gating]
    E --> F[Double-Softplus Linear Core<br/>Level Output + Strain Curve]
    F --> G[Export model.onnx<br/>(4.60 MB Dual Output)]
    G --> H[Tosu Static Overlay<br/>(~8-18ms Live WASM)]
```

### Pipeline Steps:
1. **Inspection (`scripts/step1_inspect.py`):** Scans chart directories, validates 7K keys, parses server metadata, and checks level tag formatting.
2. **Dataset Building (`scripts/step2_build_dataset.py`):** Parses `.osu` text files and binary `.ojn` files (isolating Hard package offsets) into discrete `(T, 7)` uint8 grids binned at 20ms (50 FPS), deduplicating charts across servers.
3. **Training & Export (`scripts/step3_train.py`):** Trains a 1D Dilated Residual CNN (~10.1s / 505 frames receptive field) with 60 kinematic channels, sustain-damped 2s burst pooling, peak-relative active stamina gating, and a linear-core Double-Softplus regression head (`[1.0, 195.0]`) on CUDA, exporting `model.onnx`.
4. **Real-time Extension (`extension/o2jam_level_predictor/`):** Self-contained web overlay running ONNX Runtime Web via SIMD-threaded WebAssembly in Tosu. Features native `settings.json` configuration, minimal level badge display with optional song title and rate badge, and automatic DoubleTime (1.5x DT) / HalfTime (0.75x HT) speed scaling.

---

## 3. Current Project State

| Metric / Dimension | Specification / Value |
| :--- | :--- |
| **Total Unique Charts** | **4,423 charts** (deduplicated across DMJAM, NSJAM, O2EZ, and Private Servers) |
| **Level Coverage** | **Lv. 4 to Lv. 180** (mean Lv. 81.8) |
| **Model Size** | **4.60 MB** (`model.onnx`, Dual Output: `predicted_level`, `strain_curve`) |
| **Held-Out Validation MAE (Primary)** | **5.52 levels** ($n=664$, 15% split) |
| **Held-Out Validation RMSE** | **8.14 levels** |
| **Held-Out Pearson Correlation ($R$)** | **0.9776** |
| **Held-Out Median Error (P50)** | **$\le 3.80$ levels** |
| **Held-Out 90th Percentile (P90)** | **$\le 12.74$ levels** |
| **Full Dataset MAE (Secondary, 85% Train)** | **4.69 levels** ($n=4,423$; RMSE 6.69, $R=0.9840$) |
| **Full Dataset P50 / P90** | **$\le 3.33$ levels** / **$\le 10.38$ levels** |
| **Inference Latency** | **~8 - 18 ms** (WebAssembly client-side typical) |

---

## 4. Documentation Index

- [Architecture & Neural Network Design](architecture.md): In-depth breakdown of the 60 kinematic channels, dilated residual backbone, receptive field, sustain-damped burst pooling, active gating, and linear-core Double-Softplus head.
- [Dataset & Parsing](dataset.md): Native binary `.ojn` decoder, Hard difficulty seek pointers, `.osu` parsing, deduplication priority, and quantization format.
- [Benchmarks & Error Analysis](benchmarks.md): Comprehensive validation metrics, percentiles, tier-by-tier MAE, and outlier case studies.
- [Tosu Extension & Real-time Integration](extension.md): Live overlay installation, offline WASM bundle, HT/DT physical rate scaling, and native settings.
