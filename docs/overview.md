# Project Overview: O2Jam 7K Level Calculator

## 1. What is This Project?
**O2Jam 7K Level Calculator** is an end-to-end machine learning system that predicts the human-assigned **O2Jam difficulty level** (ranging from **Lv. 15 to Lv. 185+**) for any 7-key osu!mania or O2Jam beatmap directly from note timings and hold durations.

In addition to the training pipeline, it includes an **instant real-time HUD overlay extension** for [Tosu](https://github.com/KotRikD/tosu) that evaluates charts live inside osu! using ONNX Runtime Web.

---

## 2. High-Level Architecture & Workflow

```mermaid
flowchart LR
    A[osu! Songs .osu<br/>+ NSJAM .ojn] --> B[Binary Parsing<br/>& Deduplication]
    B --> C[Discrete (T, 7)<br/>20ms Grid (50 FPS)]
    C --> D[Dilated Conv1D<br/>+ Multi-Scale Pooling]
    D --> E[Bounded Output<br/>Level 15 - 185]
    E --> F[Export model.onnx<br/>(4.55 MB)]
    F --> G[Tosu Static Overlay<br/>(Sub-15ms Live WASM)]
```

### Pipeline Steps:
1. **Inspection (`scripts/step1_inspect.py`):** Scans chart directories, validates 7K keys, parses server metadata, and checks level tag formatting.
2. **Dataset Building (`scripts/step2_build_dataset.py`):** Parses `.osu` text files and binary `.ojn` files into discrete `(T, 7)` uint8 grids binned at 20ms (50 FPS), deduplicating charts across servers.
3. **Training & Export (`scripts/step3_train.py`):** Trains a 1D Dilated Residual CNN with Multi-Scale Pooling using PyTorch on CUDA, optimizing Tier-Weighted Huber Loss, and exports `model.onnx`.
4. **Real-time Extension (`extension/o2jam_level_predictor/`):** Self-contained web overlay running ONNX Runtime Web via SIMD-threaded WebAssembly in Tosu. Automatically detects DoubleTime (1.5x DT) and HalfTime (0.75x HT) speed changes in real-time.

---

## 3. Current Project State

| Metric / Dimension | Specification / Value |
| :--- | :--- |
| **Total Unique Charts** | **4,055 charts** (deduplicated across DMJAM, NSJAM, O2EZ) |
| **Level Coverage** | **Lv. 20 to Lv. 181** (avg Lv. 87.2) |
| **Model Size** | **4.55 MB** (`model.onnx`) |
| **Validation MAE** | **5.75 levels** |
| **Validation RMSE** | **8.59 levels** |
| **Pearson Correlation ($R$)** | **0.967** |
| **Median Absolute Error (P50)**| **$\le 3.88$ levels** |
| **P75 Absolute Error** | **$\le 7.44$ levels** |
| **Core Tier (Lv. 50-80) MAE** | **4.06 levels** (229 validation charts) |
| **Inference Latency** | **~8 - 18 ms** (WebAssembly client-side) |

---

## 4. Documentation Index

- [Architecture & Neural Network Design](architecture.md): In-depth breakdown of the 37-channel input, 6-block dilated residual backbone, receptive field, and temporal pooling.
- [Dataset & Parsing](dataset.md): Native binary `.ojn` decoder, `.osu` parsing, deduplication priority, and quantization format.
- [Benchmarks & Error Analysis](benchmarks.md): Comprehensive validation metrics, percentiles, tier-by-tier MAE, and outlier case studies.
- [Tosu Extension & Real-time Integration](extension.md): Live overlay installation, offline WASM bundle, HT/DT physical rate scaling, and styling.
