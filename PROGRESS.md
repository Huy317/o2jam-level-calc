# O2Jam / osu!mania 7K Level Predictor Progress

## Project Overview
Predict human-assigned O2Jam difficulty "Level" (ranging ~1 to 180+) of 7-key osu!mania beatmaps directly from raw note sequences using a lightweight 1D-CNN + Attention/Multi-Scale Pooling architecture, exported to ONNX for sub-10ms real-time inference.

---

## Progress Log

### Step 1: Dataset Inspection & Verification (`step1_inspect.py`)
- **Status:** Completed
- **Source Directory:** `D:\osu!\Songs` (Strictly Readonly)
- **Results:**
  - Total `[_O2Jam_]*` directories found: **2,739**
  - Directories without `.osu` files: **1** (`[_O2Jam_] HatsuneMIKU』 - The Snow White Princess Is` is an empty directory)
  - Total `.osu` chart files found: **2,738**
  - 7K Verification (`CircleSize: 7` and `Mode: 3`): **2,738 / 2,738 (100% Mania 7K)**
  - Regex match on `Version:` line (`\[(\d+)\]\s*$`): **2,738 / 2,738 (100% matched)**
  - Parse failures / unparseable files: **0**
  - **Level Distribution:**
    - Absolute Min: `1`, Absolute Max: `180`
    - Level `<= 1` (gimmicks/unrated to be excluded per spec): **92 charts**
    - Valid Levels (`> 1`): **2,646 charts**
    - Valid Level Range: **20 to 180** (Average: **85.15**)
  - **Server / Difficulty Tag Breakdown:**
    - `[O2Jam] [O2EZ]`: 696 charts (Modern O2EZ community server re-rates)
    - `[O2Jam] [H]`: 671 charts (Hard difficulty)
    - `[O2Jam] [E]`: 577 charts (Easy difficulty)
    - `[O2Jam] [N]`: 429 charts (Normal difficulty)
    - `[O2Jam] [D]`: 365 charts (Super Hard difficulty, [D] = SH)
  - **HitObjects & Deduplication Details:**
    - All charts contain >= 20 notes (range: 315 to 12,243 notes, average: ~3,384 notes).
    - Exactly **89 duplicate groups** detected by MD5 note hash (2,557 unique charts total).
    - In all 89 groups, duplicates are strictly between an **`[O2EZ]`** chart and a **DMJAM** chart (`[E]`, `[N]`, `[H]`, or `[D]`). Zero duplicates exist within the same family.
    - **63 pairs** share the exact same level.
    - **26 pairs** have differing level ratings between O2EZ and DMJAM. Per user directive, DMJAM levels are prioritized over O2EZ.

### Hardware & Acceleration Verification
- **GPU:** NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
- **PyTorch:** `2.6.0+cu124` installed and verified (`torch.cuda.is_available() == True`).

### Step 2: Dataset Generation & Packing (`step2_build_dataset.py`)
- **Status:** Completed
- **Source Access:** Strictly READONLY on `D:\osu!\Songs`.
- **Dataset Generation Summary:**
  - Total candidate charts parsed: **2,646**
  - Deduplicated charts retained: **2,557 unique charts** (**89 duplicates discarded**, DMJAM prioritized).
  - Time Resolution: **$\text{STEP\_MS} = 20\text{ms}$** (50 FPS).
  - Encoding: `0` (Empty), `1` (Tap Note), `2` (LN Head), `3` (LN Body), `4` (LN Tail).
  - Total frames across all charts: **20,995,209** (140.16 MB uncompressed).
  - Packed Storage Format: Single zero-copy flat uint8 array + index offsets in compressed NumPy archive (`data/o2jam_dataset.npz`, **12.01 MB**).
  - Chart Metadata: Saved in structured JSON (`data/dataset_metadata.json`).
  - Pipeline Execution Time: **35.71s**
- **Integrity Self-Check:**
  - All 2,557 charts successfully loaded and verified.
  - Level Statistics: Min: `20`, Max: `180`, Mean: `85.04`, Std: `33.63`.
  - Frame Length Statistics: Min: `1,170` (23.4s), Median: `7,317` (146.3s), Max: `24,060` (481.2s).
  - Sample reconstruction checked across 5 checkpoints: all discrete codes `[0, 1, 2, 3, 4]` present and valid.

### Step 3: Model Architecture, Training & ONNX Export (`step3_train.py`)
- **Status:** Completed
- **Model Architecture (`O2JamLevelPredictor`):**
  - **Input:** `(Batch, T, 7)` uint8 integer grid [0..4] $\to$ One-hot encoding $\to$ `(Batch, 35, T)`.
  - **Backbone:** 4 Dilated Residual Conv1d Blocks (kernel size 5, dilations $1, 2, 4, 8$) with BatchNorm1d and GELU (covering receptive fields up to ~1.6s patterns).
  - **Multi-Scale Aggregation:** Global Average Pooling (overall stamina/density) + Global Max Pooling (peak burst) + Attention Pooling (hardest difficulty spikes) concatenated to $(B, 3 \times C)$.
  - **Head:** 3-layer MLP with Dropout(0.2) mapping pooled features to scalar predicted level.
  - **Total Parameters:** **719,874** (~2.88 MB).
- **Training Setup:**
  - Train (85% = 2,173 charts) / Validation (15% = 384 charts) split.
  - Data Augmentation: Horizontal column mirror (`col -> 6 - col`) with 50% probability during training.
  - Dynamic Batching: Padded each batch to batch-max sequence length (zero truncation, no wasted memory).
  - Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`) with Cosine Annealing scheduler.
  - Loss Function: `SmoothL1Loss(beta=1.0)` (Huber Loss).
  - Mixed Precision: PyTorch AMP (`torch.amp.autocast('cuda')`).
  - Hardware: NVIDIA GeForce RTX 4070 SUPER.
  - Total Training Time: **261.9s (~4.3 minutes)** for 30 epochs.
- **Validation Results (Best Checkpoint Epoch 26):**
  - **Validation MAE:** **6.68 levels** (across full range 20 to 180)
  - **Validation RMSE:** **9.87**
  - **Pearson Correlation ($R$):** **0.964** (extremely high linear correlation with human rankings)
  - **Sample Predictions:**
    - `[E]` Lv.22 $\to$ **Lv.24.6** (Err: +2.6)
    - `[E]` Lv.67 $\to$ **Lv.67.6** (Err: +0.6)
    - `[N]` Lv.73 $\to$ **Lv.71.6** (Err: -1.4)
    - `[N]` Lv.78 $\to$ **Lv.79.8** (Err: +1.8)
    - `[H]` Lv.111 $\to$ **Lv.114.1** (Err: +3.1)
    - `[D]` Lv.153 (ADAM) $\to$ **Lv.153.5** (Err: +0.5)
- **ONNX Export & Latency Verification:**
  - Exported to [`model.onnx`](file:///d:/Projects/o2jam-level-calc/model.onnx) (**2.76 MB**) with dynamic axes for variable sequence length $T$ and batch size.
  - Mathematical parity with PyTorch verified on sample charts: max absolute difference $< 1.5 \times 10^{-5}$.
  - **Inference Latencies (Full 2.5-minute song, 7,500 frames):**
    - **GPU (RTX 4070 SUPER):** **1.23 ms** (8x faster than the <10ms requirement)
    - **CPU (ONNX Runtime):** **4.55 ms** (30s window), **7.89 ms** (60s window), **20.23 ms** (full 2.5-minute song)
