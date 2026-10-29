# O2Jam / osu!mania 7K Level Predictor Progress

## Project Overview
Predict human-assigned O2Jam difficulty "Level" (ranging ~1 to 180+) of 7-key osu!mania beatmaps directly from raw note sequences using a lightweight 1D-CNN + Attention/Top-K Pooling architecture, exported to ONNX for sub-10ms real-time inference.

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
  - Packed Storage Format: Single zero-copy flat uint8 array + index offsets in compressed NumPy format (`data/o2jam_dataset.npz`, **12.01 MB**).
  - Chart Metadata: Saved in structured JSON (`data/dataset_metadata.json`).
  - Pipeline Execution Time: **35.71s**
- **Integrity Self-Check:**
  - All 2,557 charts successfully loaded and verified.
  - Level Statistics: Min: `20`, Max: `180`, Mean: `85.04`, Std: `33.63`.
  - Frame Length Statistics: Min: `1,170` (23.4s), Median: `7,317` (146.3s), Max: `24,060` (481.2s).
  - Sample reconstruction checked across 5 checkpoints: all discrete codes `[0, 1, 2, 3, 4]` present and valid.

---

### Next Steps
- [ ] **Step 3: Model Architecture & Training (`step3_train.py`)**
  - Train/validation split (85% / 15%, stratified or random with fixed seed).
  - Dynamic batching PyTorch DataLoader (padding each batch to batch-max length).
  - Horizontal column mirror data augmentation (`col -> 6 - col` with 50% probability).
  - 1D-CNN with Conv1d residual blocks, dilations (1, 2, 4), and Attention / Top-K Pooling.
  - Mixed precision training (`torch.amp.autocast`) with `SmoothL1Loss` (Huber Loss) on RTX 4070 SUPER.
  - Evaluation metrics: MAE, RMSE, Pearson $R$.
  - Export best model to `model.onnx` and verify <10ms latency benchmark.
