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
    - `[O2Jam] [O2EZ]`: 696 charts (Modern O2Jam community server re-rates)
    - `[O2Jam] [H]`: 671 charts (Hard difficulty)
    - `[O2Jam] [E]`: 577 charts (Easy difficulty)
    - `[O2Jam] [N]`: 429 charts (Normal difficulty)
    - `[O2Jam] [D]`: 365 charts (Super Hard difficulty, [D] = SH)
  - **HitObjects & Deduplication Details:**
    - All charts contain >= 20 notes (range: 315 to 12,243 notes, average: ~3,384 notes).
    - Exactly **89 duplicate groups** detected by MD5 note hash (2,557 unique charts total).
    - In all 89 groups, duplicates are strictly between an **`[O2EZ]`** chart and a **DMJAM** chart (`[E]`, `[N]`, `[H]`, or `[D]`). Zero duplicates exist within the same family.
    - **63 pairs** share the exact same level.
    - **26 pairs** have differing level ratings between O2EZ and DMJAM (e.g. Mopemope 137 vs 154, Red Goose 135 vs 140, Viva Happy 54 vs 47). Awaiting user verdict on prioritization.

---

### Next Steps
- [ ] **Step 2: Build Dataset (`step2_build_dataset.py`)**
  - Finalize duplicate resolution based on user priority feedback (O2EZ vs DMJAM).
  - Convert charts to `(T, 7)` uint8 grids at `STEP_MS = 20` (50 FPS).
  - Categorical encoding: `0` (Empty), `1` (Tap), `2` (LN Head), `3` (LN Body), `4` (LN Tail).
  - Save dataset in fast-loading compressed format (`o2jam_dataset.npz` with metadata index).
- [ ] **Step 3: Model Architecture & Training (`step3_train.py`)**
  - 1D-CNN with Conv1d residual blocks, dilations (1, 2, 4), and Attention / Top-K Pooling.
  - Data augmentation (horizontal mirror 50%).
  - Train/validation split (85%/15%), Huber loss (`SmoothL1Loss`), evaluation metrics (MAE, RMSE, R²).
  - ONNX export and verification (<10ms inference check).
