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
    - `[O2Jam] [D]`: 365 charts (Draft/Dan difficulty)
  - **HitObjects & Deduplication Preview:**
    - All charts have >= 20 notes (range: 315 to 12,243 notes, average: ~3,384 notes).
    - Exactly 89 duplicate charts identified by MD5 note hash, leaving **2,557 unique charts** for the final dataset. Modern `[O2EZ]` tags take precedence in deduplication.

---

### Next Steps
- [ ] **Step 2: Build Dataset (`step2_build_dataset.py`)**
  - Convert charts to `(T, 7)` uint8 grids at `STEP_MS = 20` (50 FPS).
  - Categorical encoding: `0` (Empty), `1` (Tap), `2` (LN Head), `3` (LN Body), `4` (LN Tail).
  - Deduplicate with priority for modern `[O2EZ]` tags.
  - Save dataset in an efficient format (e.g. `.npz` or memory-mapped files) with metadata index.
- [ ] **Step 3: Model Architecture & Training (`step3_train.py`)**
  - 1D-CNN with Conv1d residual blocks, dilations (1, 2, 4), and Attention / Top-K Pooling.
  - Data augmentation (horizontal mirror 50%).
  - Train/validation split (85%/15%), Huber loss (`SmoothL1Loss`), evaluation metrics (MAE, RMSE, R²).
  - ONNX export and verification (<10ms inference check).
