# Dataset & Chart Parsing

## 1. Data Sources & Composition

The training dataset integrates **4,423 unique 7-key charts** across four major O2Jam server rating systems, with uncompressed native `.ojn` files taking priority over `.osu` text converts:

| Source | Priority ID | File Format | Role / Server Context | Unique Charts |
| :--- | :---: | :--- | :--- | :---: |
| **Private Server Native** | `0` | `.ojn` native binary | Uncorrupted master charts from community channels (`DSong`, `ESong`, `HSong`, `NSong`) | 2,367 |
| **NSJAM Native** | `1` | `.ojn` native binary | Modern community grading (`NSM01` Easy, `NSM02` Normal, `NSM03` Hard, `NSM04` LN) | 1,269 |
| **osu! DMJAM Converts** | `2` | `.osu` converts | osu!mania 7K DMJAM conversions (`[E]`, `[N]`, `[H]`, `[D]`) | 361 |
| **osu! O2EZ Converts** | `3` | `.osu` converts | Legacy O2EZ conversions (`[O2EZ]`, calibrated via train-only server adapter) | 426 |
| **Total** | — | — | **Deduplicated Unified Dataset** | **4,423** |

Charts with level $\le 3$ or fewer than 20 notes are filtered out during parsing.

### Credits
- DMJAM & O2EZ charts provided by the [O2Jam osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975).
- `.ojn` binary decoding logic referenced from [o2jam-to-osu](https://github.com/LuzianU/o2jam-to-osu) by LuzianU.

---

## 2. Direct Binary `.ojn` Decoder

Rather than converting `.ojn` to `.osu` text files (which introduces floating-point BPM rounding and timing drift), `scripts/step2_build_dataset.py` implements a zero-intermediate **pure Python binary reader**:

### Binary Specification & Offset Isolation:
- **Header (300 Bytes):**
  - Offset `0..3`: Song ID (`int32`)
  - Offset `4..7`: Signature string `"ojn\0"` (`0x006E6A6F`, `uint32`)
  - Offset `16..19`: Initial BPM (`float32`)
  - Offset `20..27`: Level ratings for `[Easy, Normal, Hard, pad]` (`int16[4]`). Code reads the Hard difficulty level at offset `24..25` (`int16`).
  - Offset `108..171`: Song title (64 bytes, CP949 null-terminated string)
  - Offset `172..203`: Artist name (32 bytes, CP949 null-terminated string)
  - Offset `284..287`: Easy note data offset (`int32`)
  - Offset `288..291`: Normal note data offset (`int32`)
  - Offset `292..295`: Hard note data offset (`int32`, code reads here)
  - Offset `296..299`: Cover image offset / package boundary (`int32`, code reads here)
- **Block Header (8 Bytes):**
  - `measure` (`int32`), `channel` (`int16`), `event_count` (`int16`)
- **Channels:**
  - `0`: Fractional measure length multiplier (`float32`)
  - `1`: BPM change event (`float32`)
  - `2..8`: 7-Key note lanes 1 to 7 (`col = channel - 2`)
  - *Other channels:* Ignored by parser.
- **Note Event (4 Bytes for note channels 2..8):**
  - `val` (`int16`), `vol_pan` (`uint8`), `n_type` (`uint8`).
  - Code extracts the note flag as `n_type % 4`:
    - `0`: Normal tap note
    - `2`: Long Note (LN) hold head
    - `3`: Long Note (LN) hold tail

### Hard Difficulty Isolation:
The `.ojn` container encapsulates Easy, Normal, and Hard charts sequentially in one binary file. The parser seeks strictly to `hard_offset = struct.unpack('<i', h[292:296])[0]` and reads up to `cover_off = struct.unpack('<i', h[296:300])[0]`. Files with invalid pointers (`hard_off < 300`, `hard_off >= file_len`, or `cover_off <= hard_off`) are rejected and logged in the per-source skip summary.

Hold notes are paired by matching opening `0x02` events with their succeeding `0x03` event on the same lane, directly outputting start and release timestamps in milliseconds.

---

## 3. Robust Deduplication & Priority Rules

Many charts exist across multiple servers under different rating iterations. To prevent identical or converter-shifted note patterns from entering both training and validation sets:
1. **Multi-Factor Deduplication Check (`are_charts_duplicate`):**
   - **Bitwise match:** Exact MD5 match of the 50 FPS quantized note grid (`c1["grid_hash"] == c2["grid_hash"]`).
   - **Candidate matching:** Must have identical note count (`c1["note_count"] == c2["note_count"]`) and duration within 2.0s ($|\Delta \text{duration}| \le 2000\text{ ms}$).
   - **Title token match:** If title token sets (alphanumeric and CJK/Hangul tokens, excluding common terms like `o2jam`, `hard`, `mix`, etc.) overlap between candidates, they are marked duplicate.
   - **Fallback match:** If note count and duration match, level ratings are identical, and duration is within 0.5s ($|\Delta \text{duration}| \le 500\text{ ms}$), the charts are marked duplicate.
2. **Duplicate collisions are resolved strictly by source authority:**
   $$\text{Private Server Native (0)} > \text{NSJAM Native (1)} > \text{osu! DMJAM (2)} > \text{osu! O2EZ (3)}$$
   Uncorrupted native `.ojn` files always take precedence over text-converted `.osu` files.

---

## 4. Packing Format (`data/o2jam_dataset.npz`)

For training efficiency, all charts are serialized into a compressed NumPy archive:
- `flat_grids`: `uint8` 1D array storing all concatenated `(T, 7)` grids. Total size: 257,789,868 elements (~245.85 MB uncompressed in memory).
- `offsets`: `int64` array of size $N + 1$ marking chart boundaries in `flat_grids`.
- `lengths`: `int32` array of frame counts $T$ for each chart.
- `levels`: `float32` target difficulty levels.
- `durations`: `int32` song durations in milliseconds.
- `priorities`: `int32` array of source priority IDs (`0` = Private, `1` = NSJAM, `2` = DMJAM, `3` = O2EZ).

The archive is saved via `np.savez_compressed` (occupying ~20.38 MB on disk) and loaded into memory using `np.load` (fully decompressed into RAM rather than memory-mapped).

---

## 5. Expanding the Dataset with Old Servers (Multi-Server Calibration)

Legacy O2Jam servers often exhibit non-linear rating offsets relative to modern `Private` server standards. For example, across 293 duplicate chart pairs, `O2EZ` matches modern ratings in `Lv. 75–95` ($\text{bias} = -0.14$), but underrates `Lv. 1–50` by $-5.09$ levels, `Lv. 50–75` by $-1.51$ levels, and `Lv. 95–150` by $-3.08$ levels.

Simply dropping `O2EZ` removes ~10% of pattern diversity and hurts generalization, while training on raw unadjusted `O2EZ` labels pulls predictions down. We evaluated three strategies on the exact same `SEED=42` split:

| Strategy | Canonical Val MAE ($n=613$, Private+NSJAM+DMJAM) | Canonical Val RMSE | Full Val MAE ($n=664$) | Full Val RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **1. Raw Unadjusted Labels (`baseline_raw`)** | 5.683 | 8.801 | 5.707 | 8.778 |
| **2. Static Duplicate-Pair Label Shift (`approach2_empirical`)** | 5.481 | 8.253 | 5.514 | 8.258 |
| **3. Train-Only `ServerCalibrationAdapter` (`approach1_adapter`)** | **5.186** | **7.697** | **5.239** | **7.669** |

### How to Add New Old Servers in the Future:
1. **Assign a Priority ID (`4, 5, ...`):** In [`scripts/step2_build_dataset.py`](../scripts/step2_build_dataset.py), assign the new server a `PRIORITY_*` integer $> 0$ (below `PRIORITY_PRIVATE = 0` so canonical `Private` charts always win deduplication) and include its `priority` in `data/o2jam_dataset.npz`.
2. **Check Duplicate Offsets:** Run `python scripts/step2_build_dataset.py` and inspect the printed `Cross-Source Duplicate Rating Offsets` table.
3. **Train with `ServerCalibrationAdapter`:** In [`scripts/step3_train.py`](../scripts/step3_train.py), `ServerCalibrationAdapter(num_sources=...)` learns a 4-knot piecewise-linear offset curve (`knots = [35.0, 62.5, 85.0, 115.0]`) for every non-Private server (`priority > 0`) during training, while pinning `priority == 0` (`Private`) strictly to `0.0` offset.
4. **Zero-Overhead ONNX Export:** At validation and ONNX export time, `ServerCalibrationAdapter` is discarded so `model.onnx` always outputs strictly on the canonical `Private` rating scale without needing a server ID input in the browser.
