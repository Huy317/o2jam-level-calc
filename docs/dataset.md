# Dataset & Chart Parsing

## 1. Data Sources & Composition

The training dataset integrates **4,423 unique 7-key charts** across four major O2Jam server rating systems, with uncompressed native `.ojn` files taking highest priority:

| Source | File Format | Role / Server Context | Unique Charts |
| :--- | :--- | :--- | :---: |
| **Private Server Native** | `.ojn` native binary | Uncorrupted master charts from community channels (`DSong`, `ESong`, `HSong`, `NSong`) | 2,367 |
| **NSJAM Native** | `.ojn` native binary | Modern community grading (`NSM01` Easy, `NSM02` Normal, `NSM03` Hard, `NSM04` LN) | 1,225 |
| **osu! Converts** | `.osu` converts | osu!mania 7K conversions (405 DMJAM `[E],[N],[H],[D]`, 426 O2EZ `[O2EZ]`) | 831 |
| **Total** | — | **Deduplicated Unified Dataset** | **4,423** |

Charts with level $\le 3$ or fewer than 20 notes are filtered out during parsing. The osu! converts tag split (405 DMJAM, 426 O2EZ) is tracked via the dataset priority metadata.

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
The `.ojn` container encapsulates Easy, Normal, and Hard charts sequentially in one binary file. The parser seeks strictly to `hard_offset = struct.unpack('<i', h[292:296])[0]` and reads up to `cover_off = struct.unpack('<i', h[296:300])[0]`.

> [!WARNING]
> **Known Issue:** If the Hard seek pointer is invalid (`hard_off < 300` or `hard_off >= file_len`), the parser currently falls back to `hard_off = 300` (the beginning of the file's note data). This causes Easy, Normal, and Hard notes to be merged together into a single corrupted chart. This fallback is a known bug to be addressed by rejecting invalid files.

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
   $$\text{Private Server Native (0)} > \text{osu! DMJAM (1)} > \text{NSJAM Native (2)} > \text{osu! O2EZ (3)}$$
   Uncorrupted native `.ojn` files always take precedence over text-converted `.osu` files.

---

## 4. Packing Format (`data/o2jam_dataset.npz`)

For training efficiency, all charts are serialized into a compressed NumPy archive:
- `flat_grids`: `uint8` 1D array storing all concatenated `(T, 7)` grids. Total size: 257,789,875 elements (~245.8 MB uncompressed in memory).
- `offsets`: `int64` array of size $N + 1$ marking chart boundaries in `flat_grids`.
- `lengths`: `int32` array of frame counts $T$ for each chart.
- `levels`: `float32` target difficulty levels.
- `durations`: `int32` song durations in milliseconds.

The archive is saved via `np.savez_compressed` (occupying ~20.4 MB on disk) and loaded into memory using `np.load` (fully decompressed into RAM rather than memory-mapped).
