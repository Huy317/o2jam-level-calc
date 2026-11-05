# Dataset & Chart Parsing

## 1. Data Sources & Composition

The training dataset integrates **4,438 unique 7-key charts** across four major O2Jam server rating systems, with uncompressed native `.ojn` files taking highest priority:

| Source | File Format | Role / Server Context | Unique Charts |
| :--- | :--- | :--- | :---: |
| **Private Server Native** | `.ojn` native binary | Uncorrupted master charts from community channels (`DSong`, `ESong`, `HSong`, `NSong`) | 2,381 |
| **NSJAM Native** | `.ojn` native binary | Modern community grading (`NSM01` Easy, `NSM02` Normal, `NSM03` Hard, `NSM04` LN) | 1,227 |
| **osu! O2EZ** | `.osu` converts | O2EZ community modern re-ratings (`[O2EZ]`) | 425 |
| **osu! DMJAM** | `.osu` converts | Legacy converts of DMJAM charts (`[E]`, `[N]`, `[H]`, `[D]`) | 405 |
| **Total** | — | **Deduplicated Unified Dataset** | **4,438** |

### Credits
- DMJAM & O2EZ charts provided by the [O2Jam osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975).
- `.ojn` binary decoding logic referenced from [o2jam-to-osu](https://github.com/LuzianU/o2jam-to-osu) by LuzianU.

---

## 2. Direct Binary `.ojn` Decoder

Rather than converting `.ojn` to `.osu` text files (which introduces floating-point BPM rounding and timing drift), `scripts/step2_build_dataset.py` implements a zero-intermediate **pure Python binary reader**:

### Binary Specification & Offset Isolation:
- **Header (300 Bytes):**
  - Offset `0..3`: File signature / Song ID (`int32`)
  - Offset `4..7`: Signature string (`o2ma`)
  - Offset `8..27`: Song title string
  - Offset `28..47`: Artist string
  - Offset `88..107`: Level ratings for `[Easy, Normal, Hard]` (`int16` array). We target the competitive `Hard (HX)` difficulty.
  - Offset `112..123`: Note counts `[Easy, Normal, Hard]` (`int32`)
  - Offset `124..135`: Measure counts `[Easy, Normal, Hard]` (`int32`)
  - Offset `136..147`: Package offsets `[Easy, Normal, Hard]` (`int32`)
  - Offset `292..295`: Package Hard difficulty seek pointer (`int32`)
  - Offset `296..299`: Cover image offset / package boundary (`int32`)
- **Block Header (8 Bytes):**
  - `measure` (`int32`), `channel` (`int16`), `event_count` (`int16`)
- **Channels:**
  - `0`: Fractional measure length
  - `1`: Fixed BPM change
  - `2`: Extended fractional BPM change
  - `3..9`: 7-Key Note Lanes (1 to 7)
- **Note Event (4 Bytes):**
  - Note value, volume, pan, and note-type flag (`0x00` = normal tap, `0x02` = LN hold head, `0x03` = LN hold tail).

### Hard Difficulty Isolation:
The `.ojn` container encapsulates Easy, Normal, and Hard charts sequentially in one binary file. The parser seeks strictly to `hard_offset = struct.unpack('<i', h[292:296])[0]` and reads up to `cover_off = struct.unpack('<i', h[296:300])[0]`. This ensures Hard charts are cleanly isolated without inadvertently merging preceding Easy/Normal notes into the event stream.

Hold notes are paired by matching opening `0x02` events with their succeeding `0x03` event on the same lane, directly outputting start and release timestamps in milliseconds.

---

## 3. Robust Deduplication & Priority Rules

Many charts exist across multiple servers under different rating iterations. To prevent identical or converter-shifted note patterns from entering both training and validation sets:
1. **Multi-Factor Deduplication Check:**
   - **Bitwise match:** $\text{Hash}_1 == \text{Hash}_2$ (exact MD5 match of the 50 FPS quantized note grid).
   - **Converter-shift tolerance:** Identical note count ($\text{Notes}_1 == \text{Notes}_2$) + duration tolerance ($|\Delta t| \le 2.0\text{s}$) + title token overlap or matching level rating. This catches converter artifacts where 1 or 2 notes were quantized by 1 frame (20ms) across formats.
2. **Duplicate collisions are resolved strictly by source authority:**
   $$\text{Private Server Native (0)} > \text{osu! DMJAM (1)} > \text{NSJAM Native (2)} > \text{osu! O2EZ (3)}$$
   Uncorrupted native `.ojn` files always take precedence over text-converted `.osu` files.

---

## 4. Packing Format (`data/o2jam_dataset.npz`)

For maximum GPU loading throughput, charts are packed into a single contiguous 1D array with index offsets:
- `flat_grids`: `uint8` 1D array storing all concatenated `(T, 7)` grids.
- `offsets`: `int64` array of size $N + 1$ marking chart boundaries in `flat_grids`.
- `lengths`: `int32` array of frame counts $T$ for each chart.
- `levels`: `float32` target difficulty levels.
- `durations`: `int32` song durations in milliseconds.

Total uncompressed size: **~20.4 MB**, allowing instantaneous zero-copy memory-mapped loading in PyTorch.
