# Dataset & Chart Parsing

## 1. Data Sources & Composition

The training dataset integrates **4,055 unique 7-key charts** across three major O2Jam server rating systems:

| Source | File Format | Role / Server Context | Unique Charts |
| :--- | :--- | :--- | :--- |
| **DMJAM** | `.osu` converts | Official / classic competitive server scales (`[E]`, `[N]`, `[H]`, `[D]`) | 1,861 |
| **NSJAM** | `.ojn` native binary | Modern community grading (`NSM01` Easy, `NSM02` Normal, `NSM03` Hard, `NSM04` LN) | 1,498 |
| **O2EZ** | `.osu` converts | O2EZ community modern re-ratings (`[O2EZ]`) | 696 |
| **Total** | — | **Deduplicated Unified Dataset** | **4,055** |

### Acknowledgments & Source Attribution
The DMJAM and O2EZ chart collections were sourced from the [O2Jam to osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975) on the osu! community forums. We thank the organizers and mappers who preserved and standardized these converted charts with their original server difficulty tags intact.

---

## 2. Direct Binary `.ojn` Decoder

Rather than converting `.ojn` to `.osu` text files (which introduces floating-point BPM rounding and timing drift), `scripts/step2_build_dataset.py` implements a zero-intermediate **pure Python binary reader**:

### Binary Specification:
- **Header (300 Bytes):**
  - Offset `0..3`: File signature / Song ID (`int32`)
  - Offset `4..7`: Signature string (`o2ma`)
  - Offset `8..27`: Song title string
  - Offset `28..47`: Artist string
  - Offset `88..107`: Level ratings for `[Easy, Normal, Hard]` (`int16` array). We target the competitive `Hard (HX)` difficulty.
  - Offset `112..123`: Note counts `[Easy, Normal, Hard]` (`int32`)
  - Offset `124..135`: Measure counts `[Easy, Normal, Hard]` (`int32`)
  - Offset `136..147`: Package offsets `[Easy, Normal, Hard]` (`int32`)
- **Block Header (8 Bytes):**
  - `measure` (`int32`), `channel` (`int16`), `event_count` (`int16`)
- **Channels:**
  - `0`: Fractional measure length
  - `1`: Fixed BPM change
  - `2`: Extended fractional BPM change
  - `3..9`: 7-Key Note Lanes (1 to 7)
- **Note Event (4 Bytes):**
  - Note value, volume, pan, and note-type flag (`0x00` = normal tap, `0x02` = LN hold head, `0x03` = LN hold tail).

Hold notes are paired by matching opening `0x02` events with their succeeding `0x03` event on the same lane, directly outputting start and release timestamps in milliseconds.

---

## 3. Deduplication & Priority Rules

Many charts exist across multiple servers under different rating iterations. To prevent identical note patterns from entering both training and validation sets:
1. Every chart is hashed using its exact quantized note sequence:
   $$\text{Hash} = \text{MD5}(\text{grid}[0 \dots T \times 7])$$
2. Duplicate collisions are resolved strictly by server authority:
   $$\text{DMJAM (Priority 0)} > \text{NSJAM (Priority 1)} > \text{O2EZ (Priority 2)}$$

### Cross-Server Level Consistency Audit:
- **DMJAM vs NSJAM Overlaps:** 81 chart overlaps identified. **70.4% share the exact identical integer level**, and **79.0% are within $\pm 2$ levels**. This statistically confirmed that NSJAM grades on the identical scale as classic DMJAM.
- **O2EZ Overlaps:** Prioritizing NSJAM over O2EZ replaced 119 legacy community re-rates with verified NSJAM ratings.

---

## 4. Packing Format (`data/o2jam_dataset.npz`)

For maximum GPU loading throughput, charts are packed into a single contiguous 1D array with index offsets:
- `flat_grids`: `uint8` 1D array storing all concatenated `(T, 7)` grids.
- `offsets`: `int64` array of size $N + 1$ marking chart boundaries in `flat_grids`.
- `lengths`: `int32` array of frame counts $T$ for each chart.
- `levels`: `float32` target difficulty levels.
- `durations`: `int32` song durations in milliseconds.

Total uncompressed size: **~18.9 MB**, allowing instantaneous zero-copy memory-mapped loading in PyTorch.
