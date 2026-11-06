# O2Jam 7K Level Predictor

Predicts O2Jam difficulty levels (Lv. 4 to Lv. 180) for 7-key osu!mania charts using a neural network trained on 4,423 O2Jam and community charts.

Includes a lightweight in-game overlay for [tosu](https://github.com/KotRikD/tosu) that displays the estimated level in real time while playing or browsing songs.

---

## tosu In-Game Overlay

No Python or machine learning setup required.

### Setup

1. Download `o2jam_level_predictor.zip` from [GitHub Releases](https://github.com/Huy317/o2jam-level-calc/releases) (or grab the `extension/o2jam_level_predictor` folder directly from this repo).
2. Extract the folder into your tosu `static/` directory:
   ```
   tosu/
   └── static/
       └── o2jam_level_predictor/
           ├── index.html
           ├── index.js
           ├── style.css
           ├── model.onnx
           ├── settings.json
           └── ...
   ```
3. Start tosu and osu!.
4. Open `http://127.0.0.1:24050/o2jam_level_predictor/` in your browser, or add it as a browser source in OBS / tosu overlay manager.

### Features

- **Runs 100% locally:** Inferences run inside the browser using WebAssembly ONNX Runtime (`model.onnx`). No external server or API calls.
- **Speed & mod aware:** Automatically adjusts predictions for Double Time (1.5x DT), Half Time (0.75x HT), or custom osu! rate multipliers.
- **Low overhead:** Only recalculates when the beatmap or mod changes, keeping CPU usage minimal during gameplay.
- **Configurable:** Toggle decimal precision, custom accent colors, scale, opacity, and song title from the tosu settings dashboard.

---

## CLI Usage

You can also test local `.osu` charts from the command line:

```bash
pip install -r requirements.txt
python scripts/predict.py "path/to/chart.osu"
```

Example output:

```text
=========================================
  Song Title : xi - FREEDOM DiVE
  Difficulty : [FOUR DIMENSIONS]
  Key Count  : 7K (Mode 3)
  Notes      : 4,218 notes
  Duration   : 254.2s (12,710 frames)
-----------------------------------------
  >>> Predicted O2Jam Level : Lv. 138.4 (Round: Lv. 138) <<<
  Inference Latency         : 8.50 ms
=========================================
```

---

## How It Works

Standard rhythm game star rating algorithms often struggle to capture O2Jam difficulty curves because O2Jam charts rely heavily on long-note density, asymmetric chord jacks, and prolonged stamina drain.

This model learns directly from raw chart structure:

1. **Note Grid Input:** The chart is converted into a discrete 20ms time-grid across all 7 columns, distinguishing taps, hold heads, hold bodies, and releases.
2. **Dilated CNN Backbone:** 1D dilated residual convolutions extract pattern density, chord transitions, and repetitive jack strain across a wide receptive field (~10.1 seconds / 505 frames).
3. **Multi-Scale Pooling & Gating:** Combines sustain-damped 2s burst pooling, 5s phrase density, 15s climax pooling, and peak-relative active stamina gating into a bounded linear-core Double-Softplus rating (`[1.0, 195.0]`).
4. **Clean Dataset:** Evaluated across 4,423 deduplicated O2Jam charts (from osu! converts, NSJAM, and private server archives), filtered to eliminate dummy labels and corrupt files.

### Benchmark Summary

The primary benchmark is evaluated on the held-out validation split (15%, $n=664$ charts, unseen during training). Full-dataset metrics ($n=4,423$, which include the 85% training set) are provided as secondary reference.

| Metric | Held-Out Validation ($n=664$) | Full Dataset ($n=4,423$, 85% Train) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.52 levels** | 4.69 levels |
| **Root Mean Squared Error (RMSE)** | **8.14 levels** | 6.69 levels |
| **Pearson Correlation ($R$)** | **0.9776** | 0.9840 |
| **Median Absolute Error (P50)** | **$\le 3.80$ levels** | $\le 3.33$ levels |
| **90th Percentile Error (P90)** | **$\le 12.74$ levels** | $\le 10.38$ levels |

Comprehensive validation tables, tier breakdowns, and duration analysis are documented in [`docs/benchmarks.md`](docs/benchmarks.md).

---

## Repository Structure

```
o2jam-level-calc/
├── docs/                      # Technical documentation & benchmarks
│   ├── overview.md            # System architecture and data pipeline
│   ├── architecture.md        # Neural network layers and strain channels
│   ├── dataset.md             # Data sources, binary .ojn parsing & deduplication
│   ├── benchmarks.md          # Benchmark results and error breakdowns
│   └── extension.md           # Tosu overlay installation and configuration
├── scripts/                   # Training and analysis scripts
│   ├── step1_inspect.py       # Scans and validates input charts
│   ├── step2_build_dataset.py # Generates numpy arrays from .osu and .ojn
│   ├── step3_train.py         # PyTorch training and ONNX export
│   ├── predict.py             # CLI single-chart predictor
│   ├── analyze_errors.py      # PyTorch held-out validation error analysis
│   └── analyze_errors_onnx.py # Full ONNX benchmark suite
├── extension/                 # Standalone tosu static overlay
│   └── o2jam_level_predictor/ # Ready to drop into tosu/static/
├── model.onnx                 # Production ONNX model
└── requirements.txt
```

---

## Documentation

- [Overview & Pipeline](docs/overview.md)
- [Model Architecture](docs/architecture.md)
- [Dataset & Deduplication](docs/dataset.md)
- [Benchmarks & Error Analysis](docs/benchmarks.md)
- [Tosu Extension](docs/extension.md)

---

## Limitations

- **Personal playtesting ceiling:** I'm no ET, I can only judge the accuracy of the predictor up to level 85. So predictions beyond this range rely strictly on community dataset labels and benchmark metrics rather than first-hand playtesting.
- **High-level charts (Lv. 120+):** Extreme difficulty charts have wider variance and fewer community consensus charts to train on, leading to higher prediction margins.
- **Rice charts:** Traditional O2Jam difficulty grading revolves almost entirely around long-note (LN) dense charts. Pure rice (no LN) charts may produce unexpected ratings as the model expects standard O2Jam chart dynamics.
- **Full LN & complex release coordination:** Inverse/full-LN charts (90%+ LN) and complex multi-finger release independence may still be slightly under-rated compared to human feeling.
- **Hobbyist / AI-assisted project:** This project was developed experimentally with AI assistance without formal machine learning qualifications. If you have experience in ML or rhythm game difficulty modeling, you are more than welcome to fork, adapt, or improve upon it.

---

## License

MIT

---

## Acknowledgments

- [O2Jam osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975) for community chart archives.
- [o2jam-to-osu](https://github.com/LuzianU/o2jam-to-osu) for `.ojn` binary format reference.
- [tosu](https://github.com/KotRikD/tosu) for the memory reading and overlay platform.
