# O2Jam 7K Level Predictor

An AI-powered difficulty estimator that predicts human-assigned **O2Jam difficulty levels (Lv. 15 to Lv. 185+)** for 7-key osu!mania and O2Jam charts directly from raw note timings.

Includes an **instant in-game HUD overlay extension** for [Tosu](https://github.com/KotRikD/tosu) powered by client-side WebAssembly ONNX inference.

---

## Highlights

- **Unified 4,055-Chart Dataset:** Deduplicated across classic DMJAM, modern NSJAM (`.ojn` binary parser), and O2EZ community scales.
- **High Accuracy:** **Validation MAE 5.75**, **Pearson $R = 0.967$**, with **50% of all charts predicted within $\le 3.88$ levels** (Core Lv. 50–80 MAE is **4.06**).
- **Physical Rate Scaling (DT / HT):** Automatically compresses or expands note timing in real-time when Double Time (1.5x DT) or Half Time (0.75x HT) is toggled in osu!, displaying dedicated styled badges.
- **Sub-15ms Inference:** Runs locally via [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) using SIMD-threaded WebAssembly with zero network calls.
- **Compact Footprint:** Single 4.55 MB quantized ONNX model (`model.onnx`).

---

## Quickstart: Tosu In-Game Overlay

No Python or machine learning setup required.

1. **Download:** Grab the latest `o2jam_level_predictor.zip` from [GitHub Releases](https://github.com/Huy317/o2jam-level-calc/releases).
2. **Install:** Extract the folder directly into your Tosu `static/` directory:
   ```
   tosu/
   └── static/
       └── o2jam_level_predictor/
           ├── index.html
           ├── index.js
           ├── style.css
           ├── model.onnx
           └── ...
   ```
3. **Launch:** Start Tosu and osu!. Open `http://127.0.0.1:24050/o2jam_level_predictor/` in your browser or enable it in Tosu's In-Game Overlay manager.

---

## CLI Prediction

Predict the O2Jam level of any local 7K `.osu` file:

```bash
pip install -r requirements.txt
python scripts/predict.py "path/to/chart.osu"
```

Output:
```
Loaded ONNX model: model.onnx
Parsed Beatmap: xi - FREEDOM DiVE [FOUR DIMENSIONS]
Mode: Mania 7K | Notes: 4,218 | Duration: 254.2s (NPS: 16.6)
Predicted O2Jam Level: Lv. 138.4 (Rounded: Lv. 138)
```

---

## Repository Structure

```
o2jam-level-calc/
├── docs/                      # Technical documentation & benchmarks
│   ├── overview.md            # High-level architecture & pipeline workflow
│   ├── architecture.md        # Neural network design, receptive field & loss
│   ├── dataset.md             # Data sources, binary .ojn parser & deduplication
│   ├── benchmarks.md          # Validation metrics, percentiles & tier accuracy
│   └── extension.md           # Tosu overlay setup, WASM runtime & mod scaling
├── scripts/                   # Pipeline & diagnostics tools
│   ├── step1_inspect.py       # Scans and validates charts
│   ├── step2_build_dataset.py # Builds discrete (T, 7) grids from .osu and .ojn
│   ├── step3_train.py         # PyTorch training & ONNX export
│   ├── predict.py             # CLI single-chart inference
│   ├── analyze_errors.py      # PyTorch model validation analysis
│   └── analyze_errors_onnx.py # Production ONNX benchmark suite
├── extension/                 # Standalone Tosu static overlay package
│   └── o2jam_level_predictor/ # Ready to deploy into tosu/static/
├── data/                      # Dataset arrays & metadata
├── model.onnx                 # Production ONNX model (4.55 MB)
├── best_model.pt              # PyTorch trained checkpoint
└── requirements.txt
```

---

## Documentation

For comprehensive technical specifications, explore the `docs/` directory:
- [Overview & Architecture Workflow](docs/overview.md)
- [Neural Network Architecture (Dilated Conv1D + Multi-Scale Pooling)](docs/architecture.md)
- [Dataset Specifications & Native Binary .ojn Decoding](docs/dataset.md)
- [Benchmarks, Percentiles & Tier Analysis](docs/benchmarks.md)
- [Tosu Extension Integration & Rate Scaling](docs/extension.md)

---

## Credits & Acknowledgments

- **Chart Dataset:** Special thanks to the [O2Jam to osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975) by the osu! community for compiling and maintaining the DMJAM and O2EZ chart packs used to build our training dataset.
- **Tosu:** Built to integrate with [Tosu](https://github.com/KotRikD/tosu) by KotRikD.

---

## License

MIT License
