# O2Jam 7K Level Predictor

Predicts O2Jam difficulty levels (Lv. 4 to Lv. 180) for 7-key osu!mania charts using a neural network trained on 4,423 O2Jam and community charts.

Includes a lightweight in-game overlay for [tosu](https://github.com/KotRikD/tosu) that displays the estimated level in real time while playing or browsing songs.

---

## tosu In-Game Overlay

1. Download `o2jam_level_predictor.zip` from [GitHub Releases](https://github.com/Huy317/o2jam-level-calc/releases) (or copy [`extension/o2jam_level_predictor`](extension/o2jam_level_predictor) from this repo) into your `tosu/static/` folder.
2. Start **tosu** and **osu!**.
3. Open `http://127.0.0.1:24050/o2jam_level_predictor/` in your browser or add it as an OBS browser source.

- **Runs 100% locally:** Evaluates charts in the browser via WebAssembly ONNX Runtime (`model.onnx`) in ~8–18 ms.
- **Rate aware:** Automatically scales note timings for Double Time (1.5x DT), Half Time (0.75x HT), or custom osu! rates.
- **Configurable:** Customize decimal precision, accent colors, scale, opacity, and song title in the tosu settings dashboard.

---

## CLI Usage

```bash
pip install -r requirements.txt
python scripts/predict.py "path/to/chart.osu"
```

---

## How It Works & Benchmarks

Each chart is quantized into a 20ms `(T, 7)` note grid (taps, hold heads, hold bodies, and releases) and passed through 60 domain-engineered kinematic channels, a 6-block dilated 1D ResNet (~10.1s receptive field), multi-scale phrase pooling, and a bounded Double-Softplus regression head (`[1.0, 195.0]`).

During training, a piecewise-linear `ServerCalibrationAdapter` absorbs legacy server grading offsets (`O2EZ`) so the exported `model.onnx` predicts strictly on the canonical modern O2Jam scale.

| Metric | Canonical Held-Out Val ($n=613$) | All Held-Out Val ($n=664$) | Full Dataset ($n=4,423$, 85% Train) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.19 levels** | **5.24 levels** | 4.45 levels |
| **Root Mean Squared Error (RMSE)** | **7.70 levels** | **7.67 levels** | 6.30 levels |
| **Pearson Correlation ($R$)** | **0.9785** | **0.9772** | 0.9859 |
| **Median Absolute Error (P50)** | **$\le 3.57$ levels** | **$\le 3.65$ levels** | $\le 3.26$ levels |
| **90th Percentile Error (P90)** | **$\le 11.17$ levels** | **$\le 11.26$ levels** | $\le 9.42$ levels |

### Documentation
- [Overview & Pipeline](docs/overview.md) — System workflow and training scripts (`scripts/step1_inspect.py`, `step2_build_dataset.py`, `step3_train.py`)
- [Model Architecture](docs/architecture.md) — Kinematic channels, dilated ResNet, pooling, and server calibration
- [Dataset & Deduplication](docs/dataset.md) — `.ojn` binary parsing, source priority (`private > nsjam > dmjam > o2ez`), and adding new servers
- [Benchmarks & Error Analysis](docs/benchmarks.md) — Tier breakdowns, duration analysis, and historical model comparisons
- [tosu Extension](docs/extension.md) — Overlay setup, rate scaling, and settings reference

---

## Limitations

- **Personal playtesting ceiling:** I'm no ET, I can only judge the accuracy of the predictor up to level 85. So predictions beyond this range rely strictly on community dataset labels and benchmark metrics rather than first-hand playtesting.
- **High-level charts (Lv. 120+):** Extreme difficulty charts have wider variance and fewer community consensus charts to train on, leading to higher prediction margins.
- **Rice charts:** Traditional O2Jam difficulty grading revolves almost entirely around long-note (LN) dense charts. Pure rice (no LN) charts may produce unexpected ratings as the model expects standard O2Jam chart dynamics.
- **Full LN & complex release coordination:** Inverse/full-LN charts (90%+ LN) and complex multi-finger release independence may still be slightly under-rated compared to human feeling.
- **Hobbyist / AI-assisted project:** This project was developed experimentally with AI assistance without formal machine learning qualifications. If you have experience in ML or rhythm game difficulty modeling, you are more than welcome to fork, adapt, or improve upon it.

---

## License & Acknowledgments

MIT License. Special thanks to the [O2Jam osu!mania 7K Conversion Project](https://osu.ppy.sh/community/forums/topics/1831975), [o2jam-to-osu](https://github.com/LuzianU/o2jam-to-osu) for `.ojn` binary format reference, and [tosu](https://github.com/KotRikD/tosu).
