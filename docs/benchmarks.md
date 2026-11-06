# Benchmarks & Validation Results

## 1. Performance Overview

The model was evaluated using the exported model checkpoint (`best_model.pt` / `model.onnx`). Metrics are presented for both the held-out validation split (unseen during training) and the full dataset (which includes the 85% of charts used for training).

### Primary Metrics: Held-Out Validation Split vs. Full Dataset

| Metric | Canonical Held-Out Val ($n=613$, Modern Servers) | All Held-Out Val ($n=664$, 15% Split) | Full Dataset ($n=4,423$, 85% Train) | Train-Only ($n=3,759$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.19 levels** | **5.24 levels** | 4.45 levels | 4.31 levels |
| **Root Mean Squared Error (RMSE)** | **7.70 levels** | **7.67 levels** | 6.30 levels | 6.02 levels |
| **Pearson Correlation ($R$)** | **0.9785** | **0.9772** | 0.9859 | 0.9873 |
| **Median Absolute Error (P50)** | **$\le 3.57$ levels** | **$\le 3.65$ levels** | $\le 3.26$ levels | $\le 3.19$ levels |
| **75th Percentile (P75)** | **$\le 6.69$ levels** | **$\le 6.69$ levels** | $\le 5.80$ levels | $\le 5.64$ levels |
| **90th Percentile (P90)** | **$\le 11.17$ levels** | **$\le 11.26$ levels** | $\le 9.42$ levels | $\le 9.14$ levels |

> **Note on Multi-Server Calibration (`ServerCalibrationAdapter`):**
> On the exact same `SEED=42` split of the 4,423-chart dataset (`priority: private > nsjam > dmjam > o2ez`), adding the train-only 4-knot piecewise-linear `ServerCalibrationAdapter` reduced Canonical Held-Out Validation MAE from **5.68 $\rightarrow$ 5.19 levels** (and Full Validation MAE from **5.71 $\rightarrow$ 5.24 levels**, RMSE from **8.78 $\rightarrow$ 7.67 levels**), outperforming both unadjusted labels (`5.68`) and static empirical label shifting (`5.48`).

### Historical Comparison (Full Dataset & Held-Out Validation)

> **Important:** Historical v1–v4 numbers were evaluated across the entire dataset (including training data). Model v5 rows include both Full-Dataset and Held-Out Validation ($n=664$) metrics.

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Sigmoid) | Model v5 (Initial) | Model v5 (`FixPad` + `ServerCalibrationAdapter`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | 5.34 levels | 5.26 (Val: 5.86) | **4.45 levels** (Val: **5.24** / Canon: **5.19**) |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | 7.85 levels | 7.48 (Val: 8.34) | **6.30 levels** (Val: **7.67** / Canon: **7.70**) |
| **Pearson Correlation ($R$)** | 0.9670 | 0.9715 | 0.9785 | 0.9800 (Val: 0.9765) | **0.9859** (Val: **0.9772** / Canon: **0.9785**) |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | $\le 3.67$ levels | $\le 3.72$ (Val: $\le 4.17$) | **$\le 3.26$ levels** (Val: **$\le 3.65$** / Canon: **$\le 3.57$**) |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| $\le 11.71$ levels| $\le 11.62$ (Val: $\le 13.00$)| **$\le 9.42$ levels** (Val: **$\le 11.26$** / Canon: **$\le 11.17$**) |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | Bounded [1.0, 185.0] | Bounded [1.0, 195.0] | **Bounded [1.0, 195.0]** |

---

## 2. Accuracy Breakdown by Level Tier

### Held-Out Validation Split ($n=664$ All / $n=613$ Canonical)

| Level Tier | All Val Charts | All Val MAE (RMSE) | Canon Val Charts | Canon Val MAE (RMSE) | Canon Mean Bias |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 22 | 3.53 (4.06) | 22 | 3.53 (4.06) | +3.37 |
| **Lv. 20 – 45** | 80 | 3.99 (5.38) | 75 | 3.77 (5.13) | +2.45 |
| **Lv. 45 – 70** | 124 | 3.75 (4.94) | 107 | 3.65 (4.87) | +0.73 |
| **Lv. 70 – 95** | 277 | 4.19 (5.73) | 239 | 4.13 (5.75) | -1.87 |
| **Lv. 95 – 120** | 61 | 7.88 (10.65) | 57 | 7.70 (10.50) | +0.51 |
| **Lv. 120 – 185** | 100 | 9.78 (13.19) | 96 | 9.55 (13.03) | -4.23 |

### Full Dataset ($n=4,423$, includes training data)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 132 | 3.29 | 3.80 | +3.20 |
| **Lv. 20 – 45** | 539 | 3.67 | 5.02 | +2.50 |
| **Lv. 45 – 70** | 897 | 3.28 | 4.62 | +0.93 |
| **Lv. 70 – 95** | 1,678 | 3.71 | 4.92 | -1.91 |
| **Lv. 95 – 120** | 430 | 6.87 | 9.16 | -0.86 |
| **Lv. 120 – 185** | 747 | 6.91 | 9.26 | -1.68 |

The bulk of overall prediction error is concentrated in tiers above Lv. 95, where community rating consensus is noisier and fewer charts exist.

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

### Held-Out Validation Split ($n=664$)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 16 | 4.41 | +1.36 |
| **Medium** | $90\text{s} - 150\text{s}$ | 312 | 5.42 | +0.61 |
| **Long** | $150\text{s} - 240\text{s}$ | 247 | 5.15 | -2.17 |
| **Marathon** | $> 240\text{s}$ | 89 | 5.01 | -3.37 |

### Full Dataset ($n=4,423$, includes training data)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 160 | 5.16 | +3.57 |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,080 | 4.51 | +0.73 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,595 | 4.13 | -1.62 |
| **Marathon** | $> 240\text{s}$ | 588 | 4.93 | -2.92 |

---

## 4. Anecdotal Observations & Outlier Notes

The following qualitative observations represent informal spot-checks rather than verified benchmark claims:

1. **High-Density Bursts:** On fast burst charts, the bounded head prevents runaway exponential inflation while tracking rapid stream density.
2. **Extreme Tier Dispersion:** Charts rated Lv. 160+ exhibit wider error margins ($\text{MAE} > 10$), reflecting sparse training samples and divergent cross-server difficulty conventions.
3. **Pure LN Charts:** Inverse or pure-LN charts (where hold notes constitute >90% of objects) often display slight negative bias (~1.5–2 levels), as physical release coordination and finger independence are not explicitly weighted by simple grid occupancy.
4. **Cross-Server Consensus Noise Floor:** Across bitwise-identical 20ms note grids present on multiple servers, human rating disagreement is **0.53 MAE** (`private` vs `dmjam`, $n=366$), **1.14 MAE** (`private` vs `nsjam`, $n=357$), and **3.13 MAE** (`nsjam` vs `o2ez`, $n=47$, where `o2ez` rates ~2.1 levels lower on average).
