# Benchmarks & Validation Results

## 1. Performance Overview

The model was evaluated using the exported model checkpoint (`best_model.pt` / `model.onnx`). Metrics are presented for both the held-out validation split (unseen during training) and the full dataset (which includes the 85% of charts used for training).

### Primary Metrics: Held-Out Validation Split vs. Full Dataset

| Metric | Held-Out Validation (15%, $n=600$) | Full Dataset ($n=3,997$, 85% Train) | Train-Only ($n=3,397$) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.59 levels** | 4.96 levels | 4.85 levels |
| **Root Mean Squared Error (RMSE)** | **8.01 levels** | 7.07 levels | 6.89 levels |
| **Pearson Correlation ($R$)** | **0.9774** | 0.9838 | 0.9848 |
| **Median Absolute Error (P50)** | **$\le 3.78$ levels** | $\le 3.44$ levels | $\le 3.38$ levels |
| **75th Percentile (P75)** | **$\le 7.03$ levels** | $\le 6.54$ levels | $\le 6.45$ levels |
| **90th Percentile (P90)** | **$\le 13.12$ levels** | $\le 11.32$ levels | $\le 10.99$ levels |

> **Note on Sample Size & Measurement Error:**
> With $n=600$, the standard error of the MAE is roughly $\pm 0.25$ levels. Excluding the 426 mid-tier `o2ez` charts shifted the dataset mean level from $81.8 \rightarrow 82.7$ (increasing the share of Lv. 95–180 charts), while reducing validation RMSE from $8.34 \rightarrow 8.01$ and improving accuracy across all high-difficulty tiers ($\text{Lv. } 70+$) and long/marathon durations ($>150\text{s}$).

### Historical Comparison (Full Dataset, Non-Held-Out)

> **Important:** The historical v1–v5 numbers below were evaluated across the entire dataset (including training data) and are **not comparable** to held-out validation numbers. They are preserved here strictly for architectural progression reference.

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Sigmoid) | Model v5 (Initial) | Model v5 (`FixPad` + Clean 3,997 Dataset) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | 5.34 levels | 5.26 levels | **4.96 levels** (Val: **5.59**) |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | 7.85 levels | 7.48 levels | **7.07 levels** (Val: **8.01**) |
| **Pearson Correlation ($R$)** | 0.9670 | 0.9715 | 0.9785 | 0.9800 | **0.9838** (Val: **0.9774**) |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | $\le 3.67$ levels | $\le 3.72$ levels | **$\le 3.44$ levels** (Val: **$\le 3.78$**) |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| $\le 11.71$ levels| $\le 11.62$ levels| **$\le 11.32$ levels** (Val: **$\le 13.12$**) |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | Bounded [1.0, 185.0] | Bounded [1.0, 195.0] | **Bounded [1.0, 195.0]** |

---

## 2. Accuracy Breakdown by Level Tier

### Held-Out Validation Split ($n=600$)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 20 | 1.56 | 1.85 | -0.59 |
| **Lv. 20 – 45** | 69 | 5.69 | 7.86 | +3.35 |
| **Lv. 45 – 70** | 128 | 4.59 | 6.36 | +2.57 |
| **Lv. 70 – 95** | 224 | 4.18 | 6.18 | +1.81 |
| **Lv. 95 – 120** | 59 | 8.90 | 11.33 | +3.69 |
| **Lv. 120 – 185** | 100 | 8.81 | 11.34 | -4.03 |

### Full Dataset ($n=3,997$, includes training data)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 132 | 2.03 | 2.68 | -0.58 |
| **Lv. 20 – 45** | 495 | 4.12 | 5.68 | +2.26 |
| **Lv. 45 – 70** | 784 | 4.20 | 5.53 | +2.63 |
| **Lv. 70 – 95** | 1,456 | 3.76 | 5.46 | +1.66 |
| **Lv. 95 – 120** | 397 | 7.71 | 10.16 | +4.21 |
| **Lv. 120 – 185** | 733 | 7.76 | 10.09 | -2.46 |

The bulk of overall prediction error is concentrated in tiers above Lv. 95, where community rating consensus is noisier and fewer charts exist.

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

### Held-Out Validation Split ($n=600$)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 27 | 8.22 | +5.01 |
| **Medium** | $90\text{s} - 150\text{s}$ | 276 | 6.39 | +2.53 |
| **Long** | $150\text{s} - 240\text{s}$ | 232 | 4.57 | +0.19 |
| **Marathon** | $> 240\text{s}$ | 65 | 4.75 | -1.70 |

### Full Dataset ($n=3,997$, includes training data)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 155 | 6.65 | +4.81 |
| **Medium** | $90\text{s} - 150\text{s}$ | 1,960 | 5.63 | +2.35 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,424 | 4.11 | +0.42 |
| **Marathon** | $> 240\text{s}$ | 458 | 4.16 | -1.21 |

---

## 4. Anecdotal Observations & Outlier Notes

The following qualitative observations represent informal spot-checks rather than verified benchmark claims:

1. **High-Density Bursts:** On fast burst charts, the bounded head prevents runaway exponential inflation while tracking rapid stream density.
2. **Extreme Tier Dispersion:** Charts rated Lv. 160+ exhibit wider error margins ($\text{MAE} > 10$), reflecting sparse training samples and divergent cross-server difficulty conventions.
3. **Pure LN Charts:** Inverse or pure-LN charts (where hold notes constitute >90% of objects) often display slight negative bias (~1.5–2 levels), as physical release coordination and finger independence are not explicitly weighted by simple grid occupancy.
4. **Cross-Server Consensus Noise Floor:** Across bitwise-identical 20ms note grids present on multiple servers, human rating disagreement is **0.53 MAE** (`private` vs `dmjam`, $n=366$), **1.14 MAE** (`private` vs `nsjam`, $n=357$), and **3.13 MAE** (`nsjam` vs `o2ez`, $n=47$, where `o2ez` rates ~2.1 levels lower on average).
