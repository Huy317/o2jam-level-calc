# Benchmarks & Validation Results

## 1. Performance Overview

The model was evaluated using the exported model checkpoint (`best_model.pt` / `model.onnx`). Metrics are presented for both the held-out validation split (unseen during training) and the full dataset (which includes the 85% of charts used for training).

### Primary Metrics: Held-Out Validation Split vs. Full Dataset

| Metric | Held-Out Validation (15%, $n=664$) | Full Dataset ($n=4,423$, 85% Train) | Train-Only ($n=3,759$) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.52 levels** | 4.69 levels | 4.55 levels |
| **Root Mean Squared Error (RMSE)** | **8.14 levels** | 6.69 levels | 6.40 levels |
| **Pearson Correlation ($R$)** | **0.9776** | 0.9840 | 0.9852 |
| **Median Absolute Error (P50)** | **$\le 3.80$ levels** | $\le 3.33$ levels | $\le 3.27$ levels |
| **75th Percentile (P75)** | **$\le 6.84$ levels** | $\le 6.19$ levels | $\le 6.06$ levels |
| **90th Percentile (P90)** | **$\le 12.74$ levels** | $\le 10.38$ levels | $\le 10.03$ levels |

> **Note on Sample Size & Measurement Error:**
> With $n=664$, the standard error of the MAE is roughly $\pm 0.25$ levels. Consequently, subtle differences between model iterations (such as v4 vs v5) are within statistical noise. Full-dataset numbers heavily reflect the training set and understate generalization error.

### Historical Comparison (Full Dataset, Non-Held-Out)

> **Important:** The historical v1–v5 numbers below were evaluated across the entire dataset (including training data) and are **not comparable** to held-out validation numbers. They are preserved here strictly for architectural progression reference.

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Sigmoid) | Model v5 (Initial) | Model v5 (`FixPad` Masked Pooling) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | 5.34 levels | 5.26 levels | **4.69 levels** (Val: **5.52**) |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | 7.85 levels | 7.48 levels | **6.69 levels** (Val: **8.14**) |
| **Pearson Correlation ($R$)** | 0.9670 | 0.9715 | 0.9785 | 0.9800 | **0.9840** (Val: **0.9776**) |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | $\le 3.67$ levels | $\le 3.72$ levels | **$\le 3.33$ levels** (Val: **$\le 3.80$**) |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| $\le 11.71$ levels| $\le 11.62$ levels| **$\le 10.38$ levels** (Val: **$\le 12.74$**) |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | Bounded [1.0, 185.0] | Bounded [1.0, 195.0] | **Bounded [1.0, 195.0]** |

---

## 2. Accuracy Breakdown by Level Tier

### Held-Out Validation Split ($n=664$)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 17 | 1.62 | 2.14 | +0.45 |
| **Lv. 20 – 45** | 82 | 4.46 | 5.44 | +2.32 |
| **Lv. 45 – 70** | 145 | 4.01 | 5.24 | +0.99 |
| **Lv. 70 – 95** | 240 | 4.34 | 6.78 | -0.58 |
| **Lv. 95 – 120** | 60 | 9.37 | 12.30 | +2.19 |
| **Lv. 120 – 185** | 120 | 9.05 | 12.04 | -3.04 |

### Full Dataset ($n=4,423$, includes training data)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 1 – 20** | 132 | 2.20 | 3.01 | +1.30 |
| **Lv. 20 – 45** | 540 | 4.45 | 5.84 | +3.34 |
| **Lv. 45 – 70** | 896 | 3.49 | 4.80 | +0.95 |
| **Lv. 70 – 95** | 1,678 | 3.68 | 5.18 | -0.79 |
| **Lv. 95 – 120** | 430 | 7.58 | 9.79 | +0.75 |
| **Lv. 120 – 185** | 747 | 7.38 | 9.79 | -1.36 |

The bulk of overall prediction error is concentrated in tiers above Lv. 95, where community rating consensus is noisier and fewer charts exist.

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

### Held-Out Validation Split ($n=664$)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 24 | 10.23 | +5.22 |
| **Medium** | $90\text{s} - 150\text{s}$ | 322 | 5.51 | +1.08 |
| **Long** | $150\text{s} - 240\text{s}$ | 238 | 5.12 | -1.61 |
| **Marathon** | $> 240\text{s}$ | 80 | 5.33 | -1.49 |

### Full Dataset ($n=4,423$, includes training data)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 160 | 5.88 | +3.67 |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,080 | 4.90 | +1.59 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,595 | 4.26 | -1.06 |
| **Marathon** | $> 240\text{s}$ | 588 | 4.83 | -2.37 |

---

## 4. Anecdotal Observations & Outlier Notes

The following qualitative observations represent informal spot-checks rather than verified benchmark claims:

1. **High-Density Bursts:** On fast burst charts, the bounded head prevents runaway exponential inflation while tracking rapid stream density.
2. **Extreme Tier Dispersion:** Charts rated Lv. 160+ exhibit wider error margins ($\text{MAE} > 10$), reflecting sparse training samples and divergent cross-server difficulty conventions.
3. **Pure LN Charts:** Inverse or pure-LN charts (where hold notes constitute >90% of objects) often display slight negative bias (~1.5–2 levels), as physical release coordination and finger independence are not explicitly weighted by simple grid occupancy.
4. **Cross-Server Consensus Noise Floor:** Across bitwise-identical 20ms note grids present on multiple servers, human rating disagreement is **0.53 MAE** (`private` vs `dmjam`, $n=366$), **1.14 MAE** (`private` vs `nsjam`, $n=357$), and **3.13 MAE** (`nsjam` vs `o2ez`, $n=47$, where `o2ez` rates ~2.1 levels lower on average).
