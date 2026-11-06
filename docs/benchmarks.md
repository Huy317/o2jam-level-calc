# Benchmarks & Validation Results

## 1. Performance Overview

The model was evaluated using the exported model checkpoint (`best_model.pt` / `model.onnx`). Metrics are presented for both the held-out validation split (unseen during training) and the full dataset (which includes the 85% of charts used for training).

### Primary Metrics: Held-Out Validation Split vs. Full Dataset

| Metric | Held-Out Validation (15%, $n=664$) | Full Dataset ($n=4,423$, 85% Train) | Train-Only ($n=3,759$) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **5.86 levels** | 5.26 levels | 5.15 levels |
| **Root Mean Squared Error (RMSE)** | **8.34 levels** | 7.48 levels | 7.31 levels |
| **Pearson Correlation ($R$)** | **0.9765** | 0.9800 | 0.9806 |
| **Median Absolute Error (P50)** | **$\le 4.17$ levels** | $\le 3.72$ levels | $\le 3.65$ levels |
| **75th Percentile (P75)** | **$\le 7.28$ levels** | $\le 6.83$ levels | $\le 6.68$ levels |
| **90th Percentile (P90)** | **$\le 13.00$ levels** | $\le 11.62$ levels | $\le 11.40$ levels |

> **Note on Sample Size & Measurement Error:**
> With $n=664$, the standard error of the MAE is roughly $\pm 0.25$ levels. Consequently, subtle differences between model iterations (such as v4 vs v5) are within statistical noise. Full-dataset numbers heavily reflect the training set and understate generalization error.

### Historical Comparison (Full Dataset, Non-Held-Out)

> **Important:** The historical v1–v4 numbers below were evaluated across the entire dataset (including training data) and are **not comparable** to held-out validation numbers. They are preserved here strictly for architectural progression reference.

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Sigmoid) | Model v5 (Linear-Core & Gating, Full Set) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | 5.34 levels | **5.26 levels** |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | 7.85 levels | **7.48 levels** |
| **Pearson Correlation ($R$)** | 0.9670 | 0.9715 | 0.9785 | **0.9800** |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | $\le 3.67$ levels | **$\le 3.72$ levels** |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| $\le 11.71$ levels| **$\le 11.62$ levels**|
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | Bounded [1.0, 185.0] | **Bounded [1.0, 195.0]** |

---

## 2. Accuracy Breakdown by Level Tier

> **Note:** Tier evaluation buckets cover Lv. 20 to 179. Charts with $\text{Lv} < 20$ and $\text{Lv} \ge 180$ (134 charts in the full set, 17 in held-out validation) are excluded from these tier rows.

### Held-Out Validation Split ($n=664$, 647 in tiers)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 20 – 45** | 82 | 3.47 | 4.42 | +1.37 |
| **Lv. 45 – 70** | 145 | 5.17 | 6.33 | -0.14 |
| **Lv. 70 – 95** | 240 | 4.33 | 6.46 | -0.25 |
| **Lv. 95 – 120** | 60 | 8.18 | 11.26 | +1.49 |
| **Lv. 120 – 180** | 120 | 10.55 | 13.18 | -4.27 |

### Full Dataset ($n=4,423$, 4,289 in tiers, includes training data)

| Level Tier | Charts | MAE | RMSE | Mean Bias (Pred - Actual) |
| :---: | :---: | :---: | :---: | :---: |
| **Lv. 20 – 45** | 540 | 3.74 | 5.36 | +2.19 |
| **Lv. 45 – 70** | 896 | 4.48 | 5.97 | -0.14 |
| **Lv. 70 – 95** | 1,678 | 3.75 | 5.25 | +0.00 |
| **Lv. 95 – 120** | 430 | 7.18 | 9.54 | -0.06 |
| **Lv. 120 – 180** | 745 | 9.73 | 12.21 | -2.59 |

The bulk of overall prediction error is concentrated in tiers above Lv. 95, where community rating consensus is noisier and fewer charts exist.

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

### Held-Out Validation Split ($n=664$)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 24 | 8.98 | +5.23 |
| **Medium** | $90\text{s} - 150\text{s}$ | 322 | 5.80 | +1.05 |
| **Long** | $150\text{s} - 240\text{s}$ | 238 | 5.89 | -2.61 |
| **Marathon** | $> 240\text{s}$ | 80 | 5.08 | -2.13 |

### Full Dataset ($n=4,423$, includes training data)

| Duration Category | Time Range | Charts | MAE | Mean Bias (Pred - Actual) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 160 | 5.74 | +3.37 |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,080 | 5.61 | +1.64 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,595 | 4.77 | -1.56 |
| **Marathon** | $> 240\text{s}$ | 588 | 5.23 | -3.08 |

---

## 4. Anecdotal Observations & Outlier Notes

The following qualitative observations represent informal spot-checks rather than verified benchmark claims:

1. **High-Density Bursts:** On fast burst charts, the bounded head prevents runaway exponential inflation while tracking rapid stream density.
2. **Extreme Tier Dispersion:** Charts rated Lv. 160+ exhibit wider error margins ($\text{MAE} > 10$), reflecting sparse training samples and divergent cross-server difficulty conventions.
3. **Pure LN Charts:** Inverse or pure-LN charts (where hold notes constitute >90% of objects) often display slight negative bias (~1.5–2 levels), as physical release coordination and finger independence are not explicitly weighted by simple grid occupancy.

---

## 5. Known Issues

- **Checkpoint Metadata Mismatch:** `best_model.pt` records `val_mae: 5.28` in its checkpoint dictionary (epoch 17). However, re-evaluating the checkpoint weights on its stored `val_indices` with current code and data yields an MAE of `5.86`. The root cause is undetermined, likely due to code adjustments or dataset ordering drift that occurred after training finished.
