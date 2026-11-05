# Benchmarks & Validation Results

## 1. Overall Performance Across Unified Dataset (4,423 Clean Charts)

Evaluated across all 4,423 clean, unique charts (with dummy `level <= 3` custom gimmick charts filtered out) using the exported ONNX model (`model.onnx`):

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Clean 4,423 Charts) | Change / Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | **5.34 levels** | Sub-5.4 levels on clean dataset |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | **7.85 levels** | **Lowest overall variance (outlier contamination removed)** |
| **Pearson Correlation ($R$)** | 0.967 | 0.9715 | **0.9785** | **Highest linear correlation achieved** |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | **$\le 3.67$ levels** | Over 50% of charts within 3.7 levels |
| **75th Percentile (P75)** | $\le 7.44$ levels | $\le 9.46$ levels | **$\le 6.88$ levels** | 75% of charts within 6.9 levels |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| **$\le 11.71$ levels**| Top 90% well within standard Dan bracket |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | **Bounded [1.0, 185.0]** | Spans full human range [4.0, 180.0] |

---

## 2. Accuracy Breakdown by Level Tier (4,423 Charts)

| Level Tier | Charts | Model v4 MAE | Model v4 RMSE | Mean Bias (Pred - Target) | Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Lv. 20 – 45** | 540 | **3.45** | 4.78 | +0.74 | High precision on entry/beginner tier |
| **Lv. 45 – 70** | 896 | **4.32** | 5.88 | +1.02 | Solid alignment across core progression charts |
| **Lv. 70 – 95** | 1,678 | **4.08** | 5.81 | -1.27 | Sub-4.1 MAE across standard chordstreams |
| **Lv. 95 – 120** | 430 | **9.07** | 12.14 | +1.11 | Handles technical dense bursts without drift |
| **Lv. 120 – 180** | 745 | **9.14** | 12.05 | -0.17 | Neutral mean bias across elite Dan tier |

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

| Duration Category | Time Range | Charts | Model v4 MAE | Model v4 Bias |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 160 | **6.53** | +4.43 |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,080 | **5.66** | +1.78 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,595 | **4.79** | -1.63 |
| **Marathon** | $> 240\text{s}$ | 588 | **5.38** | -3.42 |

Length-bucketed training and exact unpadded masked pooling ensure single-chart inference does not suffer from zero-padding dilution.

---

## 4. Notable Outliers & Case Studies

1. **`POSSESSION` (Rated Lv. 137, 320 BPM burst):**
   - Accurately captures high density without inflating into theoretical runaway territory.

2. **`Almagest` (Rated Lv. 173):**
   - Cleanly seated in the peak Lv. 160–175 tier.

3. **`sky_delta - Justice` (Rated Lv. 84):**
   - Accurate within standard human rating variance.

4. **Filtering Dummy Header Levels:**
   - Filtering `level <= 3` cleanly eliminated 14 custom gimmick/meme charts with dummy Lv. 2 headers and 30–47 NPS (e.g. *HandAttack*, *Saika*), completely removing all $+70$ to $+155$ artificial error spikes without affecting legitimate beginner progression charts (Lv. 4+).
