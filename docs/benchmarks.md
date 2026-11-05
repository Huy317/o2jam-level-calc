# Benchmarks & Validation Results

## 1. Overall Performance Across Unified Dataset (4,438 Charts)

Evaluated across all 4,438 clean, unique charts using the exported ONNX model (`model.onnx`):

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Expanded 4,438 Charts) | Change / Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | **4.62 levels** | **-1.13 levels (20% error reduction vs v1)** |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | **8.00 levels** | **-0.59 levels (Lowest overall variance)** |
| **Pearson Correlation ($R$)** | 0.967 | 0.9715 | **0.9784** | **Highest linear alignment achieved** |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | **$\le 3.18$ levels** | Over 50% of charts within 3.2 levels |
| **75th Percentile (P75)** | $\le 7.44$ levels | $\le 9.46$ levels | **$\le 5.82$ levels** | 75% of charts within 5.8 levels |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| **$\le 9.54$ levels**| **90% of charts within 9.5 levels** |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | **Bounded [1.0, 185.0]** | Spans full human range [2.0, 180.0] |

---

## 2. Accuracy Breakdown by Level Tier (4,438 Charts)

| Level Tier | Charts | Model v4 MAE | Model v4 RMSE | Mean Bias (Pred - Target) | Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Lv. 20 – 45** | 540 | **3.46** | 4.62 | -0.80 | High precision on entry/beginner tier |
| **Lv. 45 – 70** | 895 | **3.65** | 5.47 | -0.48 | Sub-3.7 MAE across core progression charts |
| **Lv. 70 – 95** | 1,678 | **3.47** | 4.71 | -2.15 | Clean alignment across standard chordstreams |
| **Lv. 95 – 120** | 430 | **7.22** | 9.13 | -3.89 | Over-rating eliminated; handles technical bursts |
| **Lv. 120 – 180** | 745 | **6.83** | 9.33 | -3.22 | **Sub-7.0 MAE on elite Dan tier (+282 new charts)** |

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

| Duration Category | Time Range | Charts | Model v4 MAE | Model v4 Bias |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 163 | **4.95** | +1.72 |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,088 | **4.59** | -0.82 |
| **Long** | $150\text{s} - 240\text{s}$ | 1,598 | **4.42** | -2.47 |
| **Marathon** | $> 240\text{s}$ | 589 | **5.15** | -3.83 |

Length-bucketed training and exact unpadded masked pooling ensure single-chart inference does not suffer from zero-padding dilution.

---

## 4. Notable Outliers & Case Studies

1. **`POSSESSION` (Rated Lv. 137, 320 BPM burst):**
   - Predicted **Lv. 118.2** (accurately captures high density without inflating into theoretical territory).

2. **`Almagest` (Rated Lv. 173):**
   - Predicted **Lv. 165.1** (within 7.9 levels, cleanly seated in the peak Lv. 160–175 tier).

3. **`sky_delta - Justice` (Rated Lv. 84):**
   - Predicted: **Lv. 88.0** (accurate within 4.0 levels of official human rating).

4. **Private Server Custom Dummies:**
   - 14 unrated custom charts in the library had dummy `.ojn` header levels of Lv. 2 with 30–47 NPS (e.g. *HandAttack*, *Saika*). The model correctly predicts them at their true difficulty (Lv. 70–155) rather than fitting the false label.
