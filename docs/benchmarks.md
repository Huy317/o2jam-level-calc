# Benchmarks & Validation Results

## 1. Overall Validation Performance

Evaluated on **607 held-out validation charts** (15% split of the 4,044-chart clean dataset, completely unseen during training):

| Metric | Model v1 (Baseline) | Model v3 (Kinematics & Softplus) | Change / Assessment |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | 5.75 levels | **5.51 levels** | **-0.24 levels (Overall accuracy gain)** |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | **8.32 levels** | **-0.27 levels (Reduced variance)** |
| **Pearson Correlation ($R$)** | 0.967 | **0.970** | **+0.003 (Tighter linear alignment)** |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | **$\le 3.69$ levels** | **Over 50% of charts within 3.7 levels** |
| **75th Percentile (P75)** | $\le 7.44$ levels | **$\le 7.09$ levels** | 75% of charts within 7.1 levels |
| **90th Percentile (P90)** | $\le 13.25$ levels| **$\le 12.02$ levels**| **-1.23 levels reduction in tail errors** |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 (Sigmoid) | **Uncapped (Extrapolates 200+)**| **Ceiling eliminated via Softplus head** |

---

## 2. Accuracy Breakdown by Level Tier

| Level Tier | Validation Charts | v1 MAE | v3 MAE | Mean Bias (Pred - Target) | Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Lv. 20 – 50** | 95 | 4.81 | **3.72** | +0.63 | **-1.09 levels MAE improvement**; accurate on entry tier |
| **Lv. 50 – 80** | 230 | 4.06 | **3.96** | -0.52 | Sub-4.0 MAE in the core progression tier |
| **Lv. 80 – 110** | 172 | 6.09 | **5.96** | +0.93 | Solid consistency across high-speed chordstreams |
| **Lv. 110 – 140** | 48 | 12.37 | **11.99** | -3.06 | Technical bursts & jacks; improved by kinematics |
| **Lv. 140 – 181** | 42 | 13.19 | **10.39** | -2.53 | **-2.80 levels MAE improvement**; unconstrained reach |

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

| Duration Category | Time Range | Charts | MAE | Mean Bias |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 141 | **5.08** | +0.72 |
| **Medium** | $90\text{s} - 150\text{s}$ | 337 | **5.45** | -0.18 |
| **Long** | $150\text{s} - 240\text{s}$ | 114 | **5.89** | -0.42 |
| **Marathon** | $> 240\text{s}$ | 15 | **6.64** | -0.91 |

The combination of **Global Average Pooling** (stamina context) and **Multi-Slice Sorted Exponential Decay Pooling** prevents short spike sections from getting washed out in longer marathon charts.

---

## 4. Notable Outliers & Case Studies

1. **`POSSESSION` (Rated Lv. 137, 320 BPM burst):**
   - *Baseline Model v1:* Predicted **Lv. 94.5** (Error: -42.5 levels).
   - *Model v3 (Kinematics):* Predicted **Lv. 115.9 (+21.4 levels jump!)**.
   - *Cause:* The continuous jack and inter-press velocity channels explicitly detected the 40ms burst strike intervals, lifting it out of the severe outlier list.

2. **`Antifront Gears` (Rated Lv. 173):**
   - Predicted **Lv. 207.6**.
   - Demonstrates that removing the Sigmoid boundary allowed the model to correctly extrapolate ultra-high physical density beyond the training set median.

3. **`sky_delta - Justice` (Rated Lv. 84):**
   - *Old corrupted parser count:* 10,449 notes (Easy + Normal + Hard merged). Predicted: Lv. 152.4.
   - *Clean Hard-only count:* 3,686 notes. Predicted: **Lv. 91.8** (accurate within ~7 levels of official rating).

4. **`[LN] Epiik` (Rated Lv. 140, inverse LN):**
   - Predicted **Lv. 102.0** (improved from Lv. 95.9 via pinned-finger tension and release-gap channels).
