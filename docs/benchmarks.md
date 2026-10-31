# Benchmarks & Validation Results

## 1. Overall Validation Performance

Evaluated on **609 held-out validation charts** (15% split of the 4,055-chart dataset, completely unseen during training):

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **5.75 levels** | Average error across all levels (20–181) |
| **Root Mean Squared Error (RMSE)**| **8.59 levels** | Penalizes severe outlier errors |
| **Pearson Correlation ($R$)** | **0.967** | Exceptionally linear alignment with human grading |
| **Median Absolute Error (P50)** | **$\le 3.88$ levels** | **Over 50% of all maps are within 3.9 levels** |
| **75th Percentile (P75)** | **$\le 7.44$ levels** | 75% of all charts within 7.4 levels |
| **90th Percentile (P90)** | **$\le 13.25$ levels**| 90% of charts within 13.2 levels |
| **95th Percentile (P95)** | **$\le 18.16$ levels**| Top 5% tail error threshold |

---

## 2. Accuracy Breakdown by Level Tier

| Level Tier | Validation Charts | MAE | Mean Bias (Pred - Target) | Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **Lv. 20 – 50** | 98 | **4.81** | +1.49 | High accuracy; slight safe over-estimation on slow charts |
| **Lv. 50 – 80** | 229 | **4.06** | +0.53 | **Peak performance tier** (core O2Jam progression) |
| **Lv. 80 – 110** | 173 | **6.09** | -0.40 | Very accurate; negligible bias |
| **Lv. 110 – 140** | 46 | **12.37** | -4.67 | Technical patterns (jacks vs brackets vs rolls) |
| **Lv. 140 – 181** | 43 | **9.54** | -5.38 | Hardest tier; reduced from 12.10 MAE in v1 |

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

| Duration Category | Time Range | Charts | MAE | Mean Bias |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | $< 90\text{s}$ | 142 | **5.12** | +0.81 |
| **Medium** | $90\text{s} - 150\text{s}$ | 338 | **5.64** | -0.12 |
| **Long** | $150\text{s} - 240\text{s}$ | 114 | **6.41** | -0.84 |
| **Marathon** | $> 240\text{s}$ | 15 | **7.20** | -1.15 |

The combination of **Global Average Pooling** (stamina) and **1s/5s/15s Max Pooling** (burst intensity) maintains balanced accuracy across short TV sizes and 4-minute stamina drain charts.

---

## 4. Notable Outliers Analysis

Large errors typically stem from subjective server ratings or extreme physical gimmicks:
- **`sky_delta - Justice` (Rated Lv. 84, Pred Lv. 152):** Unorthodox low rating assigned by community server despite containing sustained 34 NPS chordstreams.
- **`POSSESSION` (Rated Lv. 137, Pred Lv. 95):** Extreme 320 BPM burst density in a single 4-second section; model averages out the lower overall song density.
- **`[LN] Epiik` (Rated Lv. 140, Pred Lv. 95):** 100% inverse long note chart where difficulty stems from release timing coordination rather than high key density.
