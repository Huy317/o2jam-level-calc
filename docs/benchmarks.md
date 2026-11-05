# Benchmarks & Validation Results

## 1. Overall Performance Across Unified Dataset (4,423 Clean Charts)

Evaluated across all 4,423 clean, unique charts (with dummy `level <= 3` custom gimmick charts filtered out) using the exported ONNX model (`model.onnx`):

| Metric | Model v1 (Baseline) | Model v3 (Unbounded) | Model v4 (Sigmoid) | Model v5 (Linear-Core & Gating) | Change / Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | 5.75 levels | 7.45 levels | 5.34 levels | **5.26 levels** | **Best overall accuracy achieved** |
| **Root Mean Squared Error (RMSE)**| 8.59 levels | 11.23 levels | 7.85 levels | **7.48 levels** | **Lowest overall variance in project history** |
| **Pearson Correlation ($R$)** | 0.967 | 0.9715 | 0.9785 | **0.9785** | High linear correlation across all tiers |
| **Median Absolute Error (P50)** | $\le 3.88$ levels | $\le 4.84$ levels | $\le 3.67$ levels | **$\le 3.55$ levels** | Over 50% of charts within 3.6 levels |
| **75th Percentile (P75)** | $\le 7.44$ levels | $\le 9.46$ levels | $\le 6.88$ levels | **$\le 6.64$ levels** | 75% of charts within 6.6 levels |
| **90th Percentile (P90)** | $\le 13.25$ levels| $\le 16.98$ levels| $\le 11.71$ levels| **$\le 11.45$ levels**| Top 90% well within standard Dan bracket |
| **High-End Ceiling** | Capped at $\sim$Lv. 150 | Uncapped (Runaway > 250) | Bounded [1.0, 185.0] | **Bounded [1.0, 195.0]** | Zero curvature sag; safe asymptotic limit |

---

## 2. Accuracy Breakdown by Level Tier (4,423 Charts)

| Level Tier | Charts | Model v4 MAE | Model v4 Bias | Model v5 MAE | Model v5 RMSE | Model v5 Bias | Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lv. 20 – 45** | 540 | 3.45 | +0.74 | **3.48** | 4.82 | +0.52 | High precision on entry/beginner tier |
| **Lv. 45 – 70** | 896 | 4.32 | +1.02 | **4.29** | 5.82 | -0.15 | Over-rating eliminated in core tier |
| **Lv. 70 – 95** | 1,678 | 4.08 | -1.27 | **3.82** | 5.46 | -0.22 | **Under-rating eliminated (-1.27 -> -0.22)** |
| **Lv. 95 – 120** | 430 | 9.07 | +1.11 | **8.65** | 11.62 | +0.94 | MAE improved by 0.4 levels |
| **Lv. 120 – 180** | 745 | 9.14 | -0.17 | **9.18** | 12.01 | -0.19 | Balanced zero-bias across elite Dan tier |

### Key Win in Core Tier (Lv. 75 – 85):
- In Model v4, **77.9% of charts in Lv. 80–85 were under-rated** with a mean bias of $-2.39$.
- In Model v5, the Double-Softplus linear core and sustain-damped pooling cut the negative bias down to $-1.04$, while Lv. 75–80 achieved a balanced $50.1\%$ split with **$+0.30$ mean bias** and **$2.67$ MAE**.

---

## 3. Accuracy Breakdown by Song Duration (Stamina)

| Duration Category | Time Range | Charts | Model v4 Bias | Model v5 Bias | Model v5 MAE | Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Short** | $< 90\text{s}$ | 160 | +4.43 | **+3.35** | 5.75 | Over-rating reduced by > 1.0 level |
| **Medium** | $90\text{s} - 150\text{s}$ | 2,080 | +1.78 | **+1.64** | 5.61 | Consistent across standard song lengths |
| **Long** | $150\text{s} - 240\text{s}$ | 1,595 | -1.63 | **-1.58** | 4.77 | Lowest MAE across length brackets |
| **Marathon** | $> 240\text{s}$ | 588 | -3.42 | **-3.10** | 5.23 | Active gating protects climaxes |

---

## 4. Notable Outliers & Case Studies

1. **`POSSESSION` (Rated Lv. 137, 320 BPM burst):**
   - Accurately captures high density without inflating into theoretical runaway territory.
2. **`Almagest` (Rated Lv. 173):**
   - Cleanly seated in the peak Lv. 160–175 tier.
3. **`sky_delta - Justice` (Rated Lv. 84):**
   - Accurate within standard human rating variance.
4. **Pure LN Charts (`Mr. Chu [LN]`, `Quaver [LN]`):**
   - Extreme pure-LN charts (95%+ long notes) still exhibit mild negative bias (~1.5–1.8 levels) because human ratings reward finger release coordination and independence beyond raw note density.
