# Model Architecture & Design

## 1. Input Representation (37 Channels)

Each beatmap is represented as a discrete 2D time grid $(T, 7)$ sampled at **$20\text{ms}$ steps ($50\text{ FPS}$)**, where $T$ is the number of time frames. At each 20ms step, every lane contains one of 5 categorical states:
- `0`: Empty (no note active)
- `1`: Tap Note (instantaneous strike)
- `2`: Long Note Head (start of hold)
- `3`: Long Note Body (ongoing hold)
- `4`: Long Note Tail (release)

```
(T, 7) Grid (Values 0..4)
  │
  ├─> One-Hot Encoding: 7 lanes × 5 states = 35 channels
  ├─> Channel 36: Attack Count (Number of new notes struck at frame t: states 1 or 2)
  └─> Channel 37: Hold Count   (Number of active holds at frame t: states 2 or 3)
  │
  ▼
Input Tensor: (B, 37, T)
```

The explicit **Attack Count** and **Hold Count** channels allow early convolutional layers to immediately isolate chord densities (single, double, quad, full-chord) and hybrid LN complexity without requiring complex multi-lane cross-channel correlations.

---

## 2. Receptive Field & Dilated Residual Backbone

Difficulty in 7K rhythm games depends on both **micro-patterns** (trills, jacks, rolling staircases spanning 100ms - 500ms) and **macro-stamina** (dense 30 NPS chordstreams lasting 5 - 15 seconds).

To capture both without downsampling time resolution, the network utilizes **6 Dilated Residual 1D Convolutional Blocks** with exponentially increasing dilation factors:

| Block | Kernel Size ($K$) | Dilation ($d$) | Stride | Receptive Field Growth | Cumulative Context Window |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Stem** | 5 | 1 | 1 | 4 frames | 80 ms |
| **ResBlock 1** | 3 | 1 | 1 | $2 \times (3-1) \times 1 = 4$ frames | 160 ms |
| **ResBlock 2** | 3 | 2 | 1 | $2 \times (3-1) \times 2 = 8$ frames | 320 ms |
| **ResBlock 3** | 3 | 4 | 1 | $2 \times (3-1) \times 4 = 16$ frames | 640 ms |
| **ResBlock 4** | 3 | 8 | 1 | $2 \times (3-1) \times 8 = 32$ frames | 1,280 ms (1.28s) |
| **ResBlock 5** | 3 | 16 | 1 | $2 \times (3-1) \times 16 = 64$ frames | 2,560 ms (2.56s) |
| **ResBlock 6** | 3 | 32 | 1 | $2 \times (3-1) \times 32 = 128$ frames | **5,120 ms (5.12s)** |

Each block consists of:
$$\text{Conv1D}(d) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.1) \rightarrow \text{Conv1D}(d) \rightarrow \text{BatchNorm1d} + \text{Residual}$$

Total receptive field per frame is **253 frames $\approx 5.06$ seconds**, enough to recognize full musical phrases, chordstream transitions, and sustained anchor patterns.

---

## 3. Multi-Scale Temporal Pooling

Rhythm game charts cannot be judged purely by peak density or global averages:
- A 3-minute chart with one 3-second Lv. 150 stream is not a Lv. 150 chart, nor is it a Lv. 50 chart.
- Human level ratings balance **sustained density** (stamina) with **peak bursts** (technical difficulty).

The architecture computes **5 complementary temporal poolings** over the sequence $(B, C, T)$:

```
Feature Sequence (B, 128, T)
  ├── 1. Global Average Pooling (GAP)        --> (B, 128)  [Overall song stamina]
  ├── 2. 1-Second Max-Pool (stride 250ms)    --> (B, 128)  [Micro-burst / spike intensity]
  ├── 3. 5-Second Max-Pool (stride 1.25s)    --> (B, 128)  [Phrase-level sustained density]
  ├── 4. 15-Second Max-Pool (stride 3.75s)   --> (B, 128)  [Climax / drop section difficulty]
  └── 5. Temporal Attention Pooling          --> (B, 128)  [Learnable importance weighting]
                                                  │
                                                  ▼ Concatenation
                                             (B, 640)
```

### Temporal Attention Formulation:
For sequence feature $h_t \in \mathbb{R}^{128}$:
$$\alpha_t = \frac{\exp(W_a h_t)}{\sum_{\tau=1}^T \exp(W_a h_\tau)}, \quad h_{\text{att}} = \sum_{t=1}^T \alpha_t h_t$$

---

## 4. Sigmoid Bounded Head

O2Jam difficulty levels cannot be negative, nor can they realistically exceed 185. Standard unconstrained linear heads risk predicting wild extrapolations ($< 0$ or $> 250$) on abnormal gimmick charts.

The regression head uses a smooth, bounded Sigmoid mapping:
$$\hat{y} = L_{\min} + (L_{\max} - L_{\min}) \cdot \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot h_{\text{pooled}} + b_1) + b_2)$$
where $L_{\min} = 15.0$ and $L_{\max} = 185.0$.

---

## 5. Loss Function: Tier-Weighted Asymmetric Huber Loss

1. **Huber Loss ($\delta = 3.0$):** Linear penalties on large outliers prevent rare mislabeled community charts from pulling model gradients, while quadratic penalties near zero ensure precise sub-level convergence.
2. **Tier Weighting ($w_{\text{tier}}$):** High-tier charts (Lv. 110–185) are naturally rarer than mid-tier charts (Lv. 50–80). High-tier samples receive up to $1.8\times$ weight to ensure the network does not underfit difficult charts.
3. **Asymmetric Penalty ($1.15\times$ for under-predictions on Lv. 80+):** Players are significantly more sensitive to hard charts being under-rated (which makes a map feel unexpectedly crushing) than slight over-ratings.
