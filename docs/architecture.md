# Model Architecture & Design

## 1. Input Representation (60 Kinematic Channels)

Each beatmap is accepted as a discrete 2D time grid $(T, 7)$ sampled at **20 ms steps (50 FPS)**, where $T$ is the number of time frames. At each 20 ms step, every lane contains one of 5 categorical states:
- `0`: Empty (no note active)
- `1`: Tap Note (instantaneous strike)
- `2`: Long Note Head (start of hold)
- `3`: Long Note Body (ongoing hold)
- `4`: Long Note Tail (release)

To provide the model with explicit execution context while maintaining a compact `(B, T, 7)` uint8 input representation for WebAssembly inference, **60 continuous kinematic channels** are derived directly inside the PyTorch `forward()` pass:

```
Input Grid: (B, T, 7) uint8
  │
  ├─> 1. Categorical One-Hot: 7 lanes × 5 states = 35 channels
  ├─> 2. Macro Aggregate Channels:
  │       • taps_sum  (strikes at frame t: states 1 or 2) = 1 channel
  │       • holds_sum (active holds at frame t: states 2 or 3) = 1 channel
  ├─> 3. Per-Lane Continuous Kinematic Channels:
  │       • tension: (other lanes holding / 6) at event frames = 7 channels
  │       • jacks: hits × depthwise conv over previous 4 frames = 7 channels
  │       • rel_gaps: hits × release conv over previous 4 frames = 7 channels
  └─> 4. Hand-Split Roll Dynamics:
          • l_roll (lanes 1–3 hit count × 2-frame lookback conv) = 1 channel
          • r_roll (lanes 5–7 hit count × 2-frame lookback conv) = 1 channel
  │
  ▼
Expanded Feature Tensor: (B, 60, T)
```

The channels are concatenated in the following order: `one_hot` (35), `taps_sum` (1), `holds_sum` (1), `tension` (7), `jacks` (7), `rel_gaps` (7), `l_roll` (1), `r_roll` (1), totaling 60 channels:

1. **Categorical One-Hot (35 channels):** 7 lanes $\times$ 5 states.
2. **Chord Aggregates (2 channels):** `taps_sum` sums instantaneous strikes (states 1, 2) across all 7 lanes; `holds_sum` sums active holds (states 2, 3) across all 7 lanes.
3. **Pinned-Finger Tension (7 channels):** Tracks concurrent hold strain per lane:
   $$\text{tension}_i(t) = \frac{\text{holds\_sum}(t) - \text{holds}_i(t)}{6.0} \cdot \mathbb{I}(\text{lane}_i(t) \in \{1, 2, 4\})$$
   Evaluated strictly when lane $i$ undergoes an action (tap press, LN head press, or LN tail release).
4. **Same-Lane Jack Velocity (7 channels):** Computed per lane via depthwise 1D convolution over `hits` with a static 4-frame kernel $w = [0.25, 0.50, 0.75, 1.00]$ (most recent prior frame has weight 1.0), gated by hits at the current frame:
   $$\text{jacks}_i(t) = \text{hits}_i(t) \cdot (\text{padded\_hits}_i * w)(t)$$
   *Known limitation:* The 4-frame lookback window covers only 20–80 ms prior to the hit, which captures 1/4 jacks above ~188 BPM but does not cover slower jacks (100–250 ms).
5. **LN Release-to-Press Gaps (7 channels):** Applies the same 4-frame kernel $w$ to releases (state 4), gated by hits at the current frame:
   $$\text{rel\_gaps}_i(t) = \text{hits}_i(t) \cdot (\text{padded\_releases}_i * w)(t)$$
   Shares the same 20–80 ms window limitation as the jack channel.
6. **Hand-Split Roll Dynamics (2 channels):** Evaluates same-hand hit density for the left hand (lanes 1–3) and right hand (lanes 5–7), excluding center lane 4:
   $$l_{\text{roll}}(t) = l_{\text{hits}}(t) \cdot (\text{padded\_}l * [0.5, 1.0])(t)$$
   $$r_{\text{roll}}(t) = r_{\text{hits}}(t) \cdot (\text{padded\_}r * [0.5, 1.0])(t)$$
   This produces a 40 ms roll density per hand rather than a spatial coordinate delta.

---

## 2. Receptive Field & Dilated Residual Backbone

Difficulty in 7K rhythm games depends on both micro-patterns (trills, jacks, rolls) and macro-stamina (sustained high-NPS streams).

To capture temporal structure without downsampling, the network uses a 1D convolution stem followed by **6 Dilated Residual 1D Convolutional Blocks**:
- **Stem:** `Conv1D(60, 128, kernel_size=1, bias=False)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `GELU`.
- **Residual Blocks:** Each block contains two dilated conv layers with kernel size $K=5$, padding $2d$, batch normalization, and GELU activations. GELU is applied after the residual addition; no dropout is used inside the residual blocks:
  $$\text{Conv1D}(K=5, d) \rightarrow \text{BatchNorm1d} \rightarrow \text{GELU} \rightarrow \text{Conv1D}(K=5, d) \rightarrow \text{BatchNorm1d} + \text{Residual} \rightarrow \text{GELU}$$

Each convolution with kernel size $K=5$ and dilation $d$ expands the receptive field by $(K-1) \cdot d = 4d$ frames. With two convolutions per block, each ResBlock adds $8d$ frames:

| Block | Kernel Size ($K$) | Dilation ($d$) | Convs / Block | Receptive Field Growth | Cumulative Context Window |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Stem** | 1 | 1 | 1 | 1 frame | 20 ms |
| **ResBlock 1** | 5 | 1 | 2 | $2 \times 4 \times 1 = 8$ frames | 9 frames (180 ms) |
| **ResBlock 2** | 5 | 2 | 2 | $2 \times 4 \times 2 = 16$ frames | 25 frames (500 ms) |
| **ResBlock 3** | 5 | 4 | 2 | $2 \times 4 \times 4 = 32$ frames | 57 frames (1,140 ms / 1.14 s) |
| **ResBlock 4** | 5 | 8 | 2 | $2 \times 4 \times 8 = 64$ frames | 121 frames (2,420 ms / 2.42 s) |
| **ResBlock 5** | 5 | 16 | 2 | $2 \times 4 \times 16 = 128$ frames | 249 frames (4,980 ms / 4.98 s) |
| **ResBlock 6** | 5 | 32 | 2 | $2 \times 4 \times 32 = 256$ frames | **505 frames (10,100 ms ≈ 10.1 s)** |

Total receptive field per frame is **505 frames $\approx 10.1$ seconds** ($1 + 8 \times 63 = 505$ frames at 20 ms/frame).

---

## 3. Multi-Scale Pooling & Peak-Relative Active Gating

Difficulty cannot be evaluated purely by local density spikes or whole-song global averages. The pooling stage extracts temporal features across multiple scales while suppressing isolated gimmicks and quiet sections:

### 1. Multi-Scale Windowed Averages:
Temporal average pooling is evaluated across three window sizes over masked features (`feat_masked = feat * mask`), normalized by the pooled mask coverage (`m = avg_pool1d(mask)`), and masked with `-1e4` where `m < 1e-3` prior to `.amax(dim=-1)` so negative `GELU` activations in valid frames are never overwritten by zero-padded windows:
- `p2`: kernel 100 frames (2.0s), stride 25 frames (0.5s), padding 50 frames (1.0s), masked max over time $\rightarrow (B, 128)$.
- `p5`: kernel 250 frames (5.0s), stride 25 frames (0.5s), padding 125 frames (2.5s), masked max over time $\rightarrow (B, 128)$.
- `p15`: kernel 750 frames (15.0s), stride 50 frames (1.0s), padding 375 frames (7.5s), masked max over time $\rightarrow (B, 128)$.

Burst excess is damped using the surrounding 5s phrase context:
$$\text{excess} = \text{ReLU}(p_2 - p_5)$$
$$p_{2\text{,damped}} = p_5 + \text{excess} \cdot \tanh\left(\frac{|p_5|}{\text{excess} + 0.1}\right)$$

### 2. Peak-Relative Active Gating:
Instantaneous strain $s(t)$ is projected through a 1D convolutional head:
$$s_{\text{raw}} = \text{Softplus}(\text{Conv1D}_{1\times 1}(\text{GELU}(\text{Conv1D}_{1\times 1}(\text{feat})))) \cdot \text{mask}$$
with the output layer bias initialized to $-2.0$. Letting $s_{\text{peak}} = \max_t(s_{\text{raw}})$, frames are gated relative to peak intensity:
$$g(t) = \sigma\left(\frac{s_{\text{raw}}(t) - 0.65 \cdot s_{\text{peak}}}{0.10 \cdot s_{\text{peak}} + 10^{-5}}\right) \cdot \text{mask}(t)$$
From this gate, three summary representations are derived:
- **Active Feature Average (`active_gap`):** $\frac{\sum_t \text{feat}(t) \cdot g(t)}{\sum_t g(t)} \rightarrow (B, 128)$, averaging features over windows exceeding 65% of peak intensity.
- **Active Strain Mean ($s_{\text{active\_mean}}$):** $\frac{\sum_t s_{\text{raw}}(t) \cdot g(t)}{\sum_t g(t)}$ scalar.
- **Stamina Duration Volume (`stamina_vol`):** $\log(1 + T_{\text{active}} \times 0.02)$ scalar, measuring active climax duration in log-seconds.

### 3. Masked Attention Pooling:
Global attention pooling computes learned softmax weights via a `Conv1D(128, 1, kernel_size=1)` scoring layer over masked frames $\rightarrow$ `attn_pool` $(B, 128)$.

### 4. Pooled Representation:
The MLP head receives a concatenated vector of **5 channel vectors (128 dims each) + 3 summary scalars** (total dimension = $5 \times 128 + 3 = 643$):
$$\text{pooled} = [p_{2\text{,damped}}, p_5, p_{15}, \text{active\_gap}, \text{attn\_pool}, s_{\max} \cdot 0.01, s_{\text{active\_mean}} \cdot 0.01, \text{stamina\_vol}]$$

---

## 4. Linear-Core Double-Softplus Regression Head

The regression head consists of a 3-layer MLP:
$$\text{Linear}(643, 256) \rightarrow \text{GELU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(256, 128) \rightarrow \text{GELU} \rightarrow \text{Dropout}(0.1) \rightarrow \text{Linear}(128, 1)$$

To map the unconstrained scalar $\text{raw}$ to the valid level range $[1.0, 195.0]$, the model uses a centered Double-Softplus transform:
$$z = 75.0 + 30.0 \cdot \text{raw}$$
$$\hat{y} = L_{\min} + \tau_{\text{low}} \cdot \text{Softplus}\left(\frac{z - L_{\min}}{\tau_{\text{low}}}\right) - \tau_{\text{high}} \cdot \text{Softplus}\left(\frac{z - L_{\max}}{\tau_{\text{high}}}\right)$$
where $L_{\min} = 1.0, \tau_{\text{low}} = 5.0, L_{\max} = 195.0, \tau_{\text{high}} = 20.0$.

- **Linear Core:** The derivative $\frac{d\hat{y}}{dz} \approx 1.0$ across the primary difficulty spectrum from Lv. 15 to Lv. 140, avoiding inflection-point compression.
- **Asymptotic Bounds:** Softplus transitions provide a smooth floor near $L_{\min} = 1.0$ and smooth ceiling compression toward $L_{\max} = 195.0$.

---

## 5. Loss Function (`compute_loss`)

The training objective optimizes a composite loss combining point prediction, pairwise slope preservation, and strain calibration:
$$\mathcal{L} = \mathcal{L}_{\text{point}} + 0.25 \cdot \mathcal{L}_{\text{rank}} + 0.10 \cdot \mathcal{L}_{\text{strain}}$$

1. **Targeted Asymmetric Point Loss ($\mathcal{L}_{\text{point}}$):**
   Base loss is Smooth L1 ($\beta = 1.0$) between prediction $\hat{y}$ and label $y$. An asymmetric penalty multiplier $w_{\text{asym}}$ is applied when under-predicting ($\hat{y} < y$):
   $$w_{\text{asym}} = 1.0 + \mathbb{I}(\hat{y} < y) \cdot (0.40 \cdot f_{\text{mid}} + 0.35 \cdot f_{\text{marathon}})$$
   where:
   - $f_{\text{mid}} = \exp\left(-0.5 \left(\frac{y - 82.0}{9.0}\right)^2\right)$ targets under-predictions around Lv. 70–92 (Gaussian centered at Lv. 82, $\sigma = 9$).
   - $f_{\text{marathon}} = \sigma\left(\frac{T_{\text{sec}} - 150.0}{30.0}\right) \cdot \sigma\left(\frac{105.0 - y}{5.0}\right)$ penalizes stamina under-predictions on marathon charts (>150s) at Lv. $\le 105$.
2. **Pairwise Slope Preservation Loss ($\mathcal{L}_{\text{rank}}$, weight 0.25):**
   Evaluates Smooth L1 ($\beta = 1.0$) between predicted pairwise differences $(\hat{y}_i - \hat{y}_j)$ and target differences $(y_i - y_j)$ for all pairs in the batch whose level difference satisfies $5.0 \le |y_i - y_j| \le 45.0$. This penalizes non-linear scaling across difficulty brackets.
3. **Auxiliary Strain Calibration Loss ($\mathcal{L}_{\text{strain}}$, weight 0.10):**
   Anchors the HUD strain curve to ground-truth difficulty units:
   $$s_{\text{eff}} = 0.70 \cdot s_{\text{peak}} + 0.30 \cdot s_{\text{mean}}$$
   $$\mathcal{L}_{\text{strain}} = \text{SmoothL1}(s_{\text{eff}}, y; \beta=2.0)$$

---

## 6. Dual ONNX Model Outputs

The exported ONNX model (`model.onnx`, 4.60 MB) provides two simultaneous outputs:
1. `predicted_level` (`shape: [batch_size]`): Single scalar predicted O2Jam level.
2. `strain_curve` (`shape: [batch_size, timeline_steps]`): Instantaneous difficulty strain curve across song duration (1 sample per second via `avg_pool1d(kernel_size=50, stride=50, ceil_mode=True)`), calibrated to true level units for HUD overlay visualization.
