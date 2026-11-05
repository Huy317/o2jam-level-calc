# Model Architecture & Design

## 1. Input Representation (60 Kinematic Channels)

Each beatmap is accepted as a discrete 2D time grid $(T, 7)$ sampled at **$20\text{ms}$ steps ($50\text{ FPS}$)**, where $T$ is the number of time frames. At each 20ms step, every lane contains one of 5 categorical states:
- `0`: Empty (no note active)
- `1`: Tap Note (instantaneous strike)
- `2`: Long Note Head (start of hold)
- `3`: Long Note Body (ongoing hold)
- `4`: Long Note Tail (release)

To provide the model with explicit biomechanical and physical execution awareness while maintaining a compact `(B, T, 7)` uint8 input representation for WebAssembly inference, **60 continuous kinematic channels** are derived directly inside the PyTorch `forward()` pass:

```
Input Grid: (B, T, 7) uint8
  │
  ├─> 1. Categorical One-Hot: 7 lanes × 5 states = 35 channels
  ├─> 2. Macro Aggregate Channels:
  │       • Attack Count (strikes at frame t: states 1 or 2) = 1 channel
  │       • Hold Count   (active holds at frame t: states 2 or 3) = 1 channel
  ├─> 3. Per-Lane Continuous Kinematic Channels:
  │       • Jack Velocity: v_jack = min(1.0, 50ms / Δt_press) = 7 channels
  │       • LN Release-to-Press Gap Velocity: v_gap = min(1.0, 50ms / Δt_gap) = 7 channels
  │       • Pinned-Finger Tension: concurrent hold duration = 7 channels
  └─> 4. Hand-Split Alternation Dynamics:
          • Left Hand Delta (lanes 1–3) = 1 channel
          • Right Hand Delta (lanes 5–7) = 1 channel
  │
  ▼
Expanded Feature Tensor: (B, 60, T)
```

The continuous kinematic channels differentiate human mechanical strain:
- A 320 BPM burst (40ms interval) produces high-velocity jack/roll values distinct from standard 160 BPM streams.
- Pinned fingers during complex hold patterns are tracked explicitly, separating single-finger strain from free-hand multi-finger streams.

---

## 2. Receptive Field & Dilated Residual Backbone

Difficulty in 7K rhythm games depends on both **micro-patterns** (trills, jacks, rolling staircases spanning 100ms - 500ms) and **macro-stamina** (dense 30 NPS chordstreams lasting 5 - 15 seconds).

To capture both without temporal downsampling, the network utilizes **6 Dilated Residual 1D Convolutional Blocks** with exponentially increasing dilation factors:

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

Total receptive field per frame is **253 frames $\approx 5.06$ seconds**, encompassing full musical phrases, chordstream transitions, and sustained anchor patterns.

---

## 3. Sustain-Damped Burst Pooling & Peak-Relative Active Gating

Rhythm game charts cannot be judged purely by peak density or global averages:
- A 4-minute chart with intense Lv. 80 chordstreams should not have its rating watered down by 2 minutes of calm vocals or intros.
- Conversely, an isolated 1-second mash gimmick should not trigger runaway difficulty inflation.

To solve both challenges simultaneously, the pooling stage combines three mechanisms:

### 1. Sustain-Damped 2-Second Burst (`p2_damped`):
Multi-scale temporal pooling extracts 2.0s (`p2`), 5.0s (`p5`), and 15.0s (`p15`) windowed peaks. The 2s burst excess is gated by its surrounding 5s phrase context:
$$\text{excess} = \text{ReLU}(p_2 - p_5)$$
$$p_{2\text{,damped}} = p_5 + \text{excess} \cdot \tanh\left(\frac{|p_5|}{\text{excess} + 0.1}\right)$$
In legitimate high-difficulty charts, a 2s burst sits inside an active musical phrase, allowing the burst to pass through. In isolated gimmick mashes surrounded by silence, $p_5 \approx 0$ damps the spike completely.

### 2. Peak-Relative Active Gating (`active_gap` & `stamina_vol`):
Instead of whole-song global average pooling (which dilutes long marathon charts), frames are gated relative to the chart's own peak strain $s_{\text{peak}}$:
$$g(t) = \sigma\left(\frac{s(t) - 0.65 \cdot s_{\text{peak}}}{0.10 \cdot s_{\text{peak}} + \epsilon}\right) \cdot \text{mask}(t)$$
- **Active Feature Average (`active_gap`):** Averages features strictly across windows exceeding 65% of peak intensity, making technical difficulty immune to long quiet intros/outros.
- **Stamina Duration Volume (`stamina_vol`):** Computes $\log(1 + T_{\text{active}})$, providing a monotonic measure of sustained climax duration in seconds.

### 3. Masked Attention Pooling:
Global attention pooling assigns learned softmax weights across all valid frames without padding leakage.

---

## 4. Linear-Core Double-Softplus Regression Head

Early architectures used either a Sigmoid head (which suffered from second-order curvature sagging in the dense Lv. 75–90 bracket) or an unbounded Softplus head (which caused runaway extrapolation above Lv. 180).

Model v5 utilizes a **Centered Double-Softplus formulation**:
$$z = 76.5 + 30.0 \cdot \text{raw}$$
$$\hat{y} = L_{\min} + \tau_{\text{low}} \cdot \text{Softplus}\left(\frac{z - L_{\min}}{\tau_{\text{low}}}\right) - \tau_{\text{high}} \cdot \text{Softplus}\left(\frac{z - L_{\max}}{\tau_{\text{high}}}\right)$$
where $L_{\min} = 1.0, \tau_{\text{low}} = 5.0, L_{\max} = 195.0, \tau_{\text{high}} = 20.0$.

### Why This Works:
1. **Zero S-Curve Sag:** The derivative $\frac{d\hat{y}}{dz} \approx 1.0$ across the entire human spectrum from **Lv. 15 to Lv. 140**, completely eliminating inflection-point distortion.
2. **Smooth Asymptotic Ceiling:** Between Lv. 145 and Lv. 195, the upper softplus smoothly compresses predictions toward $L_{\max} = 195.0$, making runaway explosions mathematically impossible.

---

## 5. Dual ONNX Model Outputs

The exported ONNX model (`model.onnx`, 4.60 MB) provides two simultaneous outputs:
1. `predicted_level` (`shape: [batch_size]`): Single scalar predicted O2Jam level.
2. `strain_curve` (`shape: [batch_size, timeline_steps]`): Instantaneous difficulty strain curve across song duration (1 sample per second), calibrated to true level units and consumed directly by overlay clients.
