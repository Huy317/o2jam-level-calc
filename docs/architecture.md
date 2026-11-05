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

## 3. Instantaneous Strain Curve Projection & Multi-Slice Decay Pooling

Rhythm game charts cannot be judged purely by peak density or global averages:
- A 3-minute chart with one 3-second Lv. 140 burst should not be diluted by calm intro/outro sections.
- Human level ratings balance **sustained density** (stamina) with **peak bursts** (technical difficulty).

### Instantaneous Strain Curve:
Latent features are downsampled via 50-frame temporal max-pooling (1-second steps) and projected through a linear layer with ReLU activation:
$$s(t) = \text{ReLU}(W_{\text{proj}} \cdot x_t + b_{\text{proj}}) \ge 0$$
This generates the 1D difficulty timeline curve outputted directly by the ONNX model for real-time visualization.

### Multi-Slice Sorted Exponential Decay Pooling:
To capture both short burst spikes and long-term stamina without dilution:
1. Slices of the top 0.5s, 1.5s, 5s, and 15s highest difficulty points are extracted from $s(t)$.
2. Exponential decay weighting ($\gamma$) aggregates sorted feature vectors, ensuring peak spikes dominate the pooled representation:
$$P(\gamma) = \sum_{k=1}^K \gamma^{k-1} \cdot h_{(k)}$$
3. A global average pooling (GAP) slice provides overall stamina context.

---

## 4. Uncapped Softplus Regression Head

Early architectures utilized a bounded Sigmoid head ($[15.0, 185.0]$) which suffered from gradient saturation above Lv. 130 and created an artificial prediction ceiling.

The regression head utilizes an **Uncapped Softplus formulation**:
$$\hat{y} = L_{\min} + \text{Softplus}(W_2 \cdot \text{GELU}(W_1 \cdot h_{\text{pooled}} + b_1) + b_2)$$
where $L_{\min} = 15.0$.

This guarantees non-negative, smooth gradient propagation while allowing natural extrapolation to ultra-high difficulty charts (Lv. 185 to Lv. 220+) without ceiling compression.

---

## 5. Dual ONNX Model Outputs

The exported ONNX model (`model.onnx`) provides two simultaneous outputs:
1. `predicted_level` (`shape: [batch_size]`): Single scalar predicted O2Jam level.
2. `strain_curve` (`shape: [batch_size, timeline_steps]`): Instantaneous difficulty strain curve across song duration (1 sample per second), consumed directly by the Tosu HUD overlay to render the in-game difficulty timeline sparkline.
