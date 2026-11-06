"""
Step 3: Train O2Jam 7K Level Predictor on Unified Dataset (4,423 charts).
Model v5 Architecture:
  - 60 input channels (35 one-hot states + 2 counts + 23 kinematic channels)
  - 6 Residual Conv1D blocks (dilations 1, 2, 4, 8, 16, 32) -> ~10.1s receptive field
  - Sustain-damped 2s burst pooling + 5s phrase + 15s climax
  - Peak-relative active gating & stamina duration volume (eliminates marathon dilution)
  - Linear-core Double-Softplus bounded output [1.0, 195.0] (eliminates sigmoid curvature sag)
  - Pairwise slope preservation + mid-high/marathon targeted Huber loss + HUD strain calibration
"""

import json
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(REPO_ROOT, "data", "o2jam_dataset.npz")
METADATA_PATH = os.path.join(REPO_ROOT, "data", "dataset_metadata.json")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "best_model.pt")
ONNX_PATH = os.path.join(REPO_ROOT, "model.onnx")
EXTENSION_ONNX = os.path.join(REPO_ROOT, "extension", "o2jam_level_predictor", "model.onnx")

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class O2JamDataset(Dataset):
    def __init__(self, flat_grids, offsets, lengths, levels, indices, augment=False, priorities=None):
        self.flat_grids = flat_grids
        self.offsets = offsets
        self.lengths = lengths
        self.levels = levels
        self.indices = indices
        self.augment = augment
        self.priorities = priorities

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chart_idx = self.indices[idx]
        start = self.offsets[chart_idx]
        end = self.offsets[chart_idx + 1]
        t_len = self.lengths[chart_idx]
        grid = self.flat_grids[start:end].reshape(t_len, 7)

        # Horizontal mirror augmentation (col -> 6 - col) with 50% probability
        if self.augment and random.random() < 0.5:
            grid = np.ascontiguousarray(grid[:, ::-1])

        level = self.levels[chart_idx]
        if self.priorities is not None:
            return torch.from_numpy(grid), torch.tensor(level, dtype=torch.float32), torch.tensor(self.priorities[chart_idx], dtype=torch.long)
        return torch.from_numpy(grid), torch.tensor(level, dtype=torch.float32)


class LengthBucketSampler(torch.utils.data.Sampler):
    def __init__(self, chart_lengths, batch_size=BATCH_SIZE, shuffle=True):
        self.indices = sorted(range(len(chart_lengths)), key=lambda i: chart_lengths[i])
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    has_prio = len(batch[0]) == 3
    if has_prio:
        grids, levels, priorities = zip(*batch)
    else:
        grids, levels = zip(*batch)
    lengths = [g.shape[0] for g in grids]
    max_len = max(lengths)

    batch_size = len(grids)
    padded = torch.zeros((batch_size, max_len, 7), dtype=torch.uint8)
    mask = torch.zeros((batch_size, 1, max_len), dtype=torch.float32)
    for i, g in enumerate(grids):
        padded[i, :g.shape[0], :] = g
        mask[i, 0, :g.shape[0]] = 1.0

    levels_tensor = torch.stack(levels)
    if has_prio:
        return padded, levels_tensor, mask, torch.stack(priorities)
    return padded, levels_tensor, mask


class ServerCalibrationAdapter(nn.Module):
    """
    Train-only per-server calibration head:
    Maps canonical Private prediction `pred` -> server-specific label scale `pred_server`
    using a 4-knot piecewise linear offset for each non-Private server (priority > 0).
    Priority 0 (Private) is strictly fixed at 0 offset (identity).
    Discarded at inference / ONNX export so model.onnx predicts strictly on the canonical scale.
    """
    def __init__(self, num_sources=4):
        super().__init__()
        self.register_buffer("knots", torch.tensor([35.0, 62.5, 85.0, 115.0], dtype=torch.float32))
        init_offsets = torch.zeros((num_sources, 4), dtype=torch.float32)
        if num_sources > 3:
            init_offsets[3] = torch.tensor([-4.0, -1.5, -0.15, -2.5], dtype=torch.float32)
        self.server_offsets = nn.Parameter(init_offsets)

    def forward(self, pred, priorities):
        x = pred.unsqueeze(-1)
        k = self.knots
        b0 = ((k[1] - x) / (k[1] - k[0])).clamp(0.0, 1.0)
        b1 = torch.minimum(((x - k[0]) / (k[1] - k[0])).clamp(0.0, 1.0), ((k[2] - x) / (k[2] - k[1])).clamp(0.0, 1.0))
        b2 = torch.minimum(((x - k[1]) / (k[2] - k[1])).clamp(0.0, 1.0), ((k[3] - x) / (k[3] - k[2])).clamp(0.0, 1.0))
        b3 = ((x - k[2]) / (k[3] - k[2])).clamp(0.0, 1.0)
        basis = torch.cat([b0, b1, b2, b3], dim=-1)
        offsets_per_sample = self.server_offsets[priorities]
        is_non_private = (priorities > 0).float().unsqueeze(-1)
        return pred + (basis * offsets_per_sample * is_non_private).sum(dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class O2JamLevelPredictor(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        # Total input channels = 35 (onehot) + 1 (taps) + 1 (holds) + 7 (tension) + 7 (jacks) + 7 (rel_gaps) + 2 (hand rolls) = 60
        self.stem = nn.Sequential(
            nn.Conv1d(60, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # 6 Dilated Residual blocks covering receptive field up to ~10.1 seconds
        self.block1 = ResidualBlock1D(channels, kernel_size=5, dilation=1)   # rf: 9 (180ms)
        self.block2 = ResidualBlock1D(channels, kernel_size=5, dilation=2)   # rf: 25 (500ms)
        self.block3 = ResidualBlock1D(channels, kernel_size=5, dilation=4)   # rf: 57 (1.14s)
        self.block4 = ResidualBlock1D(channels, kernel_size=5, dilation=8)   # rf: 121 (2.42s)
        self.block5 = ResidualBlock1D(channels, kernel_size=5, dilation=16)  # rf: 249 (4.98s)
        self.block6 = ResidualBlock1D(channels, kernel_size=5, dilation=32)  # rf: 505 (10.10s)

        # Attention pooling scoring layer
        self.attn_conv = nn.Conv1d(channels, 1, kernel_size=1)

        # Instantaneous 1D strain projection layer
        self.strain_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        nn.init.constant_(self.strain_conv[2].bias, -2.0)

        # Head: 5 latent channel vectors (p2_damped, p5, p15, active_gap, attn_pool) + 3 summary scalars
        self.head = nn.Sequential(
            nn.Linear(channels * 5 + 3, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        # Static kernels for depthwise kinematic convolutions
        k_weights = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32).view(1, 1, 4).repeat(7, 1, 1)
        self.register_buffer('jack_kernel', k_weights)
        k_hand = torch.tensor([0.5, 1.0], dtype=torch.float32).view(1, 1, 2)
        self.register_buffer('hand_kernel', k_hand)

    def forward(self, x, mask=None):
        # x: (Batch, T, 7) uint8 categorical integers [0..4]
        b, t, _ = x.shape
        if mask is None:
            mask = torch.ones((b, 1, t), device=x.device, dtype=torch.float32)

        # 1. Categorical one-hot: (B, 35, T)
        one_hot = F.one_hot(x.long(), num_classes=5).float().view(b, t, 35).permute(0, 2, 1)

        # 2. Base chord hits & holds
        hits = ((x == 1) | (x == 2)).float().permute(0, 2, 1)   # (B, 7, T)
        holds = ((x == 2) | (x == 3)).float().permute(0, 2, 1)  # (B, 7, T)
        releases = (x == 4).float().permute(0, 2, 1)            # (B, 7, T)

        taps_sum = hits.sum(dim=1, keepdim=True)   # (B, 1, T)
        holds_sum = holds.sum(dim=1, keepdim=True) # (B, 1, T)

        # 3. Kinematics: Pinned finger tension per lane (B, 7, T)
        other_holds = holds_sum - holds
        active_events = ((x == 1) | (x == 2) | (x == 4)).float().permute(0, 2, 1)
        tension = (other_holds / 6.0) * active_events

        # 4. Kinematics: Same-lane jack velocity per lane (B, 7, T)
        padded_hits = F.pad(hits, (4, 0))
        conv_j = F.conv1d(padded_hits, self.jack_kernel, groups=7)[:, :, :t]
        jacks = hits * conv_j

        # 5. Kinematics: LN release-to-press gaps per lane (B, 7, T)
        padded_rel = F.pad(releases, (4, 0))
        conv_rel = F.conv1d(padded_rel, self.jack_kernel, groups=7)[:, :, :t]
        rel_gaps = hits * conv_rel

        # 6. Kinematics: Same-hand roll velocity (Left: 0-2, Right: 4-6)
        l_hits = hits[:, 0:3, :].sum(dim=1, keepdim=True)
        r_hits = hits[:, 4:7, :].sum(dim=1, keepdim=True)
        padded_l = F.pad(l_hits, (2, 0))
        padded_r = F.pad(r_hits, (2, 0))
        l_roll = l_hits * F.conv1d(padded_l, self.hand_kernel)[:, :, :t]
        r_roll = r_hits * F.conv1d(padded_r, self.hand_kernel)[:, :, :t]

        # Combine all 60 physical channels
        inp = torch.cat([one_hot, taps_sum, holds_sum, tension, jacks, rel_gaps, l_roll, r_roll], dim=1) # (B, 60, T)

        feat = self.stem(inp)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        feat = self.block5(feat)
        feat = self.block6(feat)

        # Masked multi-scale phrase pooling (mask padded windows before amax so negative GELU activations aren't clamped to 0)
        feat_masked = feat * mask
        p2_raw = F.avg_pool1d(feat_masked, kernel_size=100, stride=25, padding=50)
        m2 = F.avg_pool1d(mask, kernel_size=100, stride=25, padding=50)
        p2 = (p2_raw + (m2 < 1e-3).float() * -1e4).amax(dim=-1)

        p5_raw = F.avg_pool1d(feat_masked, kernel_size=250, stride=25, padding=125)
        m5 = F.avg_pool1d(mask, kernel_size=250, stride=25, padding=125)
        p5 = (p5_raw + (m5 < 1e-3).float() * -1e4).amax(dim=-1)

        p15_raw = F.avg_pool1d(feat_masked, kernel_size=750, stride=50, padding=375)
        m15 = F.avg_pool1d(mask, kernel_size=750, stride=50, padding=375)
        p15 = (p15_raw + (m15 < 1e-3).float() * -1e4).amax(dim=-1)

        # Sustain-damped 2s burst: captures genuine bursts, suppresses isolated gimmick spikes
        burst_excess = F.relu(p2 - p5)
        sustain_ratio = p5.abs() / (burst_excess + 0.1)
        p2_damped = p5 + burst_excess * torch.tanh(sustain_ratio)

        # Instantaneous 1D strain curve (Level units)
        s_raw = F.softplus(self.strain_conv(feat)) * mask
        s_peak = s_raw.amax(dim=-1, keepdim=True).clamp(min=1e-5)

        # Peak-relative active gating: ignores quiet intros/outros (<65% peak) to avoid marathon dilution
        active_gate = torch.sigmoid((s_raw - 0.65 * s_peak) / (0.10 * s_peak + 1e-5)) * mask
        active_sum = active_gate.sum(dim=-1).clamp(min=1.0)
        active_gap = (feat * active_gate).sum(dim=-1) / active_sum
        s_active_mean = (s_raw * active_gate).sum(dim=-1) / active_sum
        s_max = s_peak.squeeze(-1)

        # Stamina duration volume: log of active seconds
        stamina_vol = torch.log1p(active_sum * 0.02)

        # Masked attention pool
        attn_logits = self.attn_conv(feat).float() + (mask - 1.0) * 1e4
        attn_weights = (F.softmax(attn_logits, dim=-1) * mask).to(feat.dtype)
        attn_pool = (feat * attn_weights).sum(dim=-1)

        # Scaled strain scalars for MLP input conditioning
        s_max_norm = s_max * 0.01
        s_active_norm = s_active_mean * 0.01

        pooled = torch.cat([p2_damped, p5, p15, active_gap, attn_pool, s_max_norm, s_active_norm, stamina_vol], dim=-1)
        raw = self.head(pooled).squeeze(-1)

        # Centered Double-Softplus: strictly bounded in [1.0, 195.0], linear ~1.0 slope in [15.0, 140.0]
        z = 75.0 + 30.0 * raw
        level = 1.0 + 5.0 * F.softplus((z - 1.0) / 5.0) - 20.0 * F.softplus((z - 195.0) / 20.0)

        # 1-second downsampled strain curve for HUD timeline visualization
        strain_timeline = F.avg_pool1d(s_raw, kernel_size=50, stride=50, ceil_mode=True).squeeze(1)
        return level, strain_timeline


def compute_loss(pred, y, mask, strain_curve):
    pred_f = pred.float()
    y_f = y.float()
    diff = pred_f - y_f
    base_loss = F.smooth_l1_loss(pred_f, y_f, reduction="none", beta=1.0)

    # 1. Targeted under-prediction penalty focused on Lv. 70-92 & marathon charts (>150s)
    dur_sec = mask.sum(dim=-1).squeeze(1) * 0.02
    mid_high_focus = torch.exp(-0.5 * ((y_f - 82.0) / 9.0) ** 2)
    marathon_factor = torch.sigmoid((dur_sec - 150.0) / 30.0) * torch.sigmoid((105.0 - y_f) / 5.0)
    under_mask = (diff < 0).float()
    asym_weight = 1.0 + under_mask * (0.40 * mid_high_focus + 0.35 * marathon_factor)
    loss_point = (base_loss * asym_weight).mean()

    # 2. Pairwise slope loss across mini-batch (enforces 1:1 difficulty slope across gaps)
    pred_diff = pred_f.unsqueeze(1) - pred_f.unsqueeze(0)
    y_diff = y_f.unsqueeze(1) - y_f.unsqueeze(0)
    valid_pairs = ((y_diff.abs() >= 5.0) & (y_diff.abs() <= 45.0)).float()
    pair_error = F.smooth_l1_loss(pred_diff, y_diff, beta=1.0, reduction="none")
    loss_rank = (pair_error * valid_pairs).sum() / valid_pairs.sum().clamp(min=1.0)

    # 3. Auxiliary strain calibration (anchors HUD strain curve in true level units)
    s_peak = strain_curve.float().amax(dim=-1)
    s_mean = strain_curve.float().mean(dim=-1)
    effective_strain = 0.70 * s_peak + 0.30 * s_mean
    loss_strain = F.smooth_l1_loss(effective_strain, y_f, beta=2.0)

    return loss_point + 0.25 * loss_rank + 0.10 * loss_strain


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred, _ = model(x, mask)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
    mask_70_85 = (targets >= 70) & (targets < 85)
    bias_70_85 = np.mean((preds - targets)[mask_70_85]) if np.any(mask_70_85) else 0.0
    return mae, rmse, corr, bias_70_85, preds, targets


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Model v5 on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    data = np.load(DATASET_PATH)
    flat_grids = data["flat_grids"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    levels = data["levels"]
    priorities = data["priorities"] if "priorities" in data else None
    n_charts = len(levels)
    print(f"Loaded dataset: {n_charts:,} charts")

    indices = list(range(n_charts))
    random.shuffle(indices)
    split_idx = int(n_charts * (1.0 - VAL_SPLIT))
    val_indices = indices[split_idx:]
    val_set = set(val_indices)
    train_indices = [i for i in range(n_charts) if i not in val_set]
    print(f"Train set: {len(train_indices):,} charts | Validation set: {len(val_indices):,} charts")

    train_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, train_indices, augment=True, priorities=priorities)
    val_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, val_indices, augment=False)

    train_lengths = [lengths[i] for i in train_indices]
    val_lengths = [lengths[i] for i in val_indices]
    train_sampler = LengthBucketSampler(train_lengths, batch_size=BATCH_SIZE, shuffle=True)
    val_sampler = LengthBucketSampler(val_lengths, batch_size=BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = O2JamLevelPredictor(channels=128).to(device)
    adapter = ServerCalibrationAdapter().to(device) if priorities is not None else None
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    opt_params = list(model.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
    optimizer = torch.optim.AdamW(opt_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if adapter is not None:
            adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if adapter is not None:
                x, y, mask, prios = [b.to(device) for b in batch]
            else:
                x, y, mask = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred, strain_curve = model(x, mask)
                pred_for_loss = adapter(pred, prios) if adapter is not None else pred
                loss = compute_loss(pred_for_loss, y, mask, strain_curve)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation
        val_mae, val_rmse, val_r, val_bias_70_85, _, _ = evaluate(model, val_loader, device)

        is_best = val_mae < best_val_mae
        marker = " [BEST]" if is_best else ""
        if is_best:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
                "val_r": float(val_r),
                "val_bias_70_85": float(val_bias_70_85),
                "val_indices": val_indices,
            }, CHECKPOINT_PATH)

        lr_cur = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.3f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | Bias 70-85: {val_bias_70_85:+5.2f} | R: {val_r:.3f} | LR: {lr_cur:.1e}{marker}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Best Validation MAE: {best_val_mae:.2f}")

    # Export to ONNX
    print(f"\nExporting best checkpoint to {ONNX_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = model.to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.zeros((1, 500, 7), dtype=torch.uint8)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["note_grid"],
        output_names=["predicted_level", "strain_curve"],
        dynamic_axes={
            "note_grid": {0: "batch_size", 1: "time_steps"},
            "predicted_level": {0: "batch_size"},
            "strain_curve": {0: "batch_size", 1: "timeline_steps"}
        }
    )
    print(f"Saved {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1048576:.2f} MB)")
    shutil.copyfile(ONNX_PATH, EXTENSION_ONNX)
    print(f"Updated extension copy at {EXTENSION_ONNX}")


if __name__ == "__main__":
    train()
