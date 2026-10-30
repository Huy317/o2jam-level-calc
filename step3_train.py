"""
Step 3: Train O2Jam 7K Level Predictor on Unified Dataset (4,055 charts).
Architecture:
  - 37 input channels (35 one-hot states + 1 chord attack count + 1 hold count)
  - 6 Residual Conv1D blocks (dilations 1, 2, 4, 8, 16, 32) -> ~5.06s receptive field
  - Multi-scale temporal pooling (1s micro-burst, 5s phrase, 15s climax, global average stamina, attention)
  - Smooth sigmoid bounded output [15.0, 185.0]
  - Tier-weighted + Asymmetric Huber loss
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

DATASET_PATH = os.path.join("data", "o2jam_dataset.npz")
METADATA_PATH = os.path.join("data", "dataset_metadata.json")
CHECKPOINT_PATH = "best_model.pt"
ONNX_PATH = "model.onnx"
EXTENSION_ONNX = os.path.join("extension", "o2jam_level_predictor", "model.onnx")

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
    def __init__(self, flat_grids, offsets, lengths, levels, indices, augment=False):
        self.flat_grids = flat_grids
        self.offsets = offsets
        self.lengths = lengths
        self.levels = levels
        self.indices = indices
        self.augment = augment

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
        return torch.from_numpy(grid), torch.tensor(level, dtype=torch.float32)


def collate_fn(batch):
    grids, levels = zip(*batch)
    lengths = [g.shape[0] for g in grids]
    max_len = max(lengths)

    batch_size = len(grids)
    padded = torch.zeros((batch_size, max_len, 7), dtype=torch.uint8)
    for i, g in enumerate(grids):
        padded[i, :g.shape[0], :] = g

    levels_tensor = torch.stack(levels)
    return padded, levels_tensor


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
        # Input representation: 35 one-hot states + 1 chord attack count + 1 hold count = 37 channels
        self.stem = nn.Sequential(
            nn.Conv1d(37, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # 6 Dilated Residual blocks covering receptive field up to ~5.0 seconds
        self.block1 = ResidualBlock1D(channels, kernel_size=5, dilation=1)   # rf: 5 (100ms)
        self.block2 = ResidualBlock1D(channels, kernel_size=5, dilation=2)   # rf: 13 (260ms)
        self.block3 = ResidualBlock1D(channels, kernel_size=5, dilation=4)   # rf: 29 (580ms)
        self.block4 = ResidualBlock1D(channels, kernel_size=5, dilation=8)   # rf: 61 (1.22s)
        self.block5 = ResidualBlock1D(channels, kernel_size=5, dilation=16)  # rf: 125 (2.50s)
        self.block6 = ResidualBlock1D(channels, kernel_size=5, dilation=32)  # rf: 253 (5.06s)

        # Attention pooling scoring layer
        self.attn_conv = nn.Conv1d(channels, 1, kernel_size=1)

        # Head: Multi-Scale Aggregation (p1 + p5 + p15 + gap + attn = 5 * channels = 640)
        self.head = nn.Sequential(
            nn.Linear(channels * 5, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (Batch, T, 7) uint8 categorical integers [0..4]
        b, t, c = x.shape
        
        # 1. One-hot representation: (B, 35, T)
        one_hot = F.one_hot(x.long(), num_classes=5).float().view(b, t, 35).permute(0, 2, 1)
        
        # 2. Explicit chord density & hold channels (B, 1, T)
        taps = ((x == 1) | (x == 2)).float().sum(dim=-1, keepdim=True).permute(0, 2, 1)
        holds = ((x == 3) | (x == 4)).float().sum(dim=-1, keepdim=True).permute(0, 2, 1)
        
        inp = torch.cat([one_hot, taps, holds], dim=1)  # (B, 37, T)

        feat = self.stem(inp)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        feat = self.block5(feat)
        feat = self.block6(feat)

        # Multi-scale temporal pooling:
        p1 = F.avg_pool1d(feat, kernel_size=50, stride=10, padding=25).amax(dim=-1)
        p5 = F.avg_pool1d(feat, kernel_size=250, stride=25, padding=125).amax(dim=-1)
        p15 = F.avg_pool1d(feat, kernel_size=750, stride=50, padding=375).amax(dim=-1)
        gap = feat.mean(dim=-1)
        attn_w = torch.softmax(self.attn_conv(feat), dim=-1)
        attn_pool = (feat * attn_w).sum(dim=-1)

        pooled = torch.cat([p1, p5, p15, gap, attn_pool], dim=-1)
        raw = self.head(pooled).squeeze(-1)
        
        # Strictly bound output to valid O2Jam range [15.0, 185.0]
        return 15.0 + 170.0 * torch.sigmoid(raw)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
    return mae, rmse, corr, preds, targets


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Model v2 on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    data = np.load(DATASET_PATH)
    flat_grids = data["flat_grids"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    levels = data["levels"]
    n_charts = len(levels)
    print(f"Loaded dataset: {n_charts:,} charts")

    indices = list(range(n_charts))
    random.shuffle(indices)
    split_idx = int(n_charts * (1.0 - VAL_SPLIT))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    print(f"Train set: {len(train_indices):,} charts | Validation set: {len(val_indices):,} charts")

    train_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, train_indices, augment=True)
    val_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = O2JamLevelPredictor(channels=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                # Tier-weighted + Asymmetric Huber loss: prevents negative bias on elite stamina charts
                tier_weights = torch.ones_like(y)
                tier_weights = torch.where(y > 110, 1.8, tier_weights)
                tier_weights = torch.where(y > 140, 2.2, tier_weights)
                asym_weights = torch.where((pred < y) & (y > 100), 1.5, 1.0)
                loss_elements = F.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
                loss = (tier_weights * asym_weights * loss_elements).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation
        val_mae, val_rmse, val_r, _, _ = evaluate(model, val_loader, device)

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
                "val_indices": val_indices,
            }, CHECKPOINT_PATH)

        lr_cur = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.3f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | R: {val_r:.3f} | LR: {lr_cur:.1e}{marker}")

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
        output_names=["predicted_level"],
        dynamic_axes={
            "note_grid": {0: "batch_size", 1: "time_steps"},
            "predicted_level": {0: "batch_size"}
        }
    )
    print(f"Saved {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1048576:.2f} MB)")
    shutil.copyfile(ONNX_PATH, EXTENSION_ONNX)
    print(f"Updated extension copy at {EXTENSION_ONNX}")


if __name__ == "__main__":
    train()
