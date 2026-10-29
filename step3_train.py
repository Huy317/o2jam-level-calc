"""
Step 3: Train 1D-CNN Level Predictor and Export to ONNX.
Uses PyTorch GPU acceleration (RTX 4070 SUPER) with mixed precision (AMP).
Exports best checkpoint to model.onnx with verified <10ms inference.
"""

import json
import os
import random
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

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 30
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

    # Dynamic batch padding to batch-maximum length
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
        # Input representation: 7 columns * 5 categorical states = 35 channels
        self.stem = nn.Sequential(
            nn.Conv1d(35, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # 4 Dilated Residual blocks covering receptive field from short jacks to ~1.5s patterns
        self.block1 = ResidualBlock1D(channels, kernel_size=5, dilation=1)
        self.block2 = ResidualBlock1D(channels, kernel_size=5, dilation=2)
        self.block3 = ResidualBlock1D(channels, kernel_size=5, dilation=4)
        self.block4 = ResidualBlock1D(channels, kernel_size=5, dilation=8)

        # Attention pooling scoring layer
        self.attn_conv = nn.Conv1d(channels, 1, kernel_size=1)

        # Head: Multi-Scale Aggregation (Avg + Max + Attention = 3 * channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (Batch, T, 7) uint8 categorical integers [0..4]
        b, t, c = x.shape
        # One-hot encode: (B, T, 7, 5) -> (B, T, 35) -> (B, 35, T)
        one_hot = F.one_hot(x.long(), num_classes=5).float()
        one_hot = one_hot.view(b, t, 35).permute(0, 2, 1)

        feat = self.stem(one_hot)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)

        # Multi-scale aggregation
        # 1. Global Average Pooling (overall stamina/density)
        gap = feat.mean(dim=-1)
        # 2. Global Max Pooling (peak burst)
        gmp = feat.amax(dim=-1)
        # 3. Attention Pooling (hardest difficulty spikes)
        attn_weights = torch.softmax(self.attn_conv(feat), dim=-1)
        attn_pool = (feat * attn_weights).sum(dim=-1)

        pooled = torch.cat([gap, gmp, attn_pool], dim=-1)
        out = self.head(pooled)
        return out.squeeze(-1)


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
    # Pearson R
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
    return mae, rmse, corr, preds, targets


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== O2Jam 7K Level Predictor Training ===")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"Loading packed dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH)
    flat_grids = data["flat_grids"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    levels = data["levels"]
    num_samples = len(levels)

    print(f"Total charts in dataset: {num_samples}")

    # Train / Val Split (85% / 15%)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(num_samples * (1.0 - VAL_SPLIT))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"Train samples: {len(train_indices)} | Validation samples: {len(val_indices)}")

    train_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, train_indices, augment=True)
    val_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, val_indices, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    model = O2JamLevelPredictor(channels=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: 1D-CNN with Multi-Scale Pooling (Parameters: {total_params:,})")

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_epoch = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs with Mixed Precision (AMP)...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t_epoch_start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)
        val_mae, val_rmse, val_corr, _, _ = evaluate(model, val_loader, device)
        epoch_dur = time.time() - t_epoch_start

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
                "val_corr": float(val_corr),
            }, CHECKPOINT_PATH)

        star = " *" if is_best else ""
        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS:2d}] {epoch_dur:4.1f}s | Train Loss: {train_loss:6.2f} | Val MAE: {val_mae:5.2f} | Val RMSE: {val_rmse:5.2f} | R: {val_corr:.3f}{star}")

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s!")
    print(f"Best Checkpoint at Epoch {best_epoch}: Validation MAE = {best_val_mae:.2f}")

    # Load best checkpoint for export and final evaluation
    print(f"\nLoading best checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_mae, val_rmse, val_corr, val_preds, val_targets = evaluate(model, val_loader, device)
    print(f"Final Validation Metrics -> MAE: {val_mae:.2f} levels | RMSE: {val_rmse:.2f} | Pearson R: {val_corr:.3f}")

    # Inspect sample predictions across difficulty tiers
    print("\n=== Sample Validation Predictions ===")
    sample_items = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_all = json.load(f)

    for i in range(min(15, len(val_indices))):
        c_idx = val_indices[i]
        meta = meta_all[c_idx]
        actual = val_targets[i]
        predicted = val_preds[i]
        err = predicted - actual
        sample_items.append((actual, predicted, err, meta["title"], meta["version"]))

    sample_items.sort(key=lambda x: x[0])
    for actual, pred, err, title, ver in sample_items:
        print(f"Actual: Lv.{actual:3.0f} | Predicted: Lv.{pred:5.1f} (Err: {err:+5.1f}) | {ver:<25} | {title[:30]}")

    # Export to ONNX
    print(f"\n=== Exporting Model to ONNX ({ONNX_PATH}) ===")
    model.eval()
    dummy_input = torch.randint(0, 5, (1, 3000, 7), dtype=torch.uint8).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["note_grid"],
        output_names=["predicted_level"],
        dynamic_axes={
            "note_grid": {0: "batch_size", 1: "time_frames"},
            "predicted_level": {0: "batch_size"},
        },
    )
    print(f"Successfully exported to {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / (1024**2):.2f} MB)")

    # Verify ONNX model and benchmark inference speed
    print("\n=== ONNX Inference Benchmark (<10ms Target) ===")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    # Test on a realistic 2.5-minute song (7,500 frames)
    test_grid = np.random.randint(0, 5, size=(1, 7500, 7), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, {"note_grid": test_grid})

    # Benchmark 50 runs
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        ort_out = ort_session.run(None, {"note_grid": test_grid})
        latencies.append((time.perf_counter() - t_start) * 1000)

    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    print(f"Input: 1 song, 7,500 frames (~2.5 minutes)")
    print(f"ONNX CPU Latency: Mean = {avg_ms:.2f} ms | P95 = {p95_ms:.2f} ms | Min = {min(latencies):.2f} ms")
    if avg_ms < 10.0:
        print(f"VERIFICATION SUCCESS: Inference speed {avg_ms:.2f}ms is well under the <10ms target!")
    else:
        print(f"Inference latency: {avg_ms:.2f}ms")


def export_and_benchmark():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading packed dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH)
    flat_grids = data["flat_grids"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    levels = data["levels"]
    num_samples = len(levels)

    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(num_samples * (1.0 - VAL_SPLIT))
    val_indices = indices[split_idx:]

    val_dataset = O2JamDataset(flat_grids, offsets, lengths, levels, val_indices, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = O2JamLevelPredictor(channels=128).to(device)
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_mae, val_rmse, val_corr, val_preds, val_targets = evaluate(model, val_loader, device)
    print(f"Validation Metrics -> MAE: {val_mae:.2f} levels | RMSE: {val_rmse:.2f} | Pearson R: {val_corr:.3f}")

    print("\n=== Sample Validation Predictions ===")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_all = json.load(f)

    sample_items = []
    for i in range(min(15, len(val_indices))):
        c_idx = val_indices[i]
        meta = meta_all[c_idx]
        actual = val_targets[i]
        predicted = val_preds[i]
        err = predicted - actual
        sample_items.append((actual, predicted, err, meta["title"], meta["version"]))

    sample_items.sort(key=lambda x: x[0])
    for actual, pred, err, title, ver in sample_items:
        print(f"Actual: Lv.{actual:3.0f} | Predicted: Lv.{pred:5.1f} (Err: {err:+5.1f}) | {ver:<25} | {title[:30]}")

    print(f"\n=== Exporting Model to ONNX ({ONNX_PATH}) ===")
    model.eval()
    dummy_input = torch.randint(0, 5, (1, 3000, 7), dtype=torch.uint8).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["note_grid"],
        output_names=["predicted_level"],
        dynamic_axes={
            "note_grid": {0: "batch_size", 1: "time_frames"},
            "predicted_level": {0: "batch_size"},
        },
    )
    print(f"Successfully exported to {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / (1024**2):.2f} MB)")

    print("\n=== ONNX Inference Benchmark (<10ms Target) ===")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    test_grid = np.random.randint(0, 5, size=(1, 7500, 7), dtype=np.uint8)

    for _ in range(5):
        _ = ort_session.run(None, {"note_grid": test_grid})

    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = ort_session.run(None, {"note_grid": test_grid})
        latencies.append((time.perf_counter() - t_start) * 1000)

    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    print(f"Input: 1 song, 7,500 frames (~2.5 minutes)")
    print(f"ONNX CPU Latency: Mean = {avg_ms:.2f} ms | P95 = {p95_ms:.2f} ms | Min = {min(latencies):.2f} ms")
    if avg_ms < 10.0:
        print(f"VERIFICATION SUCCESS: Inference speed {avg_ms:.2f}ms is well under the <10ms target!")


if __name__ == "__main__":
    if "--eval-only" in sys.argv or (os.path.exists(CHECKPOINT_PATH) and "--train" not in sys.argv):
        export_and_benchmark()
    else:
        train()
