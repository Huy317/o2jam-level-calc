import json
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from step3_train import O2JamLevelPredictor, O2JamDataset, collate_fn, evaluate, BATCH_SIZE, VAL_SPLIT, SEED

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(REPO_ROOT, "data", "o2jam_dataset.npz")
METADATA_PATH = os.path.join(REPO_ROOT, "data", "dataset_metadata.json")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "best_model.pt")

data = np.load(DATASET_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

random.seed(SEED)
indices = list(range(len(data["levels"])))
random.shuffle(indices)
split_idx = int(len(indices) * (1.0 - VAL_SPLIT))
val_indices = indices[split_idx:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = O2JamLevelPredictor(channels=128).to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
net.load_state_dict(ckpt["model_state_dict"])

val_dataset = O2JamDataset(data["flat_grids"], data["offsets"], data["lengths"], data["levels"], val_indices, augment=False)
loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
mae, rmse, r, preds, targets = evaluate(net, loader, device)

errors = np.abs(preds - targets)
signed_errors = preds - targets

print(f"Overall Validation -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R: {r:.3f}")
print("Error percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  P{p:2d}: <= {np.percentile(errors, p):.2f} levels")

buckets = [(20, 50), (50, 80), (80, 110), (110, 140), (140, 181)]
print("\nAccuracy by Level Tier:")
for lo, hi in buckets:
    mask = (targets >= lo) & (targets < hi)
    b_mae = np.mean(errors[mask])
    b_cnt = np.sum(mask)
    b_bias = np.mean(signed_errors[mask])
    print(f"  Lv {lo:3d}-{hi:3d}: {b_cnt:3d} charts | MAE: {b_mae:5.2f} | Bias: {b_bias:+5.2f}")

worst_indices = np.argsort(errors)[-10:][::-1]
print("\nWorst 10 Outliers:")
for w in worst_indices:
    c_idx = val_indices[w]
    m = meta[c_idx]
    print(f"  Actual: Lv.{targets[w]:3.0f} | Pred: Lv.{preds[w]:5.1f} (Err: {signed_errors[w]:+5.1f}) | {m['version']:<25} | {m['title'][:30]}")
