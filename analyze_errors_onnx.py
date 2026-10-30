import json
import os
import time
import numpy as np
import onnxruntime as ort

DATASET_PATH = os.path.join("data", "o2jam_dataset.npz")
METADATA_PATH = os.path.join("data", "dataset_metadata.json")
ONNX_PATH = "model.onnx"

def main():
    print("=== O2Jam 7K Level Error Analyzer (ONNX Runtime) ===")
    if not os.path.exists(DATASET_PATH) or not os.path.exists(METADATA_PATH):
        print("Dataset not found!")
        return

    data = np.load(DATASET_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    flat_grids = data["flat_grids"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    levels = data["levels"]
    durations = data["durations"]
    n_charts = len(levels)

    print(f"Loaded {n_charts} charts from dataset.")
    print(f"Loading ONNX model from {ONNX_PATH}...")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    preds = np.zeros(n_charts, dtype=np.float32)
    t0 = time.time()

    print("Running inference across all charts...")
    for i in range(n_charts):
        start = offsets[i]
        end = offsets[i + 1]
        t_len = lengths[i]
        grid = flat_grids[start:end].reshape(1, t_len, 7)
        out = session.run(None, {input_name: grid})
        preds[i] = float(out[0][0])
        if (i + 1) % 250 == 0 or (i + 1) == n_charts:
            print(f"  Processed {i + 1}/{n_charts} charts...")

    eval_time = time.time() - t0
    print(f"Inference completed in {eval_time:.2f}s ({eval_time / n_charts * 1000:.1f}ms/chart avg)\n")

    targets = levels
    errors = np.abs(preds - targets)
    signed_errors = preds - targets

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    r = np.corrcoef(preds, targets)[0, 1] if n_charts > 1 else 0.0

    print("=" * 60)
    print(f"OVERALL PERFORMANCE ({n_charts} charts):")
    print(f"  MAE:       {mae:.2f} levels")
    print(f"  RMSE:      {rmse:.2f} levels")
    print(f"  Pearson R: {r:.4f}")
    print("=" * 60)

    print("\nERROR PERCENTILES:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: <= {np.percentile(errors, p):.2f} levels")

    print("\nPERFORMANCE BY LEVEL TIER:")
    print(f"{'Tier':<12} | {'Count':<5} | {'MAE':<6} | {'RMSE':<6} | {'Bias (Pred - Actual)':<20}")
    print("-" * 60)
    buckets = [(20, 45), (45, 70), (70, 95), (95, 120), (120, 180)]
    for lo, hi in buckets:
        mask = (targets >= lo) & (targets < hi)
        cnt = np.sum(mask)
        if cnt == 0:
            continue
        b_mae = np.mean(errors[mask])
        b_rmse = np.sqrt(np.mean(errors[mask] ** 2))
        b_bias = np.mean(signed_errors[mask])
        print(f"Lv {lo:3d}-{hi:3d}  | {cnt:5d} | {b_mae:6.2f} | {b_rmse:6.2f} | {b_bias:+6.2f}")

    print("\nPERFORMANCE BY DURATION (Song Length / Stamina):")
    print(f"{'Duration':<15} | {'Count':<5} | {'MAE':<6} | {'Bias':<6}")
    print("-" * 45)
    dur_buckets = [
        ("Short (< 90s)", durations < 90_000),
        ("Medium (90-150s)", (durations >= 90_000) & (durations < 150_000)),
        ("Long (150-240s)", (durations >= 150_000) & (durations < 240_000)),
        ("Marathon (> 240s)", durations >= 240_000),
    ]
    for name, mask in dur_buckets:
        cnt = np.sum(mask)
        if cnt == 0:
            continue
        print(f"{name:<15} | {cnt:5d} | {np.mean(errors[mask]):6.2f} | {np.mean(signed_errors[mask]):+6.2f}")

    print("\nTOP 15 OUTLIERS (Largest Absolute Error):")
    print(f"{'Actual':<7} | {'Pred':<7} | {'Err':<7} | {'Dur(s)':<6} | {'NPS':<5} | {'Version':<22} | {'Title'}")
    print("-" * 90)
    worst_indices = np.argsort(errors)[-15:][::-1]
    for idx in worst_indices:
        m = meta[idx]
        actual = targets[idx]
        pred = preds[idx]
        err = signed_errors[idx]
        dur_s = m["duration_ms"] / 1000.0
        nps = m["note_count"] / max(1.0, dur_s)
        print(f"Lv.{actual:3.0f}  | Lv.{pred:5.1f} | {err:+6.1f} | {dur_s:5.1f}s | {nps:4.1f} | {m['version'][:22]:<22} | {m['title'][:30]}")

if __name__ == "__main__":
    main()
