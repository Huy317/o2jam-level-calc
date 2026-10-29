"""
CLI O2Jam 7K Level Predictor.
Instantly predicts the O2Jam difficulty level of any 7K .osu file using model.onnx.
Usage:
  python predict.py "path/to/chart.osu"
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort

STEP_MS = 20
MAX_FRAMES_CAP = 25000
ONNX_MODEL_PATH = "model.onnx"


def parse_osu_beatmap(file_path):
    hit_objects = []
    title = ""
    version = ""
    mode = None
    circle_size = None
    is_hitobjects = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[HitObjects]":
                is_hitobjects = True
                continue
            if not is_hitobjects:
                if stripped.startswith("Title:"):
                    title = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Version:"):
                    version = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Mode:"):
                    mode = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("CircleSize:"):
                    circle_size = stripped.split(":", 1)[1].strip()
            else:
                if not stripped or stripped.startswith("["):
                    break
                parts = stripped.split(",")
                if len(parts) >= 4:
                    x = int(parts[0])
                    t = int(parts[2])
                    n_type = int(parts[3])
                    col = min(max(int(x * 7 // 512), 0), 6)
                    end_t = int(parts[5].split(":")[0]) if (n_type & 128 and len(parts) > 5) else t
                    hit_objects.append((t, end_t, col))

    if not hit_objects:
        return None, "No hit objects found."

    hit_objects.sort(key=lambda o: (o[0], o[1]))
    first_t = hit_objects[0][0]
    last_t = max(max(o[0], o[1]) for o in hit_objects)
    duration_ms = max(0, last_t - first_t)
    t_frames = min((duration_ms // STEP_MS) + 1, MAX_FRAMES_CAP)

    grid = np.zeros((1, t_frames, 7), dtype=np.uint8)
    for start_t, end_t, col in hit_objects:
        start_frame = (start_t - first_t) // STEP_MS
        end_frame = min((end_t - first_t) // STEP_MS, t_frames - 1)
        if start_frame >= t_frames:
            continue
        if end_frame <= start_frame:
            grid[0, start_frame, col] = 1
        else:
            grid[0, start_frame, col] = 2
            if end_frame > start_frame + 1:
                grid[0, start_frame + 1:end_frame, col] = 3
            grid[0, end_frame, col] = 4

    meta = {
        "title": title,
        "version": version,
        "mode": mode,
        "circle_size": circle_size,
        "note_count": len(hit_objects),
        "duration_s": duration_ms / 1000.0,
        "frames": t_frames,
    }
    return grid, meta


def predict(osu_file):
    if not os.path.exists(osu_file):
        print(f"Error: File not found: {osu_file}")
        return

    grid, meta = parse_osu_beatmap(osu_file)
    if grid is None:
        print(f"Error parsing beatmap: {meta}")
        return

    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    
    # Measure latency
    t0 = time.perf_counter()
    outputs = session.run(None, {"note_grid": grid})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predicted_level = float(outputs[0][0])

    print("=========================================")
    print(f"  Song Title : {meta['title']}")
    print(f"  Difficulty : {meta['version']}")
    print(f"  Key Count  : {meta['circle_size']}K (Mode {meta['mode']})")
    print(f"  Notes      : {meta['note_count']:,} notes")
    print(f"  Duration   : {meta['duration_s']:.1f}s ({meta['frames']:,} frames)")
    print("-----------------------------------------")
    print(f"  >>> Predicted O2Jam Level : Lv. {predicted_level:.1f} (Round: Lv. {round(predicted_level)}) <<<")
    print(f"  Inference Latency         : {latency_ms:.2f} ms")
    print("=========================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_beatmap.osu>")
    else:
        predict(sys.argv[1])
