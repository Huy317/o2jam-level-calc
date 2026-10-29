"""
Step 2: Build 2D Time-Step Grid Dataset from O2Jam Beatmaps.
Strictly READONLY access to D:\\osu!\\Songs.
Outputs:
  - data/o2jam_dataset.npz (packed uint8 arrays, offsets, lengths, levels)
  - data/dataset_metadata.json (chart metadata, file paths, song titles, versions)
"""

import hashlib
import json
import os
import re
import sys
import time
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SONGS_DIR = r"D:\osu!\Songs"
OUTPUT_DIR = "data"
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "o2jam_dataset.npz")
OUTPUT_META = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

STEP_MS = 20  # 50 frames per second
MAX_FRAMES_CAP = 25000  # Safety cap: 500 seconds (8.33 minutes)
LEVEL_REGEX = re.compile(r"\[(\d+)\]\s*$")
DMJAM_TAGS = {"[D]", "[H]", "[N]", "[E]"}


def get_priority(version_str):
    # DMJAM tags take precedence (priority 0) over O2EZ (priority 1)
    if any(tag in version_str for tag in DMJAM_TAGS):
        return 0
    if "[O2EZ]" in version_str:
        return 1
    return 2


def parse_osu_file(file_path):
    mode = None
    circle_size = None
    version_line = None
    title = ""
    hit_objects = []
    is_hitobjects = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[HitObjects]":
                is_hitobjects = True
                continue
            if not is_hitobjects:
                if stripped.startswith("Mode:"):
                    mode = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("CircleSize:"):
                    circle_size = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Version:"):
                    version_line = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Title:"):
                    title = stripped.split(":", 1)[1].strip()
            else:
                if not stripped or stripped.startswith("["):
                    break
                parts = stripped.split(",")
                if len(parts) >= 4:
                    x = int(parts[0])
                    t = int(parts[2])
                    n_type = int(parts[3])
                    col = min(max(int(x * 7 // 512), 0), 6)
                    if n_type & 128:
                        end_t = int(parts[5].split(":")[0]) if len(parts) > 5 else t
                    else:
                        end_t = t
                    hit_objects.append((t, end_t, col))

    if mode != "3" or circle_size != "7" or not version_line:
        return None

    match = LEVEL_REGEX.search(version_line)
    if not match:
        return None

    level = int(match.group(1))
    if level <= 1 or len(hit_objects) < 20:
        return None

    hit_objects.sort(key=lambda o: (o[0], o[1], o[2]))
    note_hash = hashlib.md5(str(hit_objects).encode("utf-8")).hexdigest()

    return {
        "file_path": file_path,
        "folder": os.path.basename(os.path.dirname(file_path)),
        "file_name": os.path.basename(file_path),
        "title": title,
        "version": version_line,
        "level": level,
        "note_hash": note_hash,
        "hit_objects": hit_objects,
        "priority": get_priority(version_line),
    }


def build_grid(hit_objects):
    first_t = hit_objects[0][0]
    last_t = max(max(o[0], o[1]) for o in hit_objects)
    duration_ms = max(0, last_t - first_t)

    t_frames = (duration_ms // STEP_MS) + 1
    t_frames = min(t_frames, MAX_FRAMES_CAP)

    grid = np.zeros((t_frames, 7), dtype=np.uint8)

    for start_t, end_t, col in hit_objects:
        start_frame = (start_t - first_t) // STEP_MS
        end_frame = (end_t - first_t) // STEP_MS

        if start_frame >= t_frames:
            continue

        end_frame = min(end_frame, t_frames - 1)

        if end_frame <= start_frame:
            grid[start_frame, col] = 1  # Tap note
        else:
            grid[start_frame, col] = 2  # LN Head
            if end_frame > start_frame + 1:
                grid[start_frame + 1:end_frame, col] = 3  # LN Body
            grid[end_frame, col] = 4  # LN Tail

    return grid, t_frames, duration_ms


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Scanning O2Jam directories in {SONGS_DIR}...")
    o2jam_dirs = [
        d for d in os.listdir(SONGS_DIR)
        if d.startswith("[_O2Jam_]") and os.path.isdir(os.path.join(SONGS_DIR, d))
    ]

    all_parsed = []
    for d in o2jam_dirs:
        dp = os.path.join(SONGS_DIR, d)
        for f in os.listdir(dp):
            if f.lower().endswith(".osu"):
                fp = os.path.join(dp, f)
                info = parse_osu_file(fp)
                if info:
                    all_parsed.append(info)

    print(f"Total valid candidate charts parsed: {len(all_parsed)}")

    # Deduplication grouping
    by_hash = {}
    for item in all_parsed:
        h = item["note_hash"]
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append(item)

    unique_charts = []
    discarded_duplicates = 0

    for h, group in by_hash.items():
        if len(group) == 1:
            unique_charts.append(group[0])
        else:
            discarded_duplicates += (len(group) - 1)
            # Sort by priority (0: DMJAM, 1: O2EZ, 2: other)
            group.sort(key=lambda x: x["priority"])
            unique_charts.append(group[0])

    print(f"Deduplication complete: {len(unique_charts)} unique charts retained ({discarded_duplicates} duplicates discarded).")

    # Build 2D grids
    print("Constructing 2D discrete time grids (STEP_MS = 20ms)...")
    levels = []
    lengths = []
    durations = []
    grids = []
    metadata = []

    for idx, chart in enumerate(unique_charts):
        grid, t_frames, dur_ms = build_grid(chart["hit_objects"])
        grids.append(grid)
        levels.append(chart["level"])
        lengths.append(t_frames)
        durations.append(dur_ms)
        metadata.append({
            "id": idx,
            "title": chart["title"],
            "version": chart["version"],
            "level": chart["level"],
            "folder": chart["folder"],
            "file_name": chart["file_name"],
            "note_count": len(chart["hit_objects"]),
            "duration_ms": dur_ms,
            "frames": t_frames,
            "note_hash": chart["note_hash"],
        })

    levels_arr = np.array(levels, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int32)
    durations_arr = np.array(durations, dtype=np.int32)

    # Compute offsets for zero-copy flat uint8 representation
    offsets_arr = np.zeros(len(grids) + 1, dtype=np.int64)
    for i, g in enumerate(grids):
        offsets_arr[i + 1] = offsets_arr[i] + g.size

    total_cells = offsets_arr[-1]
    flat_grids = np.empty(total_cells, dtype=np.uint8)

    for i, g in enumerate(grids):
        start = offsets_arr[i]
        end = offsets_arr[i + 1]
        flat_grids[start:end] = g.reshape(-1)

    print(f"Total dataset frames: {lengths_arr.sum():,} ({total_cells:,} bytes = {total_cells / (1024**2):.2f} MB uncompressed)")

    # Save to disk
    print(f"Saving dataset to {OUTPUT_NPZ}...")
    np.savez_compressed(
        OUTPUT_NPZ,
        flat_grids=flat_grids,
        offsets=offsets_arr,
        lengths=lengths_arr,
        levels=levels_arr,
        durations=durations_arr,
    )

    print(f"Saving metadata to {OUTPUT_META}...")
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    npz_size_mb = os.path.getsize(OUTPUT_NPZ) / (1024**2)
    print(f"Dataset successfully saved! Compressed file size: {npz_size_mb:.2f} MB")
    print(f"Total pipeline execution time: {time.time() - t0:.2f}s")

    # Verification: Read back and check random samples
    print("\n=== Dataset Integrity Self-Check ===")
    loaded = np.load(OUTPUT_NPZ)
    assert len(loaded["levels"]) == len(unique_charts), "Level count mismatch!"
    assert len(loaded["lengths"]) == len(unique_charts), "Lengths count mismatch!"
    assert len(loaded["offsets"]) == len(unique_charts) + 1, "Offsets count mismatch!"

    print(f"Self-check: Loaded {len(loaded['levels'])} charts successfully.")
    print(f"Level Statistics -> Min: {loaded['levels'].min():.0f}, Max: {loaded['levels'].max():.0f}, Mean: {loaded['levels'].mean():.2f}, Std: {loaded['levels'].std():.2f}")
    print(f"Sequence Length (frames) -> Min: {loaded['lengths'].min()}, Median: {int(np.median(loaded['lengths']))}, Max: {loaded['lengths'].max()}")

    # Verify 5 random sample reconstructions
    sample_indices = [0, len(unique_charts) // 4, len(unique_charts) // 2, 3 * len(unique_charts) // 4, len(unique_charts) - 1]
    for s_idx in sample_indices:
        s_start = loaded["offsets"][s_idx]
        s_end = loaded["offsets"][s_idx + 1]
        s_len = loaded["lengths"][s_idx]
        s_lvl = loaded["levels"][s_idx]
        reconstructed = loaded["flat_grids"][s_start:s_end].reshape(s_len, 7)
        assert reconstructed.shape == (s_len, 7), f"Shape error at {s_idx}"
        unique_vals = np.unique(reconstructed)
        meta_sample = metadata[s_idx]
        print(f"Sample {s_idx:4d}: Level {s_lvl:3.0f} | Length {s_len:5d} frames ({meta_sample['duration_ms']/1000:5.1f}s) | Unique codes: {list(unique_vals)} | {meta_sample['title'][:30]}")

    print("\nDataset generation and validation PASSED!")


if __name__ == "__main__":
    main()
