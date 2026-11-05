"""
Step 2: Unified O2Jam Dataset Builder (osu!mania converts + Private Server + NSJAM native charts).
Parses 7K .osu charts and binary .ojn charts into discrete (T, 7) uint8 grids.
Robust fuzzy deduplication by note count, duration tolerance (+-2s), and title/hash.

Outputs:
  - data/o2jam_dataset.npz (packed flat uint8 grids, offsets, lengths, levels)
  - data/dataset_metadata.json (comprehensive chart metadata)
"""

import hashlib
import json
import os
import re
import struct
import sys
import time
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SONGS_DIR = os.environ.get("OSU_SONGS_DIR", r"D:\osu!\Songs")
NSJAM_BASE = os.environ.get("NSJAM_DIR", r"D:\Games\NSJAM")
NSJAM_DIRS = {
    "NSM01_Easy": os.path.join(NSJAM_BASE, "NSM01"),
    "NSM02_Normal": os.path.join(NSJAM_BASE, "NSM02"),
    "NSM03_Hard": os.path.join(NSJAM_BASE, "NSM03"),
    "NSM04_LN": os.path.join(NSJAM_BASE, "NSM04"),
}

def find_private_server_dir():
    env_dir = os.environ.get("PRIVATE_SERVER_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    games_base = r"D:\Games"
    if os.path.exists(games_base):
        for entry in os.listdir(games_base):
            cand = os.path.join(games_base, entry)
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "DSong")):
                return cand
    return ""

PRIVATE_SERVER_BASE = find_private_server_dir()
PRIVATE_SERVER_DIRS = {
    "DSong": os.path.join(PRIVATE_SERVER_BASE, "DSong"),
    "ESong": os.path.join(PRIVATE_SERVER_BASE, "ESong"),
    "HSong": os.path.join(PRIVATE_SERVER_BASE, "HSong"),
    "NSong": os.path.join(PRIVATE_SERVER_BASE, "NSong"),
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "o2jam_dataset.npz")
OUTPUT_META = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

STEP_MS = 20  # 50 frames per second
MAX_FRAMES_CAP = 25000  # Safety cap: 500 seconds
LEVEL_REGEX = re.compile(r"\[(\d+)\]\s*$")
DMJAM_TAGS = {"[D]", "[H]", "[N]", "[E]"}

# Priority: Native Private Server (0) > osu! DMJAM converts (1) > NSJAM (2) > O2EZ converts (3)
PRIORITY_PRIVATE = 0
PRIORITY_DMJAM = 1
PRIORITY_NSJAM = 2
PRIORITY_O2EZ = 3


# ── 1. .osu Beatmap Parser ──────────────────────────────────────────────────
def parse_osu_file(file_path):
    mode = None
    circle_size = None
    version_line = None
    title = ""
    hit_objects = []
    is_hitobjects = False

    try:
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
                        end_t = int(parts[5].split(":")[0]) if (n_type & 128 and len(parts) > 5) else t
                        hit_objects.append((t, end_t, col))
    except Exception:
        return None

    if mode != "3" or circle_size != "7" or not version_line:
        return None

    match = LEVEL_REGEX.search(version_line)
    if not match:
        return None

    level = int(match.group(1))
    if level <= 1 or len(hit_objects) < 20:
        return None

    priority = PRIORITY_DMJAM if any(tag in version_line for tag in DMJAM_TAGS) else PRIORITY_O2EZ

    hit_objects.sort(key=lambda o: (o[0], o[1], o[2]))
    first_t = hit_objects[0][0]
    last_t = max(max(o[0], o[1]) for o in hit_objects)
    duration_ms = max(0, last_t - first_t)
    t_frames = min((duration_ms // STEP_MS) + 1, MAX_FRAMES_CAP)

    grid = np.zeros((t_frames, 7), dtype=np.uint8)
    for start_t, end_t, col in hit_objects:
        s_f = (start_t - first_t) // STEP_MS
        e_f = (end_t - first_t) // STEP_MS
        if s_f >= t_frames: continue
        e_f = min(e_f, t_frames - 1)
        if e_f <= s_f:
            grid[s_f, col] = 1
        else:
            grid[s_f, col] = 2
            if e_f > s_f + 1:
                grid[s_f+1:e_f, col] = 3
            grid[e_f, col] = 4

    grid_hash = hashlib.md5(grid.tobytes()).hexdigest()

    return {
        "source_type": "osu",
        "file_path": file_path,
        "folder": os.path.basename(os.path.dirname(file_path)),
        "file_name": os.path.basename(file_path),
        "title": title,
        "version": version_line,
        "level": level,
        "priority": priority,
        "note_count": len(hit_objects),
        "duration_ms": duration_ms,
        "frames": t_frames,
        "grid_hash": grid_hash,
        "grid": grid,
    }


# ── 2. Native .ojn Binary Parser ────────────────────────────────────────────
def parse_ojn_file(file_path, folder_name, source_type="nsjam", priority=PRIORITY_NSJAM):
    try:
        with open(file_path, "rb") as f:
            h = f.read(300)
            if len(h) < 300: return None
            sig = struct.unpack("<I", h[4:8])[0]
            if sig != 0x006E6A6F: return None
            
            bpm = struct.unpack("<f", h[16:20])[0]
            level = struct.unpack("<h", h[24:26])[0]  # HX level
            title = h[108:172].split(b"\x00")[0].decode("cp949", errors="ignore").strip()
            artist = h[172:204].split(b"\x00")[0].decode("cp949", errors="ignore").strip()
            hard_off = struct.unpack("<i", h[292:296])[0]
            cover_off = struct.unpack("<i", h[296:300])[0]
            file_len = os.path.getsize(file_path)

            if hard_off < 300 or hard_off >= file_len:
                hard_off = 300
            if cover_off <= hard_off or cover_off > file_len:
                cover_off = file_len

            f.seek(hard_off)
            
            events = []
            while f.tell() < cover_off:
                chunk = f.read(8)
                if len(chunk) < 8: break
                m, ch, cnt = struct.unpack("<ihh", chunk)
                if cnt == 0: continue
                if ch in (0, 1):
                    for i in range(cnt):
                        val = struct.unpack("<f", f.read(4))[0]
                        if val > 0: events.append((m, i / cnt, ch, val, 0))
                else:
                    for i in range(cnt):
                        val, vol_pan, n_type = struct.unpack("<hBB", f.read(4))
                        if val != 0: events.append((m, i / cnt, ch, val, n_type % 4))
            
            if not events: return None
            events.sort(key=lambda e: (e[0], e[1], 0 if e[2] == 1 else 1))

            max_m = max(e[0] for e in events)
            ts = [1.0] * (max_m + 1)
            for m, pos, ch, val, _ in events:
                if ch == 0 and val > 1e-10: ts[m] = val

            cur_bpm = bpm
            last_m = 0
            last_pos = 0.0
            time_offset = 0.0
            raw_notes = []

            for m, pos, ch, val, flag in events:
                if ch == 1:
                    prev_bpm = cur_bpm
                    cur_bpm = val
                    for i in range(last_m, m):
                        time_offset += (1.0 - last_pos) * 240000.0 / prev_bpm * ts[i]
                        last_pos = 0.0
                        last_m = i + 1
                    delta = pos - last_pos
                    time_offset += delta * 240000.0 / prev_bpm * ts[m]
                    last_pos = pos
                    last_m = m
                    continue
                elif 2 <= ch <= 8:
                    for i in range(last_m, m):
                        time_offset += (1.0 - last_pos) * 240000.0 / cur_bpm * ts[i]
                        last_pos = 0.0
                        last_m = i + 1
                    delta = pos - last_pos
                    time_ms = time_offset + delta * 240000.0 / cur_bpm * ts[m]
                    col = ch - 2
                    raw_notes.append((int(round(time_ms)), col, flag))

            hit_objects = []
            pending_holds = {c: [] for c in range(7)}
            for t_ms, col, flag in raw_notes:
                if flag == 0: hit_objects.append((t_ms, t_ms, col))
                elif flag == 2: pending_holds[col].append(t_ms)
                elif flag == 3:
                    if pending_holds[col]: hit_objects.append((pending_holds[col].pop(0), t_ms, col))
                    else: hit_objects.append((t_ms, t_ms, col))
            for col, starts in pending_holds.items():
                for s in starts: hit_objects.append((s, s, col))

            if len(hit_objects) < 20 or level <= 1:
                return None

            hit_objects.sort(key=lambda o: (o[0], o[1], o[2]))
            first_t = hit_objects[0][0]
            last_t = max(max(o[0], o[1]) for o in hit_objects)
            duration_ms = max(0, last_t - first_t)
            t_frames = min((duration_ms // STEP_MS) + 1, MAX_FRAMES_CAP)

            grid = np.zeros((t_frames, 7), dtype=np.uint8)
            for s_ms, e_ms, col in hit_objects:
                s_f = (s_ms - first_t) // STEP_MS
                e_f = (e_ms - first_t) // STEP_MS
                if s_f >= t_frames: continue
                e_f = min(e_f, t_frames - 1)
                if e_f <= s_f: grid[s_f, col] = 1
                else:
                    grid[s_f, col] = 2
                    if e_f > s_f + 1: grid[s_f+1:e_f, col] = 3
                    grid[e_f, col] = 4

            grid_hash = hashlib.md5(grid.tobytes()).hexdigest()
            song_title = f"{artist} - {title}".strip(" -") if artist else title

            return {
                "source_type": source_type,
                "file_path": file_path,
                "folder": folder_name,
                "file_name": os.path.basename(file_path),
                "title": song_title,
                "version": f"[{source_type.upper()}] [{folder_name}] [{level}]",
                "level": level,
                "priority": priority,
                "note_count": len(hit_objects),
                "duration_ms": duration_ms,
                "frames": t_frames,
                "grid_hash": grid_hash,
                "grid": grid,
            }
    except Exception:
        return None


def clean_tokens(title):
    tokens = set(re.findall(r"[a-zA-Z0-9\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+", title.lower()))
    tokens.difference_update({"o2jam", "o2", "ver", "mix", "shd", "hd", "nm", "ez", "hard", "easy", "normal", "original"})
    return tokens


def are_charts_duplicate(c1, c2):
    # 1. Exact bitwise grid hash match
    if c1["grid_hash"] == c2["grid_hash"]:
        return True

    # 2. Must have identical note count
    if c1["note_count"] != c2["note_count"]:
        return False

    # 3. Must have very close duration (within 2 seconds = 100 frames)
    if abs(c1["duration_ms"] - c2["duration_ms"]) > 2000:
        return False

    # 4. Check title token overlap
    tok1 = clean_tokens(c1["title"])
    tok2 = clean_tokens(c2["title"])
    if tok1 and tok2 and (tok1 & tok2):
        return True

    # 5. Fallback: identical level rating and very tight duration tolerance (0.5s)
    if c1["level"] == c2["level"] and abs(c1["duration_ms"] - c2["duration_ms"]) <= 500:
        return True

    return False


# ── 3. Main Unified Pipeline ────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=== Step 2: Unified O2Jam Dataset Generation (osu! + Private + NSJAM) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_charts = []

    # 1. Scan osu! Songs
    if os.path.exists(SONGS_DIR):
        print(f"Scanning osu! Songs: {SONGS_DIR}...")
        osu_dirs = [d for d in os.listdir(SONGS_DIR) if d.startswith("[_O2Jam_]") and os.path.isdir(os.path.join(SONGS_DIR, d))]
        osu_files = []
        for d in osu_dirs:
            dp = os.path.join(SONGS_DIR, d)
            osu_files.extend([os.path.join(dp, f) for f in os.listdir(dp) if f.lower().endswith(".osu")])
        print(f"  Found {len(osu_files)} .osu files across {len(osu_dirs)} directories.")
        
        parsed_osu = 0
        for fp in osu_files:
            c = parse_osu_file(fp)
            if c:
                all_charts.append(c)
                parsed_osu += 1
        print(f"  Successfully parsed {parsed_osu} valid 7K charts from osu!.")

    # 2. Scan Private Server Folders (Native .ojn)
    for folder_name, folder_path in PRIVATE_SERVER_DIRS.items():
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name}: directory not found ({folder_path})")
            continue
        ojn_files = [f for f in os.listdir(folder_path) if f.startswith("o2ma") and f.lower().endswith(".ojn")]
        print(f"Scanning {folder_name}: {len(ojn_files)} .ojn files...")
        parsed_priv = 0
        for fn in ojn_files:
            c = parse_ojn_file(os.path.join(folder_path, fn), folder_name, source_type="private", priority=PRIORITY_PRIVATE)
            if c:
                all_charts.append(c)
                parsed_priv += 1
        print(f"  Successfully parsed {parsed_priv} valid charts from {folder_name}.")

    # 3. Scan NSJAM Folders (Native .ojn)
    for folder_name, folder_path in NSJAM_DIRS.items():
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name}: directory not found ({folder_path})")
            continue
        ojn_files = [f for f in os.listdir(folder_path) if f.startswith("o2ma") and f.lower().endswith(".ojn")]
        print(f"Scanning {folder_name}: {len(ojn_files)} .ojn files...")
        parsed_nsjam = 0
        for fn in ojn_files:
            c = parse_ojn_file(os.path.join(folder_path, fn), folder_name, source_type="nsjam", priority=PRIORITY_NSJAM)
            if c:
                all_charts.append(c)
                parsed_nsjam += 1
        print(f"  Successfully parsed {parsed_nsjam} valid charts from {folder_name}.")

    print(f"\nTotal candidate charts parsed across all sources: {len(all_charts):,}")

    # 4. Robust Deduplication by Note Count, Duration, and Title / Hash
    print(f"\nRunning robust deduplication across {len(all_charts):,} candidates...")
    all_charts.sort(key=lambda c: (c["priority"], -c["duration_ms"]))

    unique_charts = []
    discarded_duplicates = 0
    priority_counts = {PRIORITY_PRIVATE: 0, PRIORITY_DMJAM: 0, PRIORITY_NSJAM: 0, PRIORITY_O2EZ: 0}

    by_notes = {}
    for c in all_charts:
        candidates = by_notes.setdefault(c["note_count"], [])
        is_dup = False
        for existing in candidates:
            if are_charts_duplicate(c, existing):
                is_dup = True
                discarded_duplicates += 1
                break
        if not is_dup:
            candidates.append(c)
            unique_charts.append(c)
            priority_counts[c["priority"]] += 1

    print(f"Deduplication complete:")
    print(f"  Retained unique charts: {len(unique_charts):,}")
    print(f"  Discarded duplicates:   {discarded_duplicates:,}")
    print(f"  Source breakdown:")
    print(f"    - Private Server Native:           {priority_counts[PRIORITY_PRIVATE]:,} charts")
    print(f"    - osu! DMJAM Server ([D],[H]...):  {priority_counts[PRIORITY_DMJAM]:,} charts")
    print(f"    - NSJAM Native Server:             {priority_counts[PRIORITY_NSJAM]:,} charts")
    print(f"    - osu! O2EZ Server ([O2EZ]):       {priority_counts[PRIORITY_O2EZ]:,} charts")

    # 4. Pack into zero-copy continuous uint8 array
    print(f"\nPacking {len(unique_charts):,} charts into zero-copy NumPy archive...")
    total_frames = sum(c["frames"] for c in unique_charts)
    print(f"Total time frames: {total_frames:,} ({total_frames * 7 / 1048576:.2f} MB uncompressed)")

    flat_grids = np.empty(total_frames * 7, dtype=np.uint8)
    offsets = np.empty(len(unique_charts) + 1, dtype=np.int64)
    lengths = np.empty(len(unique_charts), dtype=np.int32)
    levels = np.empty(len(unique_charts), dtype=np.float32)
    durations = np.empty(len(unique_charts), dtype=np.int32)

    metadata = []
    current_offset = 0

    for i, c in enumerate(unique_charts):
        L = c["frames"]
        flat_grids[current_offset : current_offset + L * 7] = c["grid"].reshape(-1)
        offsets[i] = current_offset
        lengths[i] = L
        levels[i] = float(c["level"])
        durations[i] = int(c["duration_ms"])
        current_offset += L * 7

        metadata.append({
            "id": i,
            "source_type": c["source_type"],
            "title": c["title"],
            "version": c["version"],
            "level": c["level"],
            "priority": c["priority"],
            "folder": c["folder"],
            "file_name": c["file_name"],
            "note_count": c["note_count"],
            "duration_ms": c["duration_ms"],
            "frames": L,
            "grid_hash": c["grid_hash"],
        })

    offsets[len(unique_charts)] = current_offset

    np.savez_compressed(
        OUTPUT_NPZ,
        flat_grids=flat_grids,
        offsets=offsets,
        lengths=lengths,
        levels=levels,
        durations=durations,
    )
    print(f"Saved packed dataset to {OUTPUT_NPZ} ({os.path.getsize(OUTPUT_NPZ) / 1048576:.2f} MB)")

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {OUTPUT_META} ({os.path.getsize(OUTPUT_META) / 1048576:.2f} MB)")

    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
