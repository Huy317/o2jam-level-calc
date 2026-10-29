"""
Step 1: Inspect O2Jam 7K beatmaps in D:\\osu!\\Songs.
Strictly READONLY inspection of [_O2Jam_]* folders.
"""

import os
import re
import sys
from collections import Counter, defaultdict

# Ensure clean UTF-8 terminal output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SONGS_DIR = r"D:\osu!\Songs"
LEVEL_REGEX = re.compile(r"\[(\d+)\]\s*$")
SERVER_TAG_REGEX = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")


def inspect():
    if not os.path.exists(SONGS_DIR):
        print(f"Error: Songs directory not found: {SONGS_DIR}")
        return

    o2jam_dirs = [
        d for d in os.listdir(SONGS_DIR)
        if d.startswith("[_O2Jam_]") and os.path.isdir(os.path.join(SONGS_DIR, d))
    ]

    total_dirs = len(o2jam_dirs)
    osu_files = []
    empty_dirs = []

    for d in o2jam_dirs:
        dir_path = os.path.join(SONGS_DIR, d)
        found = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(".osu")
        ]
        if not found:
            empty_dirs.append(d)
        else:
            osu_files.extend(found)

    print(f"=== O2Jam Songs Scan Summary ===")
    print(f"Total [_O2Jam_]* directories found: {total_dirs}")
    print(f"Total .osu chart files found: {len(osu_files)}")
    if empty_dirs:
        print(f"Directories without .osu files ({len(empty_dirs)}):")
        for ed in empty_dirs:
            print(f"  - {ed}")

    matched_regex = 0
    failed_regex = []
    non_7k = []
    non_mania = []
    level_list = []
    server_tags = Counter()
    difficulty_tags = Counter()
    levels_le_1 = []

    for file_path in osu_files:
        mode = None
        circle_size = None
        version_line = None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Mode:"):
                        mode = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("CircleSize:"):
                        circle_size = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("Version:"):
                        version_line = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("[HitObjects]"):
                        break
        except Exception as e:
            failed_regex.append((file_path, f"Read error: {e}"))
            continue

        if mode != "3":
            non_mania.append((file_path, mode))
        if circle_size != "7":
            non_7k.append((file_path, circle_size))

        if not version_line:
            failed_regex.append((file_path, "Missing Version line"))
            continue

        match = LEVEL_REGEX.search(version_line)
        if not match:
            failed_regex.append((file_path, f"No level bracket match: '{version_line}'"))
            continue

        matched_regex += 1
        level = int(match.group(1))
        level_list.append(level)

        if level <= 1:
            levels_le_1.append((file_path, version_line, level))

        # Extract all bracket tags before the level bracket
        prefix = version_line[:match.start()].strip()
        tags = SERVER_TAG_REGEX.findall(prefix)
        # Full prefix grouping
        server_tags[prefix] += 1
        # Individual tag breakdown (e.g. O2EZ, H, N, E, D)
        for t in tags:
            difficulty_tags[t] += 1

    print(f"\n=== Regex & Format Validation ===")
    print(f"Charts matching level regex `\\[(\\d+)\\]\\s*$`: {matched_regex} / {len(osu_files)}")
    print(f"Non-Mania (Mode != 3): {len(non_mania)}")
    print(f"Non-7K (CircleSize != 7): {len(non_7k)}")
    print(f"Failed to parse / unparseable files: {len(failed_regex)}")
    for fp, reason in failed_regex[:10]:
        print(f"  - {os.path.basename(fp)}: {reason}")

    print(f"\n=== Level Statistics ===")
    if level_list:
        print(f"Min Level: {min(level_list)}, Max Level: {max(level_list)}")
        valid_levels = [l for l in level_list if l > 1]
        print(f"Levels <= 1 (gimmick / unrated to be skipped): {len(levels_le_1)}")
        if valid_levels:
            print(f"Valid Levels (> 1): count={len(valid_levels)}, min={min(valid_levels)}, max={max(valid_levels)}, avg={sum(valid_levels)/len(valid_levels):.2f}")

    print(f"\n=== Server / Difficulty Tag Breakdown (Raw Prefix) ===")
    for tag, count in server_tags.most_common():
        print(f"  {tag:<30} : {count:>5} charts")

    print(f"\n=== Extracted Bracket Tags Breakdown ===")
    for tag, count in difficulty_tags.most_common():
        print(f"  [{tag}]: {count} occurrences")


if __name__ == "__main__":
    inspect()
