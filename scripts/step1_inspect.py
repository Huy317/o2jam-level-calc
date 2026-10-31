"""
Step 1: Inspect O2Jam 7K beatmaps in osu! Songs folder.
Strictly READONLY inspection of [_O2Jam_]* folders.
"""

import os
import re
import sys
from collections import Counter, defaultdict

# Ensure clean UTF-8 terminal output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SONGS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OSU_SONGS_DIR", r"D:\osu!\Songs")
LEVEL_REGEX = re.compile(r"\[(\d+)\]\s*$")
SERVER_TAG_REGEX = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")


def inspect():
    if not os.path.exists(SONGS_DIR):
        print(f"Error: Songs directory not found: {SONGS_DIR}")
        print("Usage: python scripts/step1_inspect.py <path_to_songs_folder>")
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
            levels_le_1.append((file_path, level, version_line))

        # Check server / difficulty tags
        tags = SERVER_TAG_REGEX.findall(version_line)
        for t in tags:
            tag_upper = t.upper()
            if tag_upper in ("O2JAM", "O2EZ", "DMJAM", "NSJAM"):
                server_tags[tag_upper] += 1
            elif tag_upper in ("E", "N", "H", "D", "SH", "HD", "EX"):
                difficulty_tags[tag_upper] += 1

    print(f"\n=== Verification Statistics ===")
    print(f"Valid Mania 7K files: {len(osu_files) - len(non_mania) - len(non_7k)} / {len(osu_files)}")
    if non_mania:
        print(f"Non-Mania files (Mode != 3): {len(non_mania)}")
    if non_7k:
        print(f"Non-7K files (CircleSize != 7): {len(non_7k)}")

    print(f"\nRegex Matches on Version line: {matched_regex} / {len(osu_files)} ({matched_regex / max(len(osu_files), 1) * 100:.1f}%)")
    if failed_regex:
        print(f"Failed to match ({len(failed_regex)} files):")
        for f, reason in failed_regex[:10]:
            print(f"  - {os.path.basename(f)}: {reason}")
        if len(failed_regex) > 10:
            print(f"  ... and {len(failed_regex) - 10} more.")

    if level_list:
        valid_levels = [lvl for lvl in level_list if lvl > 1]
        print(f"\n=== Level Distribution ===")
        print(f"Total parsed levels: {len(level_list)}")
        print(f"Levels <= 1 (gimmicks/unrated to exclude): {len(levels_le_1)}")
        print(f"Valid levels (> 1): {len(valid_levels)}")
        if valid_levels:
            print(f"Min level: {min(valid_levels)}")
            print(f"Max level: {max(valid_levels)}")
            print(f"Average level: {sum(valid_levels) / len(valid_levels):.2f}")

    print(f"\n=== Tag Distributions ===")
    print("Server Tags in Version line:")
    for tag, count in server_tags.most_common():
        print(f"  [{tag}]: {count}")

    print("Difficulty Tags in Version line:")
    for tag, count in difficulty_tags.most_common():
        print(f"  [{tag}]: {count}")


if __name__ == "__main__":
    inspect()
