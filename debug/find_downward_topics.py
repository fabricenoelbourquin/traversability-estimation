#!/usr/bin/env python3
"""
Scan mission bag files to find topics that look like downward-facing cameras.

Usage:
    python debug/find_downward_topics.py --mission ETH-1
    python debug/find_downward_topics.py --dir /path/to/raw/mission
    python debug/find_downward_topics.py --mission ETH-1 --keywords downward belly foot

By default it looks for topic names containing case-insensitive keywords:
    ["down", "belly", "foot", "stairs", "floor", "bottom"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader

THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.paths import get_paths
from utils.missions import resolve_mission
from utils.rosbag_tools import filter_valid_rosbags


DEFAULT_KEYWORDS = ["down", "belly", "foot", "floor", "bottom", "under"]
LIKELY_PATTERNS = [
    "*_hdr_front.bag",
    "*_hdr_left.bag",
    "*_hdr_right.bag",
    "*_anymal_depth_cameras.bag",
    "*_zed2i_images.bag",
    "*_zed2i_depth.bag",
]


def iter_candidate_bags(raw_dir: Path, patterns: list[str]) -> list[Path]:
    bags: list[Path] = []
    for pat in patterns:
        matches = sorted(raw_dir.glob(pat))
        if not matches:
            print(f"[warn] No bag matching pattern {pat!r} in {raw_dir}, skipping.")
            continue
        bags.extend(filter_valid_rosbags(matches))
    # deduplicate while preserving order
    seen = set()
    unique = []
    for bag in bags:
        if bag in seen:
            continue
        seen.add(bag)
        unique.append(bag)
    return unique


def list_topics(bag_path: Path) -> list[str]:
    topics: list[str] = []
    try:
        with AnyReader([bag_path]) as reader:
            topics = sorted({c.topic for c in reader.connections})
    except Exception as exc:
        print(f"[warn] Unable to read {bag_path.name}: {exc}")
    return topics


def resolve_raw_dir(args) -> Path:
    if args.dir:
        raw_dir = Path(args.dir).expanduser()
        if not raw_dir.exists():
            raise FileNotFoundError(f"{raw_dir} not found")
        return raw_dir
    if not args.mission:
        raise SystemExit("Provide --mission or --dir")
    paths = get_paths()
    mission = resolve_mission(args.mission, paths)
    return mission.raw


def main():
    ap = argparse.ArgumentParser(description="List downward camera topics in mission bags.")
    ap.add_argument("--mission", help="Mission alias/UUID (resolves to RAW directory)")
    ap.add_argument("--dir", help="Explicit directory containing .bag files (overrides --mission)")
    ap.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Substrings to match against topic names (case-insensitive).",
    )
    ap.add_argument(
        "--bag-patterns",
        nargs="*",
        default=LIKELY_PATTERNS,
        help="Glob patterns for likely camera bags (default: common HDR/ZED bags).",
    )
    args = ap.parse_args()

    raw_dir = resolve_raw_dir(args)
    patterns = args.bag_patterns if args.bag_patterns else LIKELY_PATTERNS
    bags = iter_candidate_bags(raw_dir, patterns)
    if not bags:
        print(f"[warn] No candidate bag files found in {raw_dir} (patterns={patterns})")
        return

    print(f"[info] listing topics for {len(bags)} bags in {raw_dir}")

    total_topics = 0
    for bag in bags:
        topics = list_topics(bag)
        if not topics:
            continue
        total_topics += len(topics)
        print(f"\n{bag.name}:")
        for topic in topics:
            print(f"  • {topic}")

    if total_topics == 0:
        print("[info] No topics collected. Inspect bags manually.")
    else:
        print(f"\n[ok] Listed {total_topics} topics from {len(bags)} candidate bags.")


if __name__ == "__main__":
    main()
