#!/usr/bin/env python3
"""Episode metadata annotation tool for LeRobot datasets.

Annotates each episode with:
  - speed: episode length discretized to 500-step bins
  - quality: task execution quality score (1-5)
  - mistake: whether the robot made a mistake (bool)

Usage:
    python scripts/annotate_metadata.py --data-dir /path/to/dataset
    python scripts/annotate_metadata.py --data-dir /path/to/dataset --auto-only
    python scripts/annotate_metadata.py --data-dir /path/to/dataset --from-csv metadata.csv
    python scripts/annotate_metadata.py --data-dir /path/to/dataset --batch-quality 5 --batch-mistake false
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


def load_episode_info(data_dir: str) -> list[dict]:
    """Load episode metadata from episodes.jsonl."""
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def compute_speed_bin(episode_length: int, bin_size: int = 500) -> int:
    """Discretize episode length to speed bin.

    Following pi0.7: values in an interval of 500 steps,
    e.g., 1750-2250 are binned to "2000 steps".
    """
    bin_center = round(episode_length / bin_size) * bin_size
    return max(bin_size, bin_center)  # minimum 1 bin


def auto_compute_metadata(episodes: list[dict]) -> dict[str, dict]:
    """Auto-compute metadata for all episodes.

    Auto-computes:
      - speed: from episode length
    Sets defaults:
      - quality: 5 (highest, assuming expert demonstrations)
      - mistake: false (assuming clean demonstrations)
    """
    metadata = {}
    for ep in episodes:
        ep_idx = str(ep["episode_index"])
        length = ep["length"]
        metadata[ep_idx] = {
            "speed": compute_speed_bin(length),
            "quality": 5,
            "mistake": False,
            "length": length,
        }
    return metadata


def annotate_episode_interactive(
    ep_idx: int, ep_length: int, existing: dict | None = None
) -> dict:
    """Interactively annotate metadata for a single episode."""
    speed = compute_speed_bin(ep_length)
    default_quality = existing.get("quality", 5) if existing else 5
    default_mistake = existing.get("mistake", False) if existing else False

    print(f"\n  Episode {ep_idx}: {ep_length} frames ({ep_length/30:.1f}s), speed_bin={speed}")

    # Quality
    while True:
        q_input = input(f"  Quality (1-5) [{default_quality}]: ").strip()
        if q_input == "":
            quality = default_quality
            break
        try:
            quality = int(q_input)
            if 1 <= quality <= 5:
                break
            print("  Must be 1-5")
        except ValueError:
            print("  Enter a number 1-5")

    # Mistake
    while True:
        m_input = input(f"  Mistake? (y/n) [{'y' if default_mistake else 'n'}]: ").strip().lower()
        if m_input == "":
            mistake = default_mistake
            break
        if m_input in ("y", "yes", "true", "1"):
            mistake = True
            break
        if m_input in ("n", "no", "false", "0"):
            mistake = False
            break
        print("  Enter y or n")

    return {
        "speed": speed,
        "quality": quality,
        "mistake": mistake,
        "length": ep_length,
    }


def import_from_csv(csv_path: str, episodes: list[dict]) -> dict[str, dict]:
    """Import metadata from CSV.

    CSV format: episode_idx, quality, mistake
    Speed is auto-computed from episode length.
    """
    ep_lengths = {ep["episode_index"]: ep["length"] for ep in episodes}
    metadata = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_idx = int(row["episode_idx"])
            ep_key = str(ep_idx)
            length = ep_lengths.get(ep_idx, 0)

            quality = int(row.get("quality", 5))
            mistake_str = row.get("mistake", "false").strip().lower()
            mistake = mistake_str in ("true", "1", "yes", "y")

            metadata[ep_key] = {
                "speed": compute_speed_bin(length),
                "quality": quality,
                "mistake": mistake,
                "length": length,
            }

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Annotate episode metadata for LeRobot dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to LeRobot dataset directory")
    parser.add_argument("--auto-only", action="store_true", help="Only auto-compute (speed from length, quality=5, mistake=false)")
    parser.add_argument("--from-csv", type=str, default=None, help="Import from CSV file")
    parser.add_argument("--batch-quality", type=int, default=None, help="Set quality for all episodes (1-5)")
    parser.add_argument("--batch-mistake", type=str, default=None, help="Set mistake for all episodes (true/false)")
    parser.add_argument("--episode", type=int, default=None, help="Annotate specific episode only")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.output or os.path.join(data_dir, "episode_metadata.json")

    # Load episodes
    episodes = load_episode_info(data_dir)
    print(f"Loaded {len(episodes)} episodes from {data_dir}")

    # Print summary
    lengths = [ep["length"] for ep in episodes]
    print(f"  Length range: {min(lengths)}-{max(lengths)} frames")
    print(f"  Average length: {sum(lengths)/len(lengths):.0f} frames ({sum(lengths)/len(lengths)/30:.1f}s)")

    # Load existing
    metadata = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} existing annotations from {output_path}")

    if args.from_csv:
        csv_metadata = import_from_csv(args.from_csv, episodes)
        metadata.update(csv_metadata)
        print(f"Imported {len(csv_metadata)} episodes from CSV")

    elif args.auto_only:
        metadata = auto_compute_metadata(episodes)
        print(f"Auto-computed metadata for {len(metadata)} episodes")

    elif args.batch_quality is not None or args.batch_mistake is not None:
        # Batch mode with specified values
        for ep in episodes:
            ep_key = str(ep["episode_index"])
            existing = metadata.get(ep_key, {})
            entry = {
                "speed": compute_speed_bin(ep["length"]),
                "quality": args.batch_quality if args.batch_quality is not None else existing.get("quality", 5),
                "mistake": (args.batch_mistake.lower() in ("true", "1", "yes")) if args.batch_mistake is not None else existing.get("mistake", False),
                "length": ep["length"],
            }
            metadata[ep_key] = entry
        print(f"Batch-set metadata for {len(episodes)} episodes")
        if args.batch_quality is not None:
            print(f"  quality={args.batch_quality}")
        if args.batch_mistake is not None:
            print(f"  mistake={args.batch_mistake}")

    else:
        # Interactive annotation
        ep_indices = [args.episode] if args.episode is not None else [ep["episode_index"] for ep in episodes]

        print(f"\nAnnotating {len(ep_indices)} episodes interactively...")
        print("(Press Ctrl+C to stop and save progress)\n")

        try:
            for ep_idx in ep_indices:
                ep_info = next((e for e in episodes if e["episode_index"] == ep_idx), None)
                if ep_info is None:
                    print(f"Episode {ep_idx} not found, skipping")
                    continue

                ep_key = str(ep_idx)
                existing = metadata.get(ep_key, None)

                if existing and args.episode is None:
                    print(f"  Episode {ep_idx} already annotated, skipping (use --episode to re-annotate)")
                    continue

                entry = annotate_episode_interactive(ep_idx, ep_info["length"], existing)
                metadata[ep_key] = entry

                # Save after each episode
                with open(output_path, "w") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving progress...")

    # Final save
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nMetadata saved to {output_path}")
    print(f"Total annotated episodes: {len(metadata)}")

    if metadata:
        qualities = [v["quality"] for v in metadata.values()]
        mistakes = [v["mistake"] for v in metadata.values()]
        speeds = [v["speed"] for v in metadata.values()]
        print(f"  Quality distribution: " + ", ".join(f"{q}: {qualities.count(q)}" for q in sorted(set(qualities))))
        print(f"  Mistake rate: {sum(mistakes)}/{len(mistakes)} ({100*sum(mistakes)/len(mistakes):.1f}%)")
        print(f"  Speed bins: {sorted(set(speeds))}")


if __name__ == "__main__":
    main()
