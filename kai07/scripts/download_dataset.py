#!/usr/bin/env python3
"""
Download the Kai0 dataset from Hugging Face to the repo's ./data directory.

Run from the repository root:
    python scripts/download_dataset.py

Optional: download only specific tasks or set a custom output path:
    python scripts/download_dataset.py --tasks FlattenFold HangCloth --local-dir ./my_data
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Process
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root (directory containing .git)."""
    path = Path(__file__).resolve().parent.parent
    if (path / ".git").exists():
        return path
    # Fallback: assume cwd is repo root
    return Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Kai0 dataset from Hugging Face to ./data (or --local-dir)."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Directory to save the dataset (default: <repo_root>/data)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["FlattenFold", "HangCloth", "TeeShirtSort"],
        default=None,
        help="Download only these tasks (default: all)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="OpenDriveLab-org/Kai0",
        help="Hugging Face dataset repo id (default: OpenDriveLab-org/Kai0)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub", file=sys.stderr)
        return 1

    repo_root = get_repo_root()
    local_dir = Path(args.local_dir) if args.local_dir else repo_root / "data"
    local_dir = local_dir.resolve()

    allow_patterns = None
    if args.tasks:
        allow_patterns = [f"{t}/*" for t in args.tasks]
        allow_patterns.append("README.md")
        allow_patterns.append("meta/*")

    print(f"Downloading dataset to {local_dir}")
    print(f"Repo: {args.repo_id}" + (f", tasks: {args.tasks}" if args.tasks else " (all tasks)"))

    # Run snapshot_download in a separate process so Ctrl+C in the main process
    # can reliably terminate the download, even if the library swallows signals.
    def _worker():
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

    proc = Process(target=_worker)
    proc.start()

    try:
        proc.join()
    except KeyboardInterrupt:
        print(
            "\nDownload interrupted by user (Ctrl+C). Terminating download process...",
            file=sys.stderr,
        )
        proc.terminate()
        proc.join()
        print("Partial data may remain in:", local_dir, file=sys.stderr)
        return 130

    if proc.exitcode != 0:
        print(f"\nDownload process exited with code {proc.exitcode}", file=sys.stderr)
        return proc.exitcode or 1

    print(f"\nDone. Dataset is at: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
