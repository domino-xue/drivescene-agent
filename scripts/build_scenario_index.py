"""Build a scenario index over every local AV2 scenario (no default sampling)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import find_scenario_files
from drivescene.evidence.index import build_scenario_index, write_scenario_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/val"))
    parser.add_argument("--output", type=Path, default=Path("outputs/scenario_index_full.parquet"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    files = find_scenario_files(args.root, limit=args.limit)
    if not files:
        parser.error(f"No scenario parquet files found in {args.root}")
    print(f"Indexing {len(files)} real scenario files", flush=True)
    chunks = [files[start:start + 50] for start in range(0, len(files), 50)]
    frames = []
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for frame in pool.map(build_scenario_index, chunks):
            frames.append(frame)
            processed += len(frame)
            if processed % 500 == 0 or processed == len(files):
                print(f"Indexed {processed}/{len(files)}", flush=True)
    index = pd.concat(frames, ignore_index=True)
    for column in ("scenario_path", "map_path"):
        index[column] = index[column].map(
            lambda value: str(value).replace("\\", "/") if value is not None else None
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_scenario_index(index, args.output)
    print(f"Wrote {len(index)} scenarios to {args.output}", flush=True)


if __name__ == "__main__":
    main()
