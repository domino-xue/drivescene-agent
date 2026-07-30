from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import load_scenario
from drivescene.maps.overlay import load_map_overlay
from drivescene.review.assets import assets_match_event, render_review_assets
from drivescene.review.queue import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    queue = load_jsonl(Path("outputs/review_queue.jsonl"))
    if args.limit is not None:
        queue = queue[: args.limit]
    root = Path("data/val")
    output_root = Path("outputs/review_assets")
    built = 0

    for item in queue:
        asset_dir = output_root / item["review_id"]
        if (
            not args.force
            and (asset_dir / "animation.gif").exists()
            and assets_match_event(asset_dir, item)
        ):
            continue
        scenario_id = item["scenario_id"]
        scene_file = root / scenario_id / f"scenario_{scenario_id}.parquet"
        if not scene_file.exists():
            print(f"Missing scenario file: {scene_file}")
            continue
        map_file = root / scenario_id / f"log_map_archive_{scenario_id}.json"
        map_overlay = load_map_overlay(map_file) if map_file.exists() else None
        df = load_scenario(scene_file)
        render_review_assets(df, item, output_root, map_overlay=map_overlay)
        built += 1
        if built % 10 == 0:
            print(f"Built {built} review asset sets")

    print(f"Built {built} new review asset sets in {output_root}")


if __name__ == "__main__":
    main()
