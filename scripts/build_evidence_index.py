from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import find_scenario_files
from drivescene.evidence.index import (
    build_event_index,
    build_scenario_index,
    write_event_index,
    write_scenario_index,
)
from drivescene.review.queue import load_jsonl


def main() -> None:
    queue = load_jsonl(Path("outputs/review_queue.jsonl"))
    reviewed = load_jsonl(Path("outputs/reviewed_events.jsonl"))
    index = build_event_index(queue, reviewed, asset_root=Path("outputs/review_assets"))

    write_event_index(index, Path("outputs/event_index.parquet"))
    labeled = index[index["is_valid_event"].notna()].copy()
    write_event_index(labeled, Path("outputs/labeled_event_index.parquet"))
    scenario_index = build_scenario_index(find_scenario_files(Path("data/val"), limit=500))
    write_scenario_index(scenario_index, Path("outputs/scenario_index.parquet"))

    print(f"Wrote {len(index)} rows to outputs/event_index.parquet")
    print(f"Wrote {len(labeled)} labeled rows to outputs/labeled_event_index.parquet")
    print(f"Wrote {len(scenario_index)} rows to outputs/scenario_index.parquet")


if __name__ == "__main__":
    main()
