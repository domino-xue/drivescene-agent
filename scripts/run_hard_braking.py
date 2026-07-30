import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import find_scenario_files, load_scenario
from drivescene.detectors.hard_braking import detect_hard_braking
from drivescene.viz.trajectory import render_trajectory_plot


def main() -> None:
    root = Path("data/val")
    output = Path("outputs/hard_braking_events.jsonl")
    plot_dir = Path("outputs/plots/hard_braking")
    output.parent.mkdir(parents=True, exist_ok=True)

    events = []
    for path in find_scenario_files(root, limit=500):
        df = load_scenario(path)
        events.extend(detect_hard_braking(df))

    events.sort(key=lambda event: event["min_acceleration_mps2"])
    with output.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=True) + "\n")

    for event in events[:10]:
        scenario_id = event["scenario_id"]
        scene_file = root / scenario_id / f"scenario_{scenario_id}.parquet"
        if not scene_file.exists():
            continue
        df = load_scenario(scene_file)
        render_trajectory_plot(
            df,
            plot_dir / f"{scenario_id}.png",
            title=f"{scenario_id} hard braking",
            highlight_track_id=event["track_id"],
        )

    print(f"Wrote {len(events)} hard braking events to {output}")
    print(f"Wrote top plots to {plot_dir}")


if __name__ == "__main__":
    main()
