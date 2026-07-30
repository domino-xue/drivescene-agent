import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import find_scenario_files, load_scenario
from drivescene.detectors.close_following import detect_close_following
from drivescene.maps.overlay import load_map_overlay
from drivescene.viz.trajectory import render_trajectory_plot


def main() -> None:
    root = Path("data/val")
    output = Path("outputs/close_following_events.jsonl")
    plot_dir = Path("outputs/plots/close_following")
    output.parent.mkdir(parents=True, exist_ok=True)

    events = []
    for path in find_scenario_files(root, limit=500):
        df = load_scenario(path)
        scenario_id = path.parent.name
        map_path = path.parent / f"log_map_archive_{scenario_id}.json"
        map_overlay = load_map_overlay(map_path) if map_path.exists() else None
        events.extend(detect_close_following(df, map_overlay=map_overlay))

    events.sort(key=lambda event: (event["min_ttc_s"], event["min_front_distance_m"]))
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
            plot_dir / f"{scenario_id}_{event['actor_id']}.png",
            title=f"{scenario_id} close following {event['actor_id']}",
            highlight_track_id=event["track_id"],
        )

    print(f"Wrote {len(events)} close following events to {output}")
    print(f"Wrote top plots to {plot_dir}")


if __name__ == "__main__":
    main()
