import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.data.loader import find_scenario_files, load_scenario
from drivescene.detectors.cut_in import detect_cut_in
from drivescene.detectors.lane_change import detect_lane_change
from drivescene.detectors.stopped_vehicle_ahead import detect_stopped_vehicle_ahead
from drivescene.maps.overlay import load_map_overlay
from drivescene.viz.trajectory import render_trajectory_plot


DETECTORS = {
    "cut_in": detect_cut_in,
    "stopped_vehicle_ahead": detect_stopped_vehicle_ahead,
    "lane_change": detect_lane_change,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine additional scene-label candidates.")
    parser.add_argument("--root", default="data/val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--event-type", choices=sorted(DETECTORS), default=None)
    parser.add_argument("--plots", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    selected = {args.event_type: DETECTORS[args.event_type]} if args.event_type else DETECTORS
    outputs = {event_type: Path(f"outputs/{event_type}_events.jsonl") for event_type in selected}
    plot_dirs = {event_type: Path("outputs/plots") / event_type for event_type in selected}
    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)

    events_by_type: dict[str, list[dict]] = {event_type: [] for event_type in selected}
    files = find_scenario_files(root, limit=args.limit)
    for index, path in enumerate(files, start=1):
        df = load_scenario(path)
        for event_type, detector in selected.items():
            events_by_type[event_type].extend(_detect_with_optional_map(detector, df, path))
        if index % 50 == 0:
            counts = {event_type: len(events) for event_type, events in events_by_type.items()}
            print(f"Processed {index}/{len(files)} scenarios: {counts}", flush=True)

    for event_type, events in events_by_type.items():
        events.sort(key=_risk_sort_key)
        with outputs[event_type].open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=True) + "\n")
        _write_top_plots(root, events[: args.plots], plot_dirs[event_type])
        print(f"Wrote {len(events)} {event_type} events to {outputs[event_type]}")


def _detect_with_optional_map(detector, df, scenario_path: Path) -> list[dict]:
    map_path = scenario_path.parent / f"log_map_archive_{_scenario_id_from_path(scenario_path)}.json"
    if map_path.exists():
        try:
            return detector(df, map_overlay=load_map_overlay(map_path))
        except TypeError:
            return detector(df)
    return detector(df)


def _scenario_id_from_path(scenario_path: Path) -> str:
    return scenario_path.stem.removeprefix("scenario_")


def _risk_sort_key(event: dict) -> tuple[float, float]:
    event_type = event["event_type"]
    if event_type == "cut_in":
        return (
            float(event.get("min_ttc_s", float("inf"))),
            float(event.get("min_front_distance_m", float("inf"))),
        )
    if event_type == "stopped_vehicle_ahead":
        return (
            float(event.get("min_ttc_s", float("inf"))),
            float(event.get("min_front_distance_m", float("inf"))),
        )
    if event_type == "lane_change":
        return (-float(event.get("lateral_displacement_m", 0.0)), 0.0)
    return (0.0, 0.0)


def _write_top_plots(root: Path, events: list[dict], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        scenario_id = event["scenario_id"]
        scene_file = root / scenario_id / f"scenario_{scenario_id}.parquet"
        if not scene_file.exists():
            continue
        df = load_scenario(scene_file)
        suffix = f"_{event['actor_id']}" if event.get("actor_id") else ""
        render_trajectory_plot(
            df,
            plot_dir / f"{scenario_id}{suffix}.png",
            title=f"{scenario_id} {event['event_type']}",
            highlight_track_id=event["track_id"],
        )


if __name__ == "__main__":
    main()
