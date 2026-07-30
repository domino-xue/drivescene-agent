import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_review_queue(
    event_files: list[Path],
    per_type_limit: int = 100,
    reviewed_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    reviewed_by_identity = build_review_lookup(reviewed_items or [])
    by_type: dict[str, list[dict[str, Any]]] = {}
    for path in event_files:
        for event in load_jsonl(path):
            by_type.setdefault(str(event["event_type"]), []).append(event)

    queue: list[dict[str, Any]] = []
    for event_type in sorted(by_type):
        ranked = sorted(by_type[event_type], key=_risk_sort_key)
        for event in ranked[:per_type_limit]:
            item = _with_review_fields(event)
            matching_review = reviewed_by_identity.get(event_identity(item))
            if matching_review is not None:
                item["review_status"] = matching_review.get("review_status", "reviewed")
                item["review"] = matching_review.get("review", item["review"])
            queue.append(item)

    for index, item in enumerate(queue, start=1):
        item["review_id"] = f"{index:06d}"
    return queue


def build_review_lookup(
    reviewed_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in reviewed_items:
        lookup[event_identity(item)] = item
    return lookup


def event_identity(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("scenario_id", "")),
            str(item.get("event_type", "")),
            str(item.get("track_id", "")),
            str(item.get("actor_id", "")),
            str(item.get("start_timestep", "")),
            str(item.get("end_timestep", "")),
        ]
    )


def _risk_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    if event["event_type"] == "hard_braking":
        return (float(event.get("min_acceleration_mps2", 0.0)), 0.0)
    if event["event_type"] == "close_following":
        front_distance = event.get("min_front_distance_m", event.get("min_distance_m", 0.0))
        return (float(event.get("min_ttc_s", float("inf"))), float(front_distance))
    if event["event_type"] == "cut_in":
        return (
            float(event.get("min_ttc_s", float("inf"))),
            float(event.get("min_front_distance_m", float("inf"))),
        )
    if event["event_type"] == "stopped_vehicle_ahead":
        return (
            float(event.get("min_ttc_s", float("inf"))),
            float(event.get("min_front_distance_m", float("inf"))),
        )
    if event["event_type"] == "lane_change":
        return (-float(event.get("lateral_displacement_m", 0.0)), 0.0)
    return (0.0, 0.0)


def _with_review_fields(event: dict[str, Any]) -> dict[str, Any]:
    item = dict(event)
    item["review_status"] = "unreviewed"
    item["review"] = {
        "is_valid_event": None,
        "correct_event_type": item.get("event_type"),
        "severity": None,
        "correct_start_timestep": item.get("start_timestep"),
        "correct_end_timestep": item.get("end_timestep"),
        "false_positive_reason": None,
        "review_note": "",
    }
    return item
