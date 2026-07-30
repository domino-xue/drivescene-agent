from pathlib import Path
from typing import Any

import pandas as pd

from drivescene.review.queue import event_identity


def build_event_index(
    queue: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    asset_root: Path,
) -> pd.DataFrame:
    reviewed_by_identity: dict[str, list[dict[str, Any]]] = {}
    for item in reviewed:
        reviewed_by_identity.setdefault(event_identity(item), []).append(item)

    rows = []
    for item in queue:
        matching_review = _matching_review(item, reviewed_by_identity.get(event_identity(item), []))
        rows.append(_event_row(item, matching_review, asset_root))
    index = pd.DataFrame(rows)
    for column in ["is_valid_event", "has_animation", "has_metrics", "has_trajectory"]:
        if column in index.columns:
            index[column] = index[column].astype(object)
    return index


def write_event_index(index: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(output_path, index=False)


def build_scenario_index(scenario_files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in scenario_files:
        df = pd.read_parquet(path)
        scenario_id = str(df["scenario_id"].iloc[0]) if "scenario_id" in df.columns and not df.empty else ""
        map_path = path.parent / f"log_map_archive_{scenario_id}.json"
        rows.append(
            {
                "scenario_id": scenario_id,
                "city": _first_value(df, "city"),
                "num_rows": len(df),
                "num_tracks": int(df["track_id"].nunique()) if "track_id" in df.columns else 0,
                "num_timestamps": int(df["timestep"].nunique()) if "timestep" in df.columns else 0,
                "min_timestep": int(df["timestep"].min()) if "timestep" in df.columns else None,
                "max_timestep": int(df["timestep"].max()) if "timestep" in df.columns else None,
                "object_types": _joined_unique(df, "object_type"),
                "has_map": bool(map_path.exists()),
                "scenario_path": str(path),
                "map_path": str(map_path) if map_path.exists() else None,
            }
        )
    index = pd.DataFrame(rows)
    if "has_map" in index.columns:
        index["has_map"] = index["has_map"].astype(object)
    return index


def write_scenario_index(index: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(output_path, index=False)


def _matching_review(
    queue_item: dict[str, Any],
    reviewed_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for reviewed_item in reversed(reviewed_items):
        if event_identity(queue_item) == event_identity(reviewed_item):
            return reviewed_item
    return None


def _event_row(
    item: dict[str, Any],
    reviewed_item: dict[str, Any] | None,
    asset_root: Path,
) -> dict[str, Any]:
    review = (reviewed_item or {}).get("review", {})
    asset_dir = asset_root / str(item["review_id"])
    evidence = item.get("evidence", {})

    row: dict[str, Any] = {
        "review_id": str(item.get("review_id", "")),
        "event_identity": event_identity(item),
        "scenario_id": str(item.get("scenario_id", "")),
        "city": item.get("city"),
        "event_type": str(item.get("event_type", "")),
        "track_id": str(item.get("track_id", "")),
        "actor_id": str(item.get("actor_id", "")),
        "start_timestep": item.get("start_timestep"),
        "end_timestep": item.get("end_timestep"),
        "duration_s": evidence.get("duration_s"),
        "min_distance_m": item.get("min_distance_m"),
        "min_front_distance_m": item.get(
            "min_front_distance_m",
            evidence.get("min_front_distance_m"),
        ),
        "min_ttc_s": item.get("min_ttc_s"),
        "min_lateral_offset_m": evidence.get("min_lateral_offset_m"),
        "max_closing_speed_mps": evidence.get("max_closing_speed_mps"),
        "min_heading_alignment": evidence.get("min_heading_alignment"),
        "max_front_angle_deg": evidence.get("max_front_angle_deg"),
        "min_acceleration_mps2": item.get("min_acceleration_mps2"),
        "speed_before_mps": evidence.get("speed_before_mps"),
        "speed_after_mps": evidence.get("speed_after_mps"),
        "min_velocity_acceleration_mps2": evidence.get("min_velocity_acceleration_mps2"),
        "min_position_acceleration_mps2": evidence.get("min_position_acceleration_mps2"),
        "is_valid_event": review.get("is_valid_event"),
        "correct_event_type": review.get("correct_event_type"),
        "severity": review.get("severity"),
        "correct_start_timestep": review.get("correct_start_timestep"),
        "correct_end_timestep": review.get("correct_end_timestep"),
        "false_positive_reason": review.get("false_positive_reason"),
        "review_note": review.get("review_note"),
        "review_status": (reviewed_item or {}).get("review_status", "unreviewed"),
        "asset_dir": str(asset_dir),
        "animation_path": str(asset_dir / "animation.gif"),
        "metrics_path": str(asset_dir / "metrics.png"),
        "trajectory_path": str(asset_dir / "trajectory_window.png"),
    }
    row["has_animation"] = Path(row["animation_path"]).exists()
    row["has_metrics"] = Path(row["metrics_path"]).exists()
    row["has_trajectory"] = Path(row["trajectory_path"]).exists()
    return row


def _first_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns or df.empty:
        return None
    return df[column].dropna().iloc[0] if not df[column].dropna().empty else None


def _joined_unique(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    values = sorted(str(value) for value in df[column].dropna().unique())
    return ",".join(values)
