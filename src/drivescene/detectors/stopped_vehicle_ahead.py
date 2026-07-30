from collections.abc import Iterable
from typing import Any

import math
import pandas as pd

from drivescene.data.loader import get_focal_track
from drivescene.features.relative_motion import compute_relative_motion
from drivescene.features.path_geometry import future_path_match
from drivescene.maps.overlay import MapOverlay, nearest_lane_index


def detect_stopped_vehicle_ahead(
    df: pd.DataFrame,
    actor_speed_threshold_mps: float = 1.0,
    focal_speed_threshold_mps: float = 3.0,
    distance_threshold_m: float = 25.0,
    lateral_threshold_m: float = 2.0,
    path_lateral_threshold_m: float = 2.0,
    path_lookahead_distance_m: float = 35.0,
    min_duration_s: float = 0.5,
    sample_period_s: float = 0.1,
    allowed_actor_types: Iterable[str] | None = None,
    map_overlay: MapOverlay | None = None,
    lane_assignment_threshold_m: float = 1.5,
    lane_constrained_actor_types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rel = compute_relative_motion(df)
    if rel.empty:
        return []

    speeds = _speed_table(df)
    rel = rel.merge(
        speeds.rename(columns={"track_id": "actor_id", "speed_mps": "actor_speed_mps"}),
        on=["actor_id", "timestep"],
        how="left",
    )
    rel = rel.merge(
        speeds.rename(columns={"track_id": "track_id", "speed_mps": "focal_speed_mps"}),
        on=["track_id", "timestep"],
        how="left",
    )

    allowed_types = (
        {"vehicle", "bus", "motorcyclist", "cyclist", "riderless_bicycle"}
        if allowed_actor_types is None
        else {str(actor_type).lower() for actor_type in allowed_actor_types}
    )
    candidates = rel[
        (rel["actor_type"].str.lower().isin(allowed_types))
        & (rel["front_projection_m"] > 0)
        & (rel["front_projection_m"] <= distance_threshold_m)
        & (rel["lateral_offset_m"].abs() <= lateral_threshold_m)
        & (rel["actor_speed_mps"] <= actor_speed_threshold_mps)
        & (rel["focal_speed_mps"] >= focal_speed_threshold_mps)
    ].sort_values(["actor_id", "timestep"])
    candidates = _filter_on_future_path(
        candidates,
        df,
        path_lateral_threshold_m,
        path_lookahead_distance_m,
    )
    if map_overlay is not None:
        lane_constrained_types = (
            {"vehicle", "bus"}
            if lane_constrained_actor_types is None
            else {str(actor_type).lower() for actor_type in lane_constrained_actor_types}
        )
        candidates = _filter_same_lane_candidates(
            candidates,
            df,
            map_overlay,
            lane_assignment_threshold_m,
            lane_constrained_types,
        )

    min_steps = max(1, int(round(min_duration_s / sample_period_s)))
    events: list[dict[str, Any]] = []
    for actor_id, group in candidates.groupby("actor_id"):
        run: list[dict[str, Any]] = []
        previous_timestep: int | None = None
        for row in group.to_dict(orient="records"):
            timestep = int(row["timestep"])
            if previous_timestep is None or timestep == previous_timestep + 1:
                run.append(row)
            else:
                _append_event(events, str(actor_id), run, min_steps, sample_period_s)
                run = [row]
            previous_timestep = timestep
        _append_event(events, str(actor_id), run, min_steps, sample_period_s)

    events.sort(key=lambda event: event["min_front_distance_m"])
    return events


def _speed_table(df: pd.DataFrame) -> pd.DataFrame:
    speeds = df[["track_id", "timestep", "velocity_x", "velocity_y"]].copy()
    speeds["speed_mps"] = speeds.apply(
        lambda row: math.hypot(float(row["velocity_x"]), float(row["velocity_y"])),
        axis=1,
    )
    return speeds[["track_id", "timestep", "speed_mps"]]


def _filter_on_future_path(
    candidates: pd.DataFrame,
    df: pd.DataFrame,
    path_lateral_threshold_m: float,
    path_lookahead_distance_m: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    focal_track = get_focal_track(df).sort_values("timestep")
    actor_positions = df[["track_id", "timestep", "position_x", "position_y"]].rename(
        columns={
            "track_id": "actor_id",
            "position_x": "actor_position_x",
            "position_y": "actor_position_y",
        }
    )
    result = candidates.merge(actor_positions, on=["actor_id", "timestep"], how="left")
    path_errors: list[float] = []
    path_progress: list[float] = []
    on_path: list[bool] = []
    for row in result.to_dict(orient="records"):
        match = future_path_match(
            focal_track,
            int(row["timestep"]),
            (float(row["actor_position_x"]), float(row["actor_position_y"])),
            lateral_threshold_m=path_lateral_threshold_m,
            lookahead_distance_m=path_lookahead_distance_m,
        )
        path_errors.append(match.lateral_error_m)
        path_progress.append(match.path_progress_m)
        on_path.append(match.on_path)

    result["path_lateral_error_m"] = path_errors
    result["path_progress_m"] = path_progress
    result["on_future_path"] = on_path
    return result[result["on_future_path"]].drop(
        columns=["actor_position_x", "actor_position_y", "on_future_path"]
    )


def _filter_same_lane_candidates(
    candidates: pd.DataFrame,
    df: pd.DataFrame,
    map_overlay: MapOverlay,
    lane_assignment_threshold_m: float,
    lane_constrained_actor_types: set[str],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    positions = df.set_index(["track_id", "timestep"])[["position_x", "position_y"]]
    keep: list[bool] = []
    focal_lanes: list[int | None] = []
    actor_lanes: list[int | None] = []
    for _, row in candidates.iterrows():
        actor_type = str(row["actor_type"]).lower()
        if actor_type not in lane_constrained_actor_types:
            keep.append(True)
            focal_lanes.append(None)
            actor_lanes.append(None)
            continue

        timestep = int(row["timestep"])
        focal_lane = _lane_for_track_at_time(
            positions,
            str(row["track_id"]),
            timestep,
            map_overlay,
            lane_assignment_threshold_m,
        )
        actor_lane = _lane_for_track_at_time(
            positions,
            str(row["actor_id"]),
            timestep,
            map_overlay,
            lane_assignment_threshold_m,
        )
        focal_lanes.append(focal_lane)
        actor_lanes.append(actor_lane)
        keep.append(focal_lane is not None and focal_lane == actor_lane)

    result = candidates.copy()
    result["focal_lane_id"] = focal_lanes
    result["actor_lane_id"] = actor_lanes
    return result.loc[keep]


def _lane_for_track_at_time(
    positions: pd.DataFrame,
    track_id: str,
    timestep: int,
    map_overlay: MapOverlay,
    lane_assignment_threshold_m: float,
) -> int | None:
    try:
        row = positions.loc[(track_id, timestep)]
    except KeyError:
        return None
    return nearest_lane_index(
        map_overlay,
        (float(row["position_x"]), float(row["position_y"])),
        max_distance_m=lane_assignment_threshold_m,
    )


def _append_event(
    events: list[dict[str, Any]],
    actor_id: str,
    rows: list[dict[str, Any]],
    min_steps: int,
    sample_period_s: float,
) -> None:
    if len(rows) < min_steps:
        return

    first = rows[0]
    last = rows[-1]
    finite_ttc = [
        float(row["ttc_s"])
        for row in rows
        if pd.notna(row["ttc_s"]) and float(row["ttc_s"]) != float("inf")
    ]
    min_ttc = min(finite_ttc) if finite_ttc else float("inf")
    events.append(
        {
            "scenario_id": str(first["scenario_id"]),
            "event_type": "stopped_vehicle_ahead",
            "track_id": str(first["track_id"]),
            "actor_id": actor_id,
            "start_timestep": int(first["timestep"]),
            "end_timestep": int(last["timestep"]),
            "min_front_distance_m": min(float(row["front_projection_m"]) for row in rows),
            "min_ttc_s": min_ttc,
            "evidence": {
                "duration_s": round(len(rows) * sample_period_s, 3),
                "min_front_distance_m": min(float(row["front_projection_m"]) for row in rows),
                "actor_max_speed_mps": max(float(row["actor_speed_mps"]) for row in rows),
                "focal_min_speed_mps": min(float(row["focal_speed_mps"]) for row in rows),
                "min_ttc_s": min_ttc,
                "min_path_lateral_error_m": min(
                    float(row["path_lateral_error_m"]) for row in rows
                ),
                "max_path_progress_m": max(float(row["path_progress_m"]) for row in rows),
                "focal_lane_id": _common_lane_id(rows, "focal_lane_id"),
                "actor_lane_id": _common_lane_id(rows, "actor_lane_id"),
            },
        }
    )


def _common_lane_id(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    unique = set(values)
    return int(values[0]) if len(unique) == 1 else None
