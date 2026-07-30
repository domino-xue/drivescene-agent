from collections.abc import Iterable
from typing import Any

import math
import pandas as pd

from drivescene.features.relative_motion import compute_relative_motion
from drivescene.maps.overlay import MapOverlay, nearest_lane_index


def detect_close_following(
    df: pd.DataFrame,
    distance_threshold_m: float = 10.0,
    ttc_threshold_s: float = 2.0,
    lateral_threshold_m: float = 2.0,
    min_duration_s: float = 0.3,
    sample_period_s: float = 0.1,
    min_heading_alignment: float = 0.0,
    excluded_actor_types: Iterable[str] | None = None,
    allowed_actor_types: Iterable[str] | None = None,
    front_angle_threshold_deg: float = 20.0,
    map_overlay: MapOverlay | None = None,
    lane_assignment_threshold_m: float = 1.5,
    lane_constrained_actor_types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rel = compute_relative_motion(df)
    if rel.empty:
        return []

    excluded_types = (
        {"pedestrian", "static"}
        if excluded_actor_types is None
        else {str(actor_type).lower() for actor_type in excluded_actor_types}
    )
    allowed_types = (
        {"vehicle", "bus", "motorcyclist", "cyclist", "riderless_bicycle"}
        if allowed_actor_types is None
        else {str(actor_type).lower() for actor_type in allowed_actor_types}
    )
    lane_constrained_types = (
        {"vehicle", "bus"}
        if lane_constrained_actor_types is None
        else {str(actor_type).lower() for actor_type in lane_constrained_actor_types}
    )
    actor_type = rel["actor_type"].str.lower()
    focal_actor_type = rel["focal_actor_type"].str.lower()
    front_angle_threshold_rad = math.radians(front_angle_threshold_deg)

    candidates = rel[
        (rel["front_projection_m"] > 0)
        & (rel["front_projection_m"] <= distance_threshold_m)
        & (rel["front_angle_rad"] <= front_angle_threshold_rad)
        & (rel["lateral_offset_m"].abs() <= lateral_threshold_m)
        & (rel["heading_alignment"] >= min_heading_alignment)
        & (~actor_type.isin(excluded_types))
        & (actor_type.isin(allowed_types))
        & (focal_actor_type.isin(allowed_types))
    ].copy()

    if map_overlay is not None:
        candidates = _filter_same_lane_candidates(
            df,
            candidates,
            map_overlay,
            lane_assignment_threshold_m,
            lane_constrained_types,
        )

    candidates = candidates.sort_values(["actor_id", "timestep"])

    min_steps = max(1, int(round(min_duration_s / sample_period_s)))
    events: list[dict[str, Any]] = []

    for actor_id, group in candidates.groupby("actor_id"):
        run_rows: list[pd.Series] = []
        previous_timestep: int | None = None
        for _, row in group.iterrows():
            timestep = int(row["timestep"])
            if previous_timestep is None or timestep == previous_timestep + 1:
                run_rows.append(row)
            else:
                _append_event(events, run_rows, min_steps, sample_period_s)
                run_rows = [row]
            previous_timestep = timestep
        _append_event(events, run_rows, min_steps, sample_period_s)

    events.sort(key=lambda event: (event["min_ttc_s"], event["min_front_distance_m"]))
    return events


def _filter_same_lane_candidates(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    map_overlay: MapOverlay,
    lane_assignment_threshold_m: float,
    lane_constrained_actor_types: set[str],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    positions = df.set_index(["track_id", "timestep"])[["position_x", "position_y"]]
    keep: list[bool] = []
    for _, row in candidates.iterrows():
        actor_type = str(row["actor_type"]).lower()
        if actor_type not in lane_constrained_actor_types:
            keep.append(True)
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
        keep.append(focal_lane is not None and focal_lane == actor_lane)

    return candidates.loc[keep]


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
    rows: list[pd.Series],
    min_steps: int,
    sample_period_s: float,
) -> None:
    if len(rows) < min_steps:
        return

    segment = pd.DataFrame(rows)
    first = segment.iloc[0]
    last = segment.iloc[-1]
    finite_ttc = segment["ttc_s"].replace([float("inf")], pd.NA).dropna()
    min_ttc = float(finite_ttc.min()) if not finite_ttc.empty else float("inf")

    events.append(
        {
            "scenario_id": str(first["scenario_id"]),
            "event_type": "close_following",
            "track_id": str(first["track_id"]),
            "actor_id": str(first["actor_id"]),
            "start_timestep": int(first["timestep"]),
            "end_timestep": int(last["timestep"]),
            "min_distance_m": float(segment["distance_m"].min()),
            "min_front_distance_m": float(segment["front_projection_m"].min()),
            "min_ttc_s": min_ttc,
            "evidence": {
                "duration_s": round(len(segment) * sample_period_s, 3),
                "min_front_distance_m": float(segment["front_projection_m"].min()),
                "min_lateral_offset_m": float(segment["lateral_offset_m"].abs().min()),
                "max_closing_speed_mps": float(segment["closing_speed_mps"].max()),
                "min_heading_alignment": float(segment["heading_alignment"].min()),
                "max_front_angle_deg": float(segment["front_angle_rad"].max() * 180 / math.pi),
            },
        }
    )
