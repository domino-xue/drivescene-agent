from collections.abc import Iterable
import math
from typing import Any

import pandas as pd

from drivescene.data.loader import get_focal_track
from drivescene.features.relative_motion import compute_relative_motion
from drivescene.maps.overlay import MapOverlay, nearest_lane_index


def detect_cut_in(
    df: pd.DataFrame,
    entry_lateral_threshold_m: float = 1.5,
    final_lateral_threshold_m: float = 1.5,
    distance_threshold_m: float = 15.0,
    min_duration_s: float = 0.3,
    sample_period_s: float = 0.1,
    allowed_actor_types: Iterable[str] | None = None,
    min_actor_speed_mps: float = 1.0,
    min_lateral_closure_m: float = 1.0,
    max_focal_heading_change_deg: float = 15.0,
    max_focal_curvature_rad_per_m: float = 0.02,
    map_overlay: MapOverlay | None = None,
    lane_assignment_threshold_m: float = 1.5,
    lane_constrained_actor_types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    rel = compute_relative_motion(df)
    if rel.empty:
        return []

    allowed_types = (
        {"vehicle", "bus", "motorcyclist", "cyclist", "riderless_bicycle"}
        if allowed_actor_types is None
        else {str(actor_type).lower() for actor_type in allowed_actor_types}
    )
    rel = rel[rel["actor_type"].str.lower().isin(allowed_types)].copy()
    rel = rel[
        (rel["front_projection_m"] > 0)
        & (rel["front_projection_m"] <= distance_threshold_m)
    ].sort_values(["actor_id", "timestep"])

    min_steps = max(1, int(round(min_duration_s / sample_period_s)))
    track_state = _track_state_by_time(df)
    events: list[dict[str, Any]] = []
    for actor_id, group in rel.groupby("actor_id"):
        rows = list(group.to_dict(orient="records"))
        if not rows:
            continue
        first = rows[0]
        last = rows[-1]
        start_lat = abs(float(first["lateral_offset_m"]))
        end_lat = abs(float(last["lateral_offset_m"]))
        if start_lat <= entry_lateral_threshold_m:
            continue
        if end_lat > final_lateral_threshold_m:
            continue
        if not _has_lateral_entry(rows, min_lateral_closure_m):
            continue
        if not _has_contiguous_entry(rows, final_lateral_threshold_m, min_steps):
            continue

        speed_stats = _actor_speed_stats(rows, track_state)
        if speed_stats["max_actor_speed_mps"] < min_actor_speed_mps:
            continue

        lane_info = _lane_transition_info(
            df,
            rows,
            map_overlay,
            lane_assignment_threshold_m,
        )
        lane_transition_valid = _is_valid_lane_transition(lane_info)

        focal_turn = _focal_turn_metrics(df, int(first["timestep"]), int(last["timestep"]))
        if not lane_transition_valid and _is_turn_ambiguous(
            focal_turn,
            max_focal_heading_change_deg,
            max_focal_curvature_rad_per_m,
        ):
            continue

        events.append(
            _event_from_window(
                str(actor_id),
                rows,
                sample_period_s,
                {**speed_stats, **focal_turn, **lane_info},
            )
        )

    events.sort(key=lambda event: event["min_front_distance_m"])
    return events


def _has_contiguous_entry(
    rows: list[dict[str, Any]],
    final_lateral_threshold_m: float,
    min_steps: int,
) -> bool:
    entered = [
        row
        for row in rows
        if abs(float(row["lateral_offset_m"])) <= final_lateral_threshold_m
    ]
    if len(entered) < min_steps:
        return False
    timesteps = [int(row["timestep"]) for row in entered[-min_steps:]]
    return timesteps == list(range(timesteps[0], timesteps[0] + min_steps))


def _has_lateral_entry(
    rows: list[dict[str, Any]],
    min_lateral_closure_m: float,
) -> bool:
    offsets = [abs(float(row["lateral_offset_m"])) for row in rows]
    if len(offsets) < 2:
        return False
    if offsets[0] - offsets[-1] < min_lateral_closure_m:
        return False

    transitions = list(zip(offsets, offsets[1:]))
    non_diverging = sum(1 for current, next_value in transitions if next_value <= current + 0.25)
    return non_diverging / len(transitions) >= 0.6


def _track_state_by_time(df: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    state: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in df.iterrows():
        state[(str(row["track_id"]), int(row["timestep"]))] = {
            "speed_mps": math.hypot(float(row["velocity_x"]), float(row["velocity_y"])),
            "position_x": float(row["position_x"]),
            "position_y": float(row["position_y"]),
        }
    return state


def _actor_speed_stats(
    rows: list[dict[str, Any]],
    track_state: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    speeds = [
        track_state.get((str(row["actor_id"]), int(row["timestep"])), {}).get("speed_mps", 0.0)
        for row in rows
    ]
    return {
        "min_actor_speed_mps": min(speeds) if speeds else 0.0,
        "max_actor_speed_mps": max(speeds) if speeds else 0.0,
    }


def _focal_turn_metrics(
    df: pd.DataFrame,
    start_timestep: int,
    end_timestep: int,
) -> dict[str, float]:
    focal = get_focal_track(df).sort_values("timestep")
    window = focal[
        (focal["timestep"].astype(int) >= start_timestep)
        & (focal["timestep"].astype(int) <= end_timestep)
    ]
    if len(window) < 2:
        return {
            "focal_heading_change_deg": 0.0,
            "focal_curvature_rad_per_m": 0.0,
        }

    headings = [float(value) for value in window["heading"]]
    total_heading_change = sum(
        abs(_wrap_to_pi(next_heading - current_heading))
        for current_heading, next_heading in zip(headings, headings[1:])
    )
    positions = list(
        zip(
            window["position_x"].astype(float),
            window["position_y"].astype(float),
        )
    )
    path_length = sum(
        math.dist(start, end)
        for start, end in zip(positions, positions[1:])
    )
    curvature = total_heading_change / path_length if path_length > 1e-6 else 0.0
    return {
        "focal_heading_change_deg": total_heading_change * 180.0 / math.pi,
        "focal_curvature_rad_per_m": curvature,
    }


def _is_turn_ambiguous(
    focal_turn: dict[str, float],
    max_focal_heading_change_deg: float,
    max_focal_curvature_rad_per_m: float,
) -> bool:
    return (
        focal_turn["focal_heading_change_deg"] > max_focal_heading_change_deg
        and focal_turn["focal_curvature_rad_per_m"] > max_focal_curvature_rad_per_m
    )


def _lane_transition_info(
    df: pd.DataFrame,
    rows: list[dict[str, Any]],
    map_overlay: MapOverlay | None,
    lane_assignment_threshold_m: float,
) -> dict[str, int | None]:
    if map_overlay is None or not rows:
        return {
            "focal_start_lane_id": None,
            "focal_end_lane_id": None,
            "actor_start_lane_id": None,
            "actor_end_lane_id": None,
        }

    positions = df.set_index(["track_id", "timestep"])[["position_x", "position_y"]]
    first = rows[0]
    last = rows[-1]
    return {
        "focal_start_lane_id": _lane_for_track_at_time(
            positions,
            str(first["track_id"]),
            int(first["timestep"]),
            map_overlay,
            lane_assignment_threshold_m,
        ),
        "focal_end_lane_id": _lane_for_track_at_time(
            positions,
            str(last["track_id"]),
            int(last["timestep"]),
            map_overlay,
            lane_assignment_threshold_m,
        ),
        "actor_start_lane_id": _lane_for_track_at_time(
            positions,
            str(first["actor_id"]),
            int(first["timestep"]),
            map_overlay,
            lane_assignment_threshold_m,
        ),
        "actor_end_lane_id": _lane_for_track_at_time(
            positions,
            str(last["actor_id"]),
            int(last["timestep"]),
            map_overlay,
            lane_assignment_threshold_m,
        ),
    }


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


def _is_valid_lane_transition(lane_info: dict[str, int | None]) -> bool:
    focal_start = lane_info["focal_start_lane_id"]
    focal_end = lane_info["focal_end_lane_id"]
    actor_start = lane_info["actor_start_lane_id"]
    actor_end = lane_info["actor_end_lane_id"]
    return (
        focal_start is not None
        and focal_end is not None
        and actor_start is not None
        and actor_end is not None
        and focal_start == focal_end
        and actor_start != focal_start
        and actor_end == focal_end
    )


def _event_from_window(
    actor_id: str,
    rows: list[dict[str, Any]],
    sample_period_s: float,
    extra_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    finite_ttc = [
        float(row["ttc_s"])
        for row in rows
        if pd.notna(row["ttc_s"]) and float(row["ttc_s"]) != float("inf")
    ]
    min_ttc = min(finite_ttc) if finite_ttc else float("inf")
    evidence = {
        "duration_s": round(len(rows) * sample_period_s, 3),
        "start_lateral_offset_m": abs(float(first["lateral_offset_m"])),
        "end_lateral_offset_m": abs(float(last["lateral_offset_m"])),
        "min_front_distance_m": min(float(row["front_projection_m"]) for row in rows),
        "min_ttc_s": min_ttc,
        "max_closing_speed_mps": max(float(row["closing_speed_mps"]) for row in rows),
    }
    if extra_evidence:
        evidence.update(extra_evidence)

    return {
        "scenario_id": str(first["scenario_id"]),
        "event_type": "cut_in",
        "track_id": str(first["track_id"]),
        "actor_id": actor_id,
        "start_timestep": int(first["timestep"]),
        "end_timestep": int(last["timestep"]),
        "min_front_distance_m": min(float(row["front_projection_m"]) for row in rows),
        "min_ttc_s": min_ttc,
        "evidence": evidence,
    }


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi
