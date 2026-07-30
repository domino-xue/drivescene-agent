from typing import Any

import math
import numpy as np
import pandas as pd

from drivescene.data.loader import get_focal_track
from drivescene.features.path_geometry import heading_delta_deg, trajectory_curvature_rad_per_m
from drivescene.maps.overlay import MapOverlay, nearest_lane_index, project_to_nearest_lane


def detect_lane_change(
    df: pd.DataFrame,
    lateral_displacement_threshold_m: float = 2.5,
    min_forward_displacement_m: float = 5.0,
    min_lateral_velocity_mps: float = 0.5,
    min_heading_delta_deg: float = 3.0,
    max_turn_heading_delta_deg: float = 20.0,
    max_turn_curvature_rad_per_m: float = 0.04,
    turn_context_steps: int = 15,
    max_lane_following_lateral_change_m: float = 1.5,
    max_lane_following_abs_offset_m: float = 1.5,
    min_duration_s: float = 0.5,
    max_duration_s: float = 5.0,
    sample_period_s: float = 0.1,
    map_overlay: MapOverlay | None = None,
    lane_assignment_threshold_m: float = 1.5,
    lane_heading_alignment_threshold_deg: float = 35.0,
) -> list[dict[str, Any]]:
    track = get_focal_track(df).sort_values("timestep")
    if len(track) < 2:
        return []

    min_steps = max(2, int(round(min_duration_s / sample_period_s)))
    max_steps = max(min_steps, int(round(max_duration_s / sample_period_s)))
    rows = track.reset_index(drop=True)
    position_x = rows["position_x"].astype(float).to_numpy()
    position_y = rows["position_y"].astype(float).to_numpy()
    velocity_x = rows["velocity_x"].astype(float).to_numpy()
    velocity_y = rows["velocity_y"].astype(float).to_numpy()
    headings = rows["heading"].astype(float).to_numpy()
    lane_projections = _lane_projection_table(
        rows,
        map_overlay,
        lane_assignment_threshold_m,
        lane_heading_alignment_threshold_deg,
    )
    events: list[dict[str, Any]] = []

    for start in range(len(rows) - min_steps + 1):
        max_end = min(len(rows), start + max_steps)
        for end in range(start + min_steps - 1, max_end):
            event = _event_from_arrays(
                rows,
                start,
                end,
                position_x,
                position_y,
                velocity_x,
                velocity_y,
                headings,
                lateral_displacement_threshold_m,
                min_forward_displacement_m,
                min_lateral_velocity_mps,
                min_heading_delta_deg,
                max_turn_heading_delta_deg,
                max_turn_curvature_rad_per_m,
                turn_context_steps,
                max_lane_following_lateral_change_m,
                max_lane_following_abs_offset_m,
                sample_period_s,
                map_overlay,
                lane_assignment_threshold_m,
                lane_heading_alignment_threshold_deg,
                lane_projections,
            )
            if event is not None:
                events.append(event)
                return events
    return events


def _event_from_arrays(
    rows: pd.DataFrame,
    start: int,
    end: int,
    position_x: np.ndarray,
    position_y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    headings: np.ndarray,
    lateral_displacement_threshold_m: float,
    min_forward_displacement_m: float,
    min_lateral_velocity_mps: float,
    min_heading_delta_deg: float,
    max_turn_heading_delta_deg: float,
    max_turn_curvature_rad_per_m: float,
    turn_context_steps: int,
    max_lane_following_lateral_change_m: float,
    max_lane_following_abs_offset_m: float,
    sample_period_s: float,
    map_overlay: MapOverlay | None,
    lane_assignment_threshold_m: float,
    lane_heading_alignment_threshold_deg: float,
    lane_projections: pd.DataFrame | None,
) -> dict[str, Any] | None:
    first = rows.iloc[start]
    last = rows.iloc[end]
    dx = float(position_x[end] - position_x[start])
    dy = float(position_y[end] - position_y[start])
    heading = float(headings[start])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    forward_displacement = dx * cos_h + dy * sin_h
    lateral_displacement = -dx * sin_h + dy * cos_h
    if abs(lateral_displacement) < lateral_displacement_threshold_m:
        return None
    if forward_displacement < min_forward_displacement_m:
        return None

    lateral_speeds = -velocity_x[start : end + 1] * sin_h + velocity_y[start : end + 1] * cos_h
    max_lateral_speed = float(np.abs(lateral_speeds).max())
    window_heading_delta_deg = heading_delta_deg(headings[start : end + 1])
    if max_lateral_speed < min_lateral_velocity_mps and window_heading_delta_deg < min_heading_delta_deg:
        return None

    curvature = trajectory_curvature_rad_per_m(
        headings[start : end + 1],
        position_x[start : end + 1],
        position_y[start : end + 1],
    )
    context_start = max(0, start - turn_context_steps)
    context_end = min(len(rows) - 1, end + turn_context_steps)
    context_heading_delta_deg = heading_delta_deg(headings[context_start : context_end + 1])
    context_curvature = trajectory_curvature_rad_per_m(
        headings[context_start : context_end + 1],
        position_x[context_start : context_end + 1],
        position_y[context_start : context_end + 1],
    )
    turning_suppressed = context_heading_delta_deg > max_turn_heading_delta_deg
    lane_info = _lane_transition_info(
        first,
        last,
        map_overlay,
        lane_assignment_threshold_m,
        lane_heading_alignment_threshold_deg,
    )
    lane_transition = _is_lane_transition(lane_info)
    lane_relative_info = _lane_relative_motion_info(
        lane_projections.iloc[start : end + 1] if lane_projections is not None else None,
        map_overlay,
    )
    if turning_suppressed:
        return None
    if _is_lane_following_on_curved_road(
        lane_relative_info,
        max_lane_following_lateral_change_m,
        max_lane_following_abs_offset_m,
    ):
        return None

    direction = "left" if lateral_displacement > 0 else "right"
    return {
        "scenario_id": str(first["scenario_id"]),
        "event_type": "lane_change",
        "track_id": str(first["track_id"]),
        "start_timestep": int(first["timestep"]),
        "end_timestep": int(last["timestep"]),
        "direction": direction,
        "lateral_displacement_m": abs(float(lateral_displacement)),
        "evidence": {
            "duration_s": round((end - start + 1) * sample_period_s, 3),
            "direction": direction,
            "lateral_displacement_m": abs(float(lateral_displacement)),
            "forward_displacement_m": float(forward_displacement),
            "max_abs_lateral_velocity_mps": max_lateral_speed,
            "heading_delta_deg": window_heading_delta_deg,
            "trajectory_curvature_rad_per_m": curvature,
            "context_heading_delta_deg": context_heading_delta_deg,
            "context_trajectory_curvature_rad_per_m": context_curvature,
            "turning_suppressed": turning_suppressed,
            "lane_transition": lane_transition,
            **lane_info,
            **lane_relative_info,
        },
    }


def _lane_transition_info(
    first: pd.Series,
    last: pd.Series,
    map_overlay: MapOverlay | None,
    lane_assignment_threshold_m: float,
    lane_heading_alignment_threshold_deg: float,
) -> dict[str, int | None]:
    if map_overlay is None:
        return {"start_lane_id": None, "end_lane_id": None}
    return {
        "start_lane_id": nearest_lane_index(
            map_overlay,
            (float(first["position_x"]), float(first["position_y"])),
            max_distance_m=lane_assignment_threshold_m,
            heading_rad=float(first["heading"]),
            max_heading_delta_deg=lane_heading_alignment_threshold_deg,
        ),
        "end_lane_id": nearest_lane_index(
            map_overlay,
            (float(last["position_x"]), float(last["position_y"])),
            max_distance_m=lane_assignment_threshold_m,
            heading_rad=float(last["heading"]),
            max_heading_delta_deg=lane_heading_alignment_threshold_deg,
        ),
    }


def _is_lane_transition(lane_info: dict[str, int | None]) -> bool:
    start_lane = lane_info["start_lane_id"]
    end_lane = lane_info["end_lane_id"]
    return start_lane is not None and end_lane is not None and start_lane != end_lane


def _lane_projection_table(
    rows: pd.DataFrame,
    map_overlay: MapOverlay | None,
    lane_assignment_threshold_m: float,
    lane_heading_alignment_threshold_deg: float,
) -> pd.DataFrame | None:
    if map_overlay is None:
        return None

    projection_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        projection = project_to_nearest_lane(
            map_overlay,
            (float(row["position_x"]), float(row["position_y"])),
            max_distance_m=lane_assignment_threshold_m,
            heading_rad=float(row["heading"]),
            max_heading_delta_deg=lane_heading_alignment_threshold_deg,
        )
        projection_rows.append(
            {
                "lane_id": projection.lane_id if projection is not None else None,
                "signed_lateral_offset_m": (
                    projection.signed_lateral_offset_m if projection is not None else None
                ),
                "tangent_heading_rad": (
                    projection.tangent_heading_rad if projection is not None else None
                ),
            }
        )
    return pd.DataFrame(projection_rows)


def _lane_relative_motion_info(
    lane_window: pd.DataFrame | None,
    map_overlay: MapOverlay | None,
) -> dict[str, Any]:
    if map_overlay is None or lane_window is None or lane_window.empty:
        return {
            "lane_following_on_curved_road": False,
            "lane_relative_lateral_change_m": None,
            "start_lane_offset_m": None,
            "end_lane_offset_m": None,
            "lane_tangent_heading_delta_deg": None,
            "nearest_lane_sequence": [],
        }

    if lane_window[["lane_id", "signed_lateral_offset_m", "tangent_heading_rad"]].isna().any().any():
        return {
            "lane_following_on_curved_road": False,
            "lane_relative_lateral_change_m": None,
            "start_lane_offset_m": None,
            "end_lane_offset_m": None,
            "lane_tangent_heading_delta_deg": None,
            "nearest_lane_sequence": [],
        }

    lane_sequence = [int(value) for value in lane_window["lane_id"].tolist()]
    offsets = [float(value) for value in lane_window["signed_lateral_offset_m"].tolist()]
    tangents = [float(value) for value in lane_window["tangent_heading_rad"].tolist()]
    lateral_change = float(max(offsets) - min(offsets)) if offsets else 0.0
    lane_tangent_delta = heading_delta_deg(tangents)
    lane_following = _is_continuous_lane_following(lane_sequence, tangents)
    return {
        "lane_following_on_curved_road": lane_following,
        "lane_relative_lateral_change_m": lateral_change,
        "start_lane_offset_m": offsets[0] if offsets else None,
        "end_lane_offset_m": offsets[-1] if offsets else None,
        "lane_tangent_heading_delta_deg": lane_tangent_delta,
        "nearest_lane_sequence": _compact_lane_sequence(lane_sequence),
    }


def _is_lane_following_on_curved_road(
    lane_relative_info: dict[str, Any],
    max_lane_following_lateral_change_m: float,
    max_lane_following_abs_offset_m: float,
) -> bool:
    if not lane_relative_info.get("lane_following_on_curved_road"):
        return False
    lateral_change = lane_relative_info.get("lane_relative_lateral_change_m")
    start_offset = lane_relative_info.get("start_lane_offset_m")
    end_offset = lane_relative_info.get("end_lane_offset_m")
    if lateral_change is not None and float(lateral_change) <= max_lane_following_lateral_change_m:
        return True
    return (
        start_offset is not None
        and end_offset is not None
        and abs(float(start_offset)) <= max_lane_following_abs_offset_m
        and abs(float(end_offset)) <= max_lane_following_abs_offset_m
    )


def _is_continuous_lane_following(
    lane_sequence: list[int],
    tangents: list[float],
    max_tangent_step_deg: float = 25.0,
) -> bool:
    if not lane_sequence or not tangents:
        return False
    if len(set(lane_sequence)) == 1:
        return True
    tangent_steps = [
        heading_delta_deg([current_tangent, next_tangent])
        for current_tangent, next_tangent in zip(tangents, tangents[1:])
    ]
    return bool(tangent_steps) and max(tangent_steps) <= max_tangent_step_deg


def _compact_lane_sequence(lane_sequence: list[int]) -> list[int]:
    compact: list[int] = []
    for lane_id in lane_sequence:
        if not compact or compact[-1] != lane_id:
            compact.append(lane_id)
    return compact


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi
