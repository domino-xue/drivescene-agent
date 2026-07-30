from typing import Any

import pandas as pd

from drivescene.data.loader import get_focal_track
from drivescene.features.kinematics import add_acceleration, add_speed


def detect_hard_braking(
    df: pd.DataFrame,
    accel_threshold_mps2: float = -3.0,
    min_duration_s: float = 0.3,
    sample_period_s: float = 0.1,
) -> list[dict[str, Any]]:
    track = get_focal_track(df)
    if track.empty:
        return []

    track = add_acceleration(add_speed(track, sample_period_s=sample_period_s), sample_period_s=sample_period_s)
    if "position_accel_smooth_mps2" not in track.columns:
        return []

    threshold = accel_threshold_mps2 + 1e-9
    braking = (track["accel_mps2"] <= threshold) & (
        track["position_accel_smooth_mps2"] <= threshold
    )
    min_steps = max(1, int(round(min_duration_s / sample_period_s)))
    events: list[dict[str, Any]] = []

    run_start: int | None = None
    run_indices: list[int] = []

    for row_index, is_braking in braking.items():
        if is_braking:
            if run_start is None:
                run_start = row_index
                run_indices = []
            run_indices.append(row_index)
        elif run_start is not None:
            _append_event_if_long_enough(events, track, run_indices, min_steps, sample_period_s)
            run_start = None
            run_indices = []

    if run_start is not None:
        _append_event_if_long_enough(events, track, run_indices, min_steps, sample_period_s)

    return events


def _append_event_if_long_enough(
    events: list[dict[str, Any]],
    track: pd.DataFrame,
    indices: list[int],
    min_steps: int,
    sample_period_s: float,
) -> None:
    if len(indices) < min_steps:
        return

    segment = track.loc[indices]
    first = segment.iloc[0]
    last = segment.iloc[-1]
    full_track = track.reset_index(drop=True)
    start_pos = int(full_track.index[full_track["timestep"] == first["timestep"]][0])
    end_pos = int(full_track.index[full_track["timestep"] == last["timestep"]][0])
    prev_speed = full_track.iloc[max(0, start_pos - 1)]["speed_mps"]
    next_speed = full_track.iloc[min(len(full_track) - 1, end_pos + 1)]["speed_mps"]

    events.append(
        {
            "scenario_id": str(first["scenario_id"]),
            "event_type": "hard_braking",
            "track_id": str(first["track_id"]),
            "start_timestep": int(first["timestep"]),
            "end_timestep": int(last["timestep"]),
            "min_acceleration_mps2": float(segment["accel_mps2"].min()),
            "evidence": {
                "duration_s": round(len(segment) * sample_period_s, 3),
                "speed_before_mps": float(prev_speed),
                "speed_after_mps": float(next_speed),
                "min_velocity_acceleration_mps2": float(segment["accel_mps2"].min()),
                "min_position_acceleration_mps2": float(segment["position_accel_smooth_mps2"].min()),
            },
        }
    )
