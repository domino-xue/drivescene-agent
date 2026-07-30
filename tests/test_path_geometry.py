import math

import pandas as pd

from drivescene.features.path_geometry import (
    future_path_match,
    heading_delta_deg,
    trajectory_curvature_rad_per_m,
)


def test_heading_delta_deg_unwraps_angle_crossing_pi() -> None:
    delta = heading_delta_deg([math.radians(170), math.radians(190)])

    assert round(delta, 3) == 20.0


def test_future_path_match_extends_straight_short_track() -> None:
    track = pd.DataFrame(
        {
            "timestep": [0, 1],
            "position_x": [0.0, 1.0],
            "position_y": [0.0, 0.0],
            "velocity_x": [10.0, 10.0],
            "velocity_y": [0.0, 0.0],
            "heading": [0.0, 0.0],
        }
    )

    match = future_path_match(track, timestep=0, point=(12.0, 0.5), lateral_threshold_m=1.5)

    assert match.on_path
    assert match.path_progress_m >= 10.0
    assert match.lateral_error_m <= 1.5


def test_future_path_match_does_not_extend_sharp_turn_path() -> None:
    rows = []
    radius_m = 12.0
    for timestep in range(21):
        progress = timestep / 20
        heading = progress * math.pi / 2
        rows.append(
            {
                "timestep": timestep,
                "position_x": radius_m * math.sin(heading),
                "position_y": radius_m * (1.0 - math.cos(heading)),
                "velocity_x": 8.0 * math.cos(heading),
                "velocity_y": 8.0 * math.sin(heading),
                "heading": heading,
            }
        )
    track = pd.DataFrame(rows)

    match = future_path_match(track, timestep=13, point=(14.0, 15.5), lateral_threshold_m=2.0)

    assert not match.on_path
    assert match.lateral_error_m > 2.0


def test_trajectory_curvature_reports_turning_arc() -> None:
    curvature = trajectory_curvature_rad_per_m(
        [0.0, math.pi / 4, math.pi / 2],
        [0.0, 7.0, 10.0],
        [0.0, 3.0, 10.0],
    )

    assert curvature > 0.08
