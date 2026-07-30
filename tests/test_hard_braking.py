import pandas as pd

from drivescene.detectors.hard_braking import detect_hard_braking


def test_detect_hard_braking_requires_sustained_deceleration() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal"] * 6,
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 1, 2, 3, 4, 5],
            "velocity_x": [10.0, 9.6, 9.1, 8.6, 8.2, 8.2],
            "velocity_y": [0.0] * 6,
            "object_type": ["vehicle"] * 6,
            "position_x": [0.0, 1.0, 1.96, 2.87, 3.73, 4.55],
            "position_y": [0.0] * 6,
        }
    )

    events = detect_hard_braking(df, accel_threshold_mps2=-3.0, min_duration_s=0.3)

    assert len(events) == 1
    assert events[0]["scenario_id"] == "scene-1"
    assert events[0]["event_type"] == "hard_braking"
    assert events[0]["track_id"] == "focal"
    assert events[0]["start_timestep"] == 2
    assert events[0]["end_timestep"] == 4
    assert round(events[0]["min_acceleration_mps2"], 6) == -4.666667
    assert events[0]["evidence"]["duration_s"] == 0.3
    assert round(events[0]["evidence"]["min_velocity_acceleration_mps2"], 6) == -4.666667
    assert round(events[0]["evidence"]["min_position_acceleration_mps2"], 6) == -4.666667


def test_detect_hard_braking_uses_smoothed_longitudinal_acceleration() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 5,
            "track_id": ["focal"] * 5,
            "focal_track_id": ["focal"] * 5,
            "timestep": [0, 1, 2, 3, 4],
            "velocity_x": [10.0, 9.4, 9.4, 9.4, 9.4],
            "velocity_y": [0.0] * 5,
            "heading": [0.0] * 5,
            "object_type": ["vehicle"] * 5,
            "position_x": [0.0] * 5,
            "position_y": [0.0] * 5,
        }
    )

    events = detect_hard_braking(df, accel_threshold_mps2=-5.0, min_duration_s=0.1)

    assert events == []


def test_detect_hard_braking_requires_position_based_deceleration() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal"] * 6,
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 1, 2, 3, 4, 5],
            "velocity_x": [10.0, 9.6, 9.1, 8.6, 8.2, 8.2],
            "velocity_y": [0.0] * 6,
            "object_type": ["vehicle"] * 6,
            "position_x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "position_y": [0.0] * 6,
        }
    )

    events = detect_hard_braking(df, accel_threshold_mps2=-3.0, min_duration_s=0.3)

    assert events == []


def test_detect_hard_braking_ignores_lateral_velocity_change() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 5,
            "track_id": ["focal"] * 5,
            "focal_track_id": ["focal"] * 5,
            "timestep": [0, 1, 2, 3, 4],
            "velocity_x": [10.0] * 5,
            "velocity_y": [8.0, 6.0, 4.0, 2.0, 0.0],
            "heading": [0.0] * 5,
            "object_type": ["vehicle"] * 5,
            "position_x": [0.0] * 5,
            "position_y": [0.0] * 5,
        }
    )

    events = detect_hard_braking(df, accel_threshold_mps2=-3.0, min_duration_s=0.3)

    assert events == []
