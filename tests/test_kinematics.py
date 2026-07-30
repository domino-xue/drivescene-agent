import pandas as pd

from drivescene.features.kinematics import add_acceleration, add_speed


def test_add_speed_uses_velocity_magnitude() -> None:
    df = pd.DataFrame({"velocity_x": [3.0, 0.0], "velocity_y": [4.0, -2.0]})

    result = add_speed(df)

    assert result["speed_mps"].tolist() == [5.0, 2.0]


def test_add_speed_computes_position_based_speed() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["a", "a", "a", "b"],
            "timestep": [0, 1, 2, 0],
            "velocity_x": [0.0, 0.0, 0.0, 1.0],
            "velocity_y": [0.0, 0.0, 0.0, 0.0],
            "position_x": [0.0, 1.0, 1.6, 5.0],
            "position_y": [0.0, 0.0, 0.0, 5.0],
        }
    )

    result = add_speed(df, sample_period_s=0.1)

    assert result["speed_mps"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert result.sort_values(["track_id", "timestep"])["position_speed_mps"].round(6).tolist() == [
        10.0,
        10.0,
        6.0,
        0.0,
    ]


def test_add_acceleration_computes_raw_and_smoothed_longitudinal_projection() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["a", "a", "a", "b", "b", "b"],
            "timestep": [0, 1, 2, 0, 1, 2],
            "velocity_x": [10.0, 9.5, 9.5, 0.0, 0.0, 0.0],
            "velocity_y": [0.0, 0.0, 0.0, 2.0, 3.0, 3.0],
            "heading": [
                0.0,
                0.0,
                0.0,
                1.5707963267948966,
                1.5707963267948966,
                1.5707963267948966,
            ],
        }
    )

    result = add_acceleration(df, sample_period_s=0.1)
    sorted_result = result.sort_values(["track_id", "timestep"])

    assert sorted_result["accel_x_mps2"].tolist() == [
        0.0,
        -5.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert sorted_result["accel_y_mps2"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        0.0,
    ]
    assert sorted_result["longitudinal_accel_mps2"].round(6).tolist() == [
        0.0,
        -5.0,
        0.0,
        0.0,
        10.0,
        0.0,
    ]
    assert sorted_result["longitudinal_accel_smooth_mps2"].round(6).tolist() == [
        -2.5,
        -1.666667,
        -2.5,
        5.0,
        3.333333,
        5.0,
    ]
    assert sorted_result["accel_mps2"].round(6).tolist() == [
        -2.5,
        -1.666667,
        -2.5,
        5.0,
        3.333333,
        5.0,
    ]


def test_add_acceleration_falls_back_to_speed_delta_without_heading() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["a", "a"],
            "timestep": [0, 1],
            "velocity_x": [3.0, 0.0],
            "velocity_y": [4.0, 4.0],
        }
    )

    result = add_acceleration(df, sample_period_s=0.1)

    assert result["speed_mps"].tolist() == [5.0, 4.0]
    assert result["longitudinal_accel_mps2"].tolist() == [0.0, -10.0]
    assert result["longitudinal_accel_smooth_mps2"].tolist() == [-5.0, -5.0]
    assert result["accel_mps2"].tolist() == [-5.0, -5.0]


def test_add_acceleration_computes_position_based_acceleration() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["a", "a", "a", "a", "a"],
            "timestep": [0, 1, 2, 3, 4],
            "velocity_x": [10.0, 9.6, 9.1, 8.6, 8.2],
            "velocity_y": [0.0] * 5,
            "position_x": [0.0, 1.0, 1.96, 2.87, 3.73],
            "position_y": [0.0] * 5,
        }
    )

    result = add_acceleration(add_speed(df, sample_period_s=0.1), sample_period_s=0.1)

    assert result["position_speed_mps"].round(6).tolist() == [10.0, 10.0, 9.6, 9.1, 8.6]
    assert result["position_accel_mps2"].round(6).tolist() == [0.0, 0.0, -4.0, -5.0, -5.0]
    assert result["position_accel_smooth_mps2"].round(6).tolist() == [
        0.0,
        -1.333333,
        -3.0,
        -4.666667,
        -5.0,
    ]
