import numpy as np
import pandas as pd


def add_speed(df: pd.DataFrame, sample_period_s: float = 0.1) -> pd.DataFrame:
    result = df.sort_values(["track_id", "timestep"]).copy() if {"track_id", "timestep"}.issubset(df.columns) else df.copy()
    result["speed_mps"] = np.hypot(result["velocity_x"], result["velocity_y"])
    if {"track_id", "position_x", "position_y"}.issubset(result.columns):
        grouped = result.groupby("track_id")
        position_dx = grouped["position_x"].diff()
        position_dy = grouped["position_y"].diff()
        position_speed = np.hypot(position_dx, position_dy) / sample_period_s
        result["position_speed_mps"] = position_speed
        first_step_speed = grouped["position_speed_mps"].transform(lambda values: values.bfill().fillna(0.0))
        result["position_speed_mps"] = result["position_speed_mps"].fillna(first_step_speed)
    return result


def add_acceleration(
    df: pd.DataFrame,
    sample_period_s: float = 0.1,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    result = df.sort_values(["track_id", "timestep"]).copy()
    grouped = result.groupby("track_id")
    result["accel_x_mps2"] = grouped["velocity_x"].diff().fillna(0.0) / sample_period_s
    result["accel_y_mps2"] = grouped["velocity_y"].diff().fillna(0.0) / sample_period_s

    if "heading" in result.columns:
        heading = result["heading"]
        longitudinal_accel = (
            result["accel_x_mps2"] * np.cos(heading)
            + result["accel_y_mps2"] * np.sin(heading)
        )
    else:
        if "speed_mps" not in result.columns:
            result = add_speed(result, sample_period_s=sample_period_s)
        longitudinal_accel = result.groupby("track_id")["speed_mps"].diff().fillna(0.0) / sample_period_s

    result["longitudinal_accel_mps2"] = longitudinal_accel
    if "position_speed_mps" in result.columns:
        result["position_accel_mps2"] = (
            result.groupby("track_id")["position_speed_mps"].diff().fillna(0.0) / sample_period_s
        )
        result["position_accel_smooth_mps2"] = _smooth_by_track(
            result,
            "position_accel_mps2",
            smoothing_window,
        )
    if smoothing_window <= 1:
        smooth_accel = result["longitudinal_accel_mps2"]
    else:
        smooth_accel = _smooth_by_track(result, "longitudinal_accel_mps2", smoothing_window)

    result["longitudinal_accel_smooth_mps2"] = smooth_accel
    result["accel_mps2"] = result["longitudinal_accel_smooth_mps2"]
    return result


def _smooth_by_track(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    if window <= 1:
        return df[column]
    return df.groupby("track_id")[column].transform(
        lambda values: values.rolling(
            window=window,
            center=True,
            min_periods=1,
        ).mean()
    )
