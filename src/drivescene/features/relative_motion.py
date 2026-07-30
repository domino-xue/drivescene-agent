import math

import numpy as np
import pandas as pd

from drivescene.data.loader import get_focal_track


def compute_relative_motion(df: pd.DataFrame) -> pd.DataFrame:
    focal = get_focal_track(df)
    if focal.empty:
        return pd.DataFrame()

    focal_by_time = focal.set_index("timestep")
    rows: list[dict[str, object]] = []

    for _, actor in df.iterrows():
        timestep = actor["timestep"]
        if str(actor["track_id"]) == str(actor["focal_track_id"]):
            continue
        if timestep not in focal_by_time.index:
            continue

        ego = focal_by_time.loc[timestep]
        dx = float(actor["position_x"] - ego["position_x"])
        dy = float(actor["position_y"] - ego["position_y"])
        heading = float(ego["heading"])
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        longitudinal = dx * cos_h + dy * sin_h
        lateral = -dx * sin_h + dy * cos_h
        distance = math.hypot(dx, dy)

        speed = math.hypot(float(ego["velocity_x"]), float(ego["velocity_y"]))
        if speed > 1e-6:
            front_unit_x = float(ego["velocity_x"]) / speed
            front_unit_y = float(ego["velocity_y"]) / speed
        else:
            front_unit_x = cos_h
            front_unit_y = sin_h

        front_projection = dx * front_unit_x + dy * front_unit_y
        if distance > 1e-6:
            front_cosine = max(-1.0, min(1.0, front_projection / distance))
            front_angle = math.acos(front_cosine)
        else:
            front_angle = 0.0

        rel_vx = float(ego["velocity_x"] - actor["velocity_x"])
        rel_vy = float(ego["velocity_y"] - actor["velocity_y"])
        closing_speed = rel_vx * cos_h + rel_vy * sin_h
        ttc = longitudinal / closing_speed if longitudinal > 0 and closing_speed > 0 else np.inf
        heading_delta = _wrap_to_pi(float(actor["heading"]) - heading)
        heading_alignment = math.cos(heading_delta)

        rows.append(
            {
                "scenario_id": str(actor["scenario_id"]),
                "timestep": int(timestep),
                "track_id": str(ego["track_id"]),
                "actor_id": str(actor["track_id"]),
                "focal_actor_type": str(ego["object_type"]),
                "actor_type": str(actor["object_type"]),
                "longitudinal_offset_m": float(longitudinal),
                "front_projection_m": float(front_projection),
                "front_angle_rad": float(front_angle),
                "lateral_offset_m": float(lateral),
                "distance_m": float(distance),
                "closing_speed_mps": float(closing_speed),
                "ttc_s": float(ttc),
                "heading_delta_rad": float(heading_delta),
                "heading_alignment": float(heading_alignment),
            }
        )

    return pd.DataFrame(rows)


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi
