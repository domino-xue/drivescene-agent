import pandas as pd

from drivescene.features.relative_motion import compute_relative_motion


def test_compute_relative_motion_uses_focal_heading_frame() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 4,
            "track_id": ["focal", "front", "focal", "front"],
            "focal_track_id": ["focal"] * 4,
            "timestep": [0, 0, 1, 1],
            "position_x": [0.0, 12.0, 1.0, 12.5],
            "position_y": [0.0, 1.0, 0.0, 1.0],
            "heading": [0.0, 0.0, 0.0, 0.0],
            "velocity_x": [10.0, 8.0, 10.0, 8.0],
            "velocity_y": [0.0, 0.0, 0.0, 0.0],
            "object_type": ["vehicle"] * 4,
        }
    )

    result = compute_relative_motion(df)

    first = result[result["timestep"] == 0].iloc[0]
    assert first["actor_id"] == "front"
    assert first["focal_actor_type"] == "vehicle"
    assert first["actor_type"] == "vehicle"
    assert first["longitudinal_offset_m"] == 12.0
    assert first["lateral_offset_m"] == 1.0
    assert round(first["distance_m"], 3) == 12.042
    assert first["closing_speed_mps"] == 2.0
    assert first["ttc_s"] == 6.0
    assert first["heading_delta_rad"] == 0.0
    assert first["heading_alignment"] == 1.0
    assert first["front_projection_m"] == 12.0
    assert first["front_angle_rad"] < 0.1
