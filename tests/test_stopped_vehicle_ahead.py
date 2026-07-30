import pandas as pd
import math

from drivescene.detectors.stopped_vehicle_ahead import detect_stopped_vehicle_ahead
from drivescene.maps.overlay import MapOverlay


def _stopped_vehicle_df(actor_speed: float = 0.2, lateral_offset: float = 0.5) -> pd.DataFrame:
    rows = []
    for timestep in range(6):
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 8.0,
                "velocity_y": 0.0,
                "position_x": timestep * 0.8,
                "position_y": 0.0,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "stopped",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": actor_speed,
                "velocity_y": 0.0,
                "position_x": 12.0 + timestep * actor_speed * 0.1,
                "position_y": lateral_offset,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )
    return pd.DataFrame(rows)


def test_detect_stopped_vehicle_ahead_finds_low_speed_front_vehicle() -> None:
    events = detect_stopped_vehicle_ahead(_stopped_vehicle_df())

    assert len(events) == 1
    assert events[0]["event_type"] == "stopped_vehicle_ahead"
    assert events[0]["actor_id"] == "stopped"
    assert events[0]["evidence"]["actor_max_speed_mps"] <= 1.0
    assert events[0]["evidence"]["focal_min_speed_mps"] >= 3.0


def test_detect_stopped_vehicle_ahead_rejects_moving_front_vehicle() -> None:
    events = detect_stopped_vehicle_ahead(_stopped_vehicle_df(actor_speed=4.0))

    assert events == []


def test_detect_stopped_vehicle_ahead_rejects_actor_off_future_turn_path() -> None:
    rows = []
    radius_m = 12.0
    actor_x = 14.0
    actor_y = 15.5
    for timestep in range(21):
        progress = timestep / 20
        heading = progress * math.pi / 2
        focal_x = radius_m * math.sin(heading)
        focal_y = radius_m * (1.0 - math.cos(heading))
        rows.append(
            {
                "scenario_id": "scene-turn",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 8.0 * math.cos(heading),
                "velocity_y": 8.0 * math.sin(heading),
                "position_x": focal_x,
                "position_y": focal_y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-turn",
                "track_id": "stopped",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "position_x": actor_x,
                "position_y": actor_y,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )

    events = detect_stopped_vehicle_ahead(pd.DataFrame(rows))

    assert events == []


def test_detect_stopped_vehicle_ahead_rejects_stopped_actor_in_different_map_lane() -> None:
    df = _stopped_vehicle_df(lateral_offset=3.5)
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (25.0, 0.0)],
            [(0.0, 3.5), (25.0, 3.5)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_stopped_vehicle_ahead(
        df,
        lateral_threshold_m=4.0,
        map_overlay=overlay,
        lane_assignment_threshold_m=1.0,
    )

    assert events == []
