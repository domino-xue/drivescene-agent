import pandas as pd

from drivescene.detectors.close_following import detect_close_following
from drivescene.maps.overlay import MapOverlay


def _base_close_following_df(
    actor_type: str = "vehicle",
    focal_type: str = "vehicle",
    actor_heading: float = 0.0,
    actor_start_x: float = 8.0,
    actor_velocity_x: float = 8.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal", "front", "focal", "front", "focal", "front"],
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 0, 1, 1, 2, 2],
            "position_x": [
                0.0,
                actor_start_x,
                1.0,
                actor_start_x + actor_velocity_x * 0.1,
                2.0,
                actor_start_x + actor_velocity_x * 0.2,
            ],
            "position_y": [0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            "heading": [0.0, actor_heading, 0.0, actor_heading, 0.0, actor_heading],
            "velocity_x": [10.0, actor_velocity_x, 10.0, actor_velocity_x, 10.0, actor_velocity_x],
            "velocity_y": [0.0] * 6,
            "object_type": [focal_type, actor_type, focal_type, actor_type, focal_type, actor_type],
        }
    )


def test_detect_close_following_finds_front_actor_with_sustained_low_distance() -> None:
    df = _base_close_following_df()

    events = detect_close_following(
        df,
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert len(events) == 1
    assert events[0]["scenario_id"] == "scene-1"
    assert events[0]["event_type"] == "close_following"
    assert events[0]["track_id"] == "focal"
    assert events[0]["actor_id"] == "front"
    assert events[0]["start_timestep"] == 0
    assert events[0]["end_timestep"] == 2
    assert events[0]["min_distance_m"] < 8.1
    assert events[0]["min_front_distance_m"] < 8.1
    assert events[0]["evidence"]["duration_s"] == 0.3


def test_detect_close_following_ignores_pedestrian_primary_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_type="pedestrian"),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_ignores_static_primary_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_type="static"),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_ignores_unknown_primary_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_type="unknown"),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_ignores_non_vehicle_like_focal_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(focal_type="pedestrian"),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_allows_cyclist_primary_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_type="cyclist"),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert len(events) == 1
    assert events[0]["actor_id"] == "front"


def test_detect_close_following_ignores_opposite_direction_actor() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_heading=3.141592653589793),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_requires_short_front_distance_even_with_low_ttc() -> None:
    events = detect_close_following(
        _base_close_following_df(actor_start_x=14.0, actor_velocity_x=0.0),
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_rejects_parallel_actor_outside_front_cone() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal", "parallel", "focal", "parallel", "focal", "parallel"],
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 0, 1, 1, 2, 2],
            "position_x": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "position_y": [0.0, 1.3, 0.0, 1.3, 0.0, 1.3],
            "heading": [0.0] * 6,
            "velocity_x": [10.0] * 6,
            "velocity_y": [0.0] * 6,
            "object_type": ["vehicle"] * 6,
        }
    )

    events = detect_close_following(
        df,
        distance_threshold_m=10.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_rejects_vehicle_in_adjacent_lane() -> None:
    lane_map = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (20.0, 0.0)],
            [(0.0, 1.8), (20.0, 1.8)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )
    df = _base_close_following_df(actor_start_x=8.0, actor_velocity_x=8.0)
    df.loc[df["track_id"] == "front", "position_y"] = 1.8

    events = detect_close_following(
        df,
        map_overlay=lane_map,
        distance_threshold_m=10.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert events == []


def test_detect_close_following_allows_motorcyclist_by_front_cone_without_same_lane() -> None:
    lane_map = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (20.0, 0.0)],
            [(0.0, 1.8), (20.0, 1.8)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )
    df = _base_close_following_df(
        actor_type="motorcyclist",
        actor_start_x=8.0,
        actor_velocity_x=8.0,
    )
    df.loc[df["track_id"] == "front", "position_y"] = 1.8

    events = detect_close_following(
        df,
        map_overlay=lane_map,
        distance_threshold_m=10.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert len(events) == 1
    assert events[0]["actor_id"] == "front"


def test_detect_close_following_sorts_equal_ttc_by_front_distance() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 9,
            "track_id": [
                "focal",
                "near_side",
                "far_center",
                "focal",
                "near_side",
                "far_center",
                "focal",
                "near_side",
                "far_center",
            ],
            "focal_track_id": ["focal"] * 9,
            "timestep": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "position_x": [0.0, 4.0, 4.2, 1.0, 5.0, 5.2, 2.0, 6.0, 6.2],
            "position_y": [0.0, 1.4, 0.0, 0.0, 1.4, 0.0, 0.0, 1.4, 0.0],
            "heading": [0.0] * 9,
            "velocity_x": [10.0] * 9,
            "velocity_y": [0.0] * 9,
            "object_type": ["vehicle"] * 9,
        }
    )

    events = detect_close_following(
        df,
        distance_threshold_m=10.0,
        ttc_threshold_s=2.0,
        lateral_threshold_m=2.0,
        min_duration_s=0.3,
    )

    assert [event["actor_id"] for event in events] == ["near_side", "far_center"]
