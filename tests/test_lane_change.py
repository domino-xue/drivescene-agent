import pandas as pd
import math

from drivescene.detectors.lane_change import detect_lane_change
from drivescene.maps.overlay import MapOverlay


def _lane_change_df(total_lateral: float = 3.2) -> pd.DataFrame:
    rows = []
    for timestep in range(8):
        progress = timestep / 7
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0,
                "velocity_y": total_lateral / 0.7,
                "position_x": timestep * 1.5,
                "position_y": progress * total_lateral,
                "heading": progress * 0.08,
                "object_type": "vehicle",
            }
        )
    return pd.DataFrame(rows)


def test_detect_lane_change_finds_sustained_lateral_displacement() -> None:
    events = detect_lane_change(_lane_change_df())

    assert len(events) == 1
    assert events[0]["event_type"] == "lane_change"
    assert events[0]["track_id"] == "focal"
    assert events[0]["evidence"]["lateral_displacement_m"] >= 2.5
    assert events[0]["evidence"]["forward_displacement_m"] >= 5.0


def test_detect_lane_change_rejects_small_lateral_drift() -> None:
    events = detect_lane_change(_lane_change_df(total_lateral=1.0))

    assert events == []


def test_detect_lane_change_rejects_turning_arc_without_lane_transition() -> None:
    rows = []
    radius_m = 10.0
    for timestep in range(11):
        progress = timestep / 10
        heading = progress * math.pi / 2
        rows.append(
            {
                "scenario_id": "scene-turn",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 8.0 * math.cos(heading),
                "velocity_y": 8.0 * math.sin(heading),
                "position_x": radius_m * math.sin(heading),
                "position_y": radius_m * (1.0 - math.cos(heading)),
                "heading": heading,
                "object_type": "vehicle",
            }
        )

    events = detect_lane_change(pd.DataFrame(rows))

    assert events == []


def test_detect_lane_change_rejects_turning_candidate_even_with_map_lane_transition() -> None:
    df = _lane_change_df(total_lateral=3.5)
    df["heading"] = [progress * 0.6 for progress in [step / 7 for step in range(8)]]
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (12.0, 0.0)],
            [(0.0, 3.5), (12.0, 3.5)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(df, map_overlay=overlay, lane_assignment_threshold_m=1.0)

    assert events == []


def test_detect_lane_change_rejects_large_radius_turning_context() -> None:
    rows = []
    radius_m = 30.0
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * math.radians(50.0)
        rows.append(
            {
                "scenario_id": "scene-large-radius-turn",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": radius_m * math.sin(heading),
                "position_y": radius_m * (1.0 - math.cos(heading)),
                "heading": heading,
                "object_type": "vehicle",
            }
        )

    events = detect_lane_change(pd.DataFrame(rows))

    assert events == []


def test_detect_lane_change_rejects_following_curved_lane_centerline() -> None:
    rows = []
    centerline = []
    radius_m = 60.0
    total_heading = math.radians(18.0)
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * total_heading
        x = radius_m * math.sin(heading)
        y = radius_m * (1.0 - math.cos(heading))
        centerline.append((x, y))
        rows.append(
            {
                "scenario_id": "scene-curved-lane",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": x,
                "position_y": y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[centerline],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(pd.DataFrame(rows), map_overlay=overlay)

    assert events == []


def test_detect_lane_change_rejects_curved_lane_following_with_moderate_lane_offset_drift() -> None:
    rows = []
    centerline = []
    radius_m = 60.0
    total_heading = math.radians(18.0)
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * total_heading
        center_x = radius_m * math.sin(heading)
        center_y = radius_m * (1.0 - math.cos(heading))
        normal_x = -math.sin(heading)
        normal_y = math.cos(heading)
        lane_offset = 1.4 * progress
        centerline.append((center_x, center_y))
        rows.append(
            {
                "scenario_id": "scene-curved-lane-offset",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": center_x + lane_offset * normal_x,
                "position_y": center_y + lane_offset * normal_y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[centerline],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(pd.DataFrame(rows), map_overlay=overlay)

    assert events == []


def test_detect_lane_change_rejects_following_curved_lane_split_into_segments() -> None:
    rows = []
    lane_segments = [[], [], []]
    radius_m = 60.0
    total_heading = math.radians(18.0)
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * total_heading
        x = radius_m * math.sin(heading)
        y = radius_m * (1.0 - math.cos(heading))
        segment_index = min(2, int(progress * 3))
        lane_segments[segment_index].append((x, y))
        rows.append(
            {
                "scenario_id": "scene-curved-lane-segments",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": x,
                "position_y": y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=lane_segments,
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(pd.DataFrame(rows), map_overlay=overlay)

    assert events == []


def test_detect_lane_change_rejects_curved_lane_following_with_crossing_lane_outlier() -> None:
    rows = []
    centerline = []
    radius_m = 80.0
    total_heading = math.radians(16.0)
    crossing_anchor = None
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * total_heading
        center_x = radius_m * math.sin(heading)
        center_y = radius_m * (1.0 - math.cos(heading))
        normal_x = -math.sin(heading)
        normal_y = math.cos(heading)
        x = center_x + 0.3 * normal_x
        y = center_y + 0.3 * normal_y
        centerline.append((center_x, center_y))
        if timestep == 32:
            crossing_anchor = (x, y)
        rows.append(
            {
                "scenario_id": "scene-curved-lane-crossing-outlier",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": x,
                "position_y": y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
    assert crossing_anchor is not None
    crossing_x, crossing_y = crossing_anchor
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            centerline,
            [(crossing_x, crossing_y - 6.0), (crossing_x, crossing_y + 6.0)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(pd.DataFrame(rows), map_overlay=overlay)

    assert events == []


def test_detect_lane_change_rejects_continuous_lane_following_with_in_lane_offset_swing() -> None:
    rows = []
    lane_segments = [[], [], []]
    radius_m = 80.0
    total_heading = math.radians(16.0)
    for timestep in range(50):
        progress = timestep / 49
        heading = progress * total_heading
        center_x = radius_m * math.sin(heading)
        center_y = radius_m * (1.0 - math.cos(heading))
        normal_x = -math.sin(heading)
        normal_y = math.cos(heading)
        lane_offset = -0.5 + 1.6 * progress
        segment_index = min(2, int(progress * 3))
        lane_segments[segment_index].append((center_x, center_y))
        rows.append(
            {
                "scenario_id": "scene-curved-lane-in-lane-swing",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": center_x + lane_offset * normal_x,
                "position_y": center_y + lane_offset * normal_y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=lane_segments,
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(pd.DataFrame(rows), map_overlay=overlay)

    assert events == []


def test_detect_lane_change_keeps_straight_map_lane_transition() -> None:
    df = _lane_change_df(total_lateral=3.5)
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (12.0, 0.0)],
            [(0.0, 3.5), (12.0, 3.5)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_lane_change(df, map_overlay=overlay, lane_assignment_threshold_m=1.0)

    assert len(events) == 1
    assert events[0]["evidence"]["lane_transition"] is True
