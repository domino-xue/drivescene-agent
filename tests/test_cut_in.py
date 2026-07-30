import math

import pandas as pd

from drivescene.detectors.cut_in import detect_cut_in
from drivescene.maps.overlay import MapOverlay
from scripts.run_scene_labels import _detect_with_optional_map


def _cut_in_df(
    end_lateral: float = 0.4,
    start_lateral: float = 3.0,
    actor_velocity_x: float = 8.0,
    actor_velocity_y: float = -2.0,
) -> pd.DataFrame:
    rows = []
    for timestep, lateral in enumerate([start_lateral, 2.2, 1.2, 0.8, end_lateral]):
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0,
                "velocity_y": 0.0,
                "position_x": timestep * 1.0,
                "position_y": 0.0,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "actor",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": actor_velocity_x,
                "velocity_y": actor_velocity_y,
                "position_x": timestep * 0.8 + 8.0,
                "position_y": lateral,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )
    return pd.DataFrame(rows)


def test_detect_cut_in_finds_actor_entering_front_corridor() -> None:
    events = detect_cut_in(_cut_in_df())

    assert len(events) == 1
    assert events[0]["event_type"] == "cut_in"
    assert events[0]["actor_id"] == "actor"
    assert events[0]["evidence"]["start_lateral_offset_m"] > 1.5
    assert events[0]["evidence"]["end_lateral_offset_m"] <= 1.5


def test_detect_cut_in_requires_lateral_entry() -> None:
    events = detect_cut_in(_cut_in_df(start_lateral=1.0, end_lateral=0.5))

    assert events == []


def test_detect_cut_in_keeps_slow_moving_actor_above_default_speed() -> None:
    events = detect_cut_in(
        _cut_in_df(actor_velocity_x=1.0, actor_velocity_y=-0.2)
    )

    assert len(events) == 1
    assert events[0]["actor_id"] == "actor"
    assert events[0]["evidence"]["max_actor_speed_mps"] >= 1.0


def test_detect_cut_in_rejects_static_front_vehicle_during_focal_turn() -> None:
    rows = []
    headings = [0.0, 0.1, 0.2, 0.3, 0.38]
    for timestep, heading in enumerate(headings):
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": timestep * 1.0,
                "position_y": 0.0,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "static-front",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "position_x": timestep * 1.0 + 8.0,
                "position_y": 3.0,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )

    events = detect_cut_in(pd.DataFrame(rows))

    assert events == []


def test_detect_cut_in_rejects_turn_induced_lateral_motion_without_map_evidence() -> None:
    rows = []
    headings = [0.0, 0.1, 0.2, 0.3, 0.38]
    for timestep, heading in enumerate(headings):
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": timestep * 1.0,
                "position_y": 0.0,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "same-lane-front",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 8.0,
                "velocity_y": 0.0,
                "position_x": timestep * 0.8 + 8.0,
                "position_y": 3.0,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )

    events = detect_cut_in(pd.DataFrame(rows))

    assert events == []


def test_detect_cut_in_keeps_low_curvature_cut_in_with_heading_change() -> None:
    rows = []
    lateral_values = [3.0, 2.7, 2.4, 2.1, 1.8, 1.4, 1.0, 0.7, 0.5, 0.3]
    focal_x = 0.0
    focal_y = 0.0
    for timestep, lateral in enumerate(lateral_values):
        heading = timestep / (len(lateral_values) - 1) * 0.38
        if timestep > 0:
            previous_heading = (timestep - 1) / (len(lateral_values) - 1) * 0.38
            focal_x += 10.0 * math.cos(previous_heading)
            focal_y += 10.0 * math.sin(previous_heading)
        actor_x = focal_x + 8.0 * math.cos(heading) - lateral * math.sin(heading)
        actor_y = focal_y + 8.0 * math.sin(heading) + lateral * math.cos(heading)
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0 * math.cos(heading),
                "velocity_y": 10.0 * math.sin(heading),
                "position_x": focal_x,
                "position_y": focal_y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )
        rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "actor",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0,
                "velocity_y": -1.0,
                "position_x": actor_x,
                "position_y": actor_y,
                "heading": heading,
                "object_type": "vehicle",
            }
        )

    events = detect_cut_in(pd.DataFrame(rows))

    assert len(events) == 1
    assert events[0]["evidence"]["focal_heading_change_deg"] > 15.0
    assert events[0]["evidence"]["focal_curvature_rad_per_m"] < 0.02


def test_detect_cut_in_uses_map_lane_transition_when_available() -> None:
    lane_map = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (20.0, 0.0)],
            [(0.0, 3.0), (20.0, 3.0)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    events = detect_cut_in(_cut_in_df(), map_overlay=lane_map)

    assert len(events) == 1
    assert events[0]["evidence"]["actor_start_lane_id"] == 1
    assert events[0]["evidence"]["actor_end_lane_id"] == 0
    assert events[0]["evidence"]["focal_end_lane_id"] == 0


def test_scene_label_script_passes_map_to_cut_in_detector(tmp_path) -> None:
    scene_dir = tmp_path / "scene-1"
    scene_dir.mkdir()
    scenario_path = scene_dir / "scenario_scene-1.parquet"
    map_path = scene_dir / "log_map_archive_scene-1.json"
    map_path.write_text(
        """
        {
          "lane_segments": {
            "ego_lane": {
              "centerline": [{"x": 0, "y": 0}, {"x": 20, "y": 0}]
            },
            "far_lane": {
              "centerline": [{"x": 0, "y": 6}, {"x": 20, "y": 6}]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    events = _detect_with_optional_map(detect_cut_in, _cut_in_df(), scenario_path)

    assert len(events) == 1
    assert events[0]["evidence"]["actor_end_lane_id"] == 0
