import json
import math
from pathlib import Path

from drivescene.maps.overlay import MapOverlay, load_map_overlay, nearest_lane_index, project_to_nearest_lane


def test_load_map_overlay_extracts_drivable_lanes_and_crosswalks(tmp_path: Path) -> None:
    map_path = tmp_path / "log_map_archive_scene.json"
    map_path.write_text(
        json.dumps(
            {
                "drivable_areas": {
                    "1": {
                        "id": 1,
                        "area_boundary": [
                            {"x": 0.0, "y": 0.0, "z": 0.0},
                            {"x": 10.0, "y": 0.0, "z": 0.0},
                            {"x": 10.0, "y": 5.0, "z": 0.0},
                        ],
                    }
                },
                "lane_segments": {
                    "2": {
                        "id": 2,
                        "centerline": [
                            {"x": 1.0, "y": 1.0, "z": 0.0},
                            {"x": 8.0, "y": 1.0, "z": 0.0},
                        ],
                        "left_lane_boundary": [
                            {"x": 1.0, "y": 2.0, "z": 0.0},
                            {"x": 8.0, "y": 2.0, "z": 0.0},
                        ],
                        "right_lane_boundary": [
                            {"x": 1.0, "y": 0.0, "z": 0.0},
                            {"x": 8.0, "y": 0.0, "z": 0.0},
                        ],
                    }
                },
                "pedestrian_crossings": {
                    "3": {
                        "id": 3,
                        "edge1": [
                            {"x": 3.0, "y": 3.0, "z": 0.0},
                            {"x": 4.0, "y": 3.0, "z": 0.0},
                        ],
                        "edge2": [
                            {"x": 3.0, "y": 4.0, "z": 0.0},
                            {"x": 4.0, "y": 4.0, "z": 0.0},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    overlay = load_map_overlay(map_path)

    assert overlay.drivable_areas == [[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]]
    assert overlay.lane_centerlines == [[(1.0, 1.0), (8.0, 1.0)]]
    assert overlay.lane_boundaries == [[(1.0, 2.0), (8.0, 2.0)], [(1.0, 0.0), (8.0, 0.0)]]
    assert overlay.crosswalk_edges == [[(3.0, 3.0), (4.0, 3.0)], [(3.0, 4.0), (4.0, 4.0)]]


def test_nearest_lane_index_uses_centerline_distance_threshold() -> None:
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (10.0, 0.0)],
            [(0.0, 3.5), (10.0, 3.5)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    assert nearest_lane_index(overlay, (4.0, 0.4), max_distance_m=1.0) == 0
    assert nearest_lane_index(overlay, (4.0, 3.1), max_distance_m=1.0) == 1
    assert nearest_lane_index(overlay, (4.0, 6.0), max_distance_m=1.0) is None


def test_project_to_nearest_lane_reports_signed_offset_progress_and_tangent() -> None:
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (10.0, 0.0)],
            [(0.0, 4.0), (0.0, 10.0)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    projection = project_to_nearest_lane(overlay, (4.0, 1.0), max_distance_m=2.0)

    assert projection is not None
    assert projection.lane_id == 0
    assert round(projection.distance_m, 3) == 1.0
    assert round(projection.signed_lateral_offset_m, 3) == 1.0
    assert round(projection.progress_m, 3) == 4.0
    assert round(projection.tangent_heading_rad, 3) == 0.0


def test_project_to_nearest_lane_can_require_heading_alignment() -> None:
    overlay = MapOverlay(
        drivable_areas=[],
        lane_centerlines=[
            [(0.0, 0.0), (10.0, 0.0)],
            [(5.0, -5.0), (5.0, 5.0)],
        ],
        lane_boundaries=[],
        crosswalk_edges=[],
    )

    projection = project_to_nearest_lane(
        overlay,
        (5.0, 0.3),
        max_distance_m=1.0,
        heading_rad=0.0,
        max_heading_delta_deg=30.0,
    )

    assert projection is not None
    assert projection.lane_id == 0
    assert math.isclose(projection.distance_m, 0.3)
