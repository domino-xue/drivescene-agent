from pathlib import Path

import pandas as pd

from drivescene.analysis.tools import AnalysisTools


def _write_scene(root: Path, df: pd.DataFrame, scenario_id: str = "scene-1") -> Path:
    scene_dir = root / scenario_id
    scene_dir.mkdir(parents=True)
    path = scene_dir / f"scenario_{scenario_id}.parquet"
    df.to_parquet(path)
    return path


def _hard_braking_scene() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal"] * 6,
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 1, 2, 3, 4, 5],
            "velocity_x": [10.0, 9.6, 9.1, 8.6, 8.2, 8.2],
            "velocity_y": [0.0] * 6,
            "position_x": [0.0, 1.0, 1.96, 2.87, 3.73, 4.55],
            "position_y": [0.0] * 6,
            "heading": [0.0] * 6,
            "object_type": ["vehicle"] * 6,
        }
    )


def _close_following_scene() -> pd.DataFrame:
    rows = []
    for timestep in range(5):
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
                "track_id": "front",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 5.0,
                "velocity_y": 0.0,
                "position_x": timestep * 0.5 + 6.0,
                "position_y": 0.2,
                "heading": 0.0,
                "object_type": "vehicle",
            }
        )
    return pd.DataFrame(rows)


def test_analysis_tools_detects_hard_braking_from_batch_scenario_ids(tmp_path: Path) -> None:
    _write_scene(tmp_path, _hard_braking_scene())

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="hard_braking",
        scenario_ids=["scene-1"],
    )

    assert batch["ok"] is True
    assert batch["summary"] == {"total": 1, "succeeded": 1, "failed": 0}
    assert batch["items"][0]["input"] == "scene-1"
    assert batch["items"][0]["ok"] is True
    events = batch["items"][0]["result"]
    assert len(events) == 1
    assert events[0]["event_type"] == "hard_braking"
    assert events[0]["start_timestep"] == 2
    assert events[0]["evidence"]["min_position_acceleration_mps2"] < -3.0


def test_analysis_tools_detects_close_following_from_same_generic_entrypoint(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path, _close_following_scene())

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="close_following",
        scenario_ids=["scene-1"],
    )

    assert batch["ok"] is True
    events = batch["items"][0]["result"]
    assert len(events) == 1
    assert events[0]["actor_id"] == "front"
    assert events[0]["min_front_distance_m"] <= 6.0


def test_analysis_tools_accepts_scenario_paths_and_glob(tmp_path: Path) -> None:
    scene_path = _write_scene(tmp_path, _hard_braking_scene())

    by_path = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="hard_braking",
        scenario_paths=[scene_path],
    )
    by_glob = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="hard_braking",
        scenario_glob=str(tmp_path / "*" / "scenario_*.parquet"),
    )

    assert by_path["summary"] == {"total": 1, "succeeded": 1, "failed": 0}
    assert by_glob["summary"] == {"total": 1, "succeeded": 1, "failed": 0}


def test_analysis_tools_computes_track_kinematics(tmp_path: Path) -> None:
    _write_scene(
        tmp_path,
        pd.DataFrame(
            {
                "scenario_id": ["scene-1"] * 3,
                "track_id": ["focal"] * 3,
                "focal_track_id": ["focal"] * 3,
                "timestep": [0, 1, 2],
                "velocity_x": [3.0, 2.0, 1.0],
                "velocity_y": [4.0, 0.0, 0.0],
                "position_x": [0.0, 0.5, 0.7],
                "position_y": [0.0, 0.0, 0.0],
                "heading": [0.0] * 3,
                "object_type": ["vehicle"] * 3,
            }
        ),
    )

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="compute_track_kinematics",
        scenario_ids=["scene-1"],
        track_id="focal",
    )
    rows = batch["items"][0]["result"]

    assert rows[0]["speed_mps"] == 5.0
    assert "position_speed_mps" in rows[0]
    assert "longitudinal_accel_smooth_mps2" in rows[0]


def test_analysis_tools_computes_relative_motion_for_actor(tmp_path: Path) -> None:
    _write_scene(
        tmp_path,
        pd.DataFrame(
            [
                {
                    "scenario_id": "scene-1",
                    "track_id": "focal",
                    "focal_track_id": "focal",
                    "timestep": 0,
                    "velocity_x": 10.0,
                    "velocity_y": 0.0,
                    "position_x": 0.0,
                    "position_y": 0.0,
                    "heading": 0.0,
                    "object_type": "vehicle",
                },
                {
                    "scenario_id": "scene-1",
                    "track_id": "front",
                    "focal_track_id": "focal",
                    "timestep": 0,
                    "velocity_x": 5.0,
                    "velocity_y": 0.0,
                    "position_x": 6.0,
                    "position_y": 0.5,
                    "heading": 0.0,
                    "object_type": "vehicle",
                },
            ]
        ),
    )

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="compute_relative_motion",
        scenario_ids=["scene-1"],
        actor_id="front",
    )
    rows = batch["items"][0]["result"]

    assert len(rows) == 1
    assert rows[0]["front_projection_m"] == 6.0
    assert rows[0]["ttc_s"] == 1.2


def test_analysis_tools_compares_velocity_position_evidence(tmp_path: Path) -> None:
    _write_scene(tmp_path, _hard_braking_scene())

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="compare_velocity_position_evidence",
        scenario_ids=["scene-1"],
        track_id="focal",
        window={"start_timestep": 2, "end_timestep": 4},
    )
    evidence = batch["items"][0]["result"]

    assert evidence["supports_hard_braking"] is True
    assert evidence["min_velocity_acceleration_mps2"] < -3.0
    assert evidence["min_position_acceleration_mps2"] < -3.0


def test_analysis_tools_reports_partial_batch_errors_without_stopping(tmp_path: Path) -> None:
    _write_scene(tmp_path, _hard_braking_scene())

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="hard_braking",
        scenario_ids=["scene-1", "missing-scene"],
    )

    assert batch["ok"] is False
    assert batch["summary"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert batch["items"][0]["ok"] is True
    assert batch["items"][1]["input"] == "missing-scene"
    assert batch["items"][1]["ok"] is False
    assert batch["items"][1]["error_type"] == "FileNotFoundError"


def test_analysis_tools_rejects_ambiguous_batch_inputs_as_structured_error(tmp_path: Path) -> None:
    batch = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="hard_braking",
        scenario_ids=["scene-1"],
        scenario_paths=[tmp_path / "scene-1.parquet"],
    )

    assert batch["ok"] is False
    assert batch["summary"] == {"total": 0, "succeeded": 0, "failed": 1}
    assert batch["items"][0]["error_type"] == "ValueError"
    assert "exactly one" in batch["items"][0]["error"]


def test_analysis_tools_rejects_unknown_operation_as_structured_error(tmp_path: Path) -> None:
    batch = AnalysisTools(tmp_path).run_analysis(operation="unknown", scenario_ids=["scene-1"])

    assert batch["ok"] is False
    assert batch["items"][0]["error_type"] == "ValueError"
    assert "Unknown analysis operation" in batch["items"][0]["error"]


def test_analysis_tools_detects_new_parameterized_scene_labels(tmp_path: Path) -> None:
    cut_in_rows = []
    for timestep, lateral in enumerate([3.0, 2.2, 1.2, 0.8, 0.4]):
        cut_in_rows.extend(
            [
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
                },
                {
                    "scenario_id": "scene-1",
                    "track_id": "actor",
                    "focal_track_id": "focal",
                    "timestep": timestep,
                    "velocity_x": 8.0,
                    "velocity_y": -2.0,
                    "position_x": timestep * 0.8 + 8.0,
                    "position_y": lateral,
                    "heading": 0.0,
                    "object_type": "vehicle",
                },
            ]
        )
    _write_scene(tmp_path, pd.DataFrame(cut_in_rows), scenario_id="cut-in-scene")

    stopped_rows = []
    for timestep in range(6):
        stopped_rows.extend(
            [
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
                },
                {
                    "scenario_id": "scene-1",
                    "track_id": "stopped",
                    "focal_track_id": "focal",
                    "timestep": timestep,
                    "velocity_x": 0.2,
                    "velocity_y": 0.0,
                    "position_x": 12.0,
                    "position_y": 0.5,
                    "heading": 0.0,
                    "object_type": "vehicle",
                },
            ]
        )
    _write_scene(tmp_path, pd.DataFrame(stopped_rows), scenario_id="stopped-scene")

    lane_rows = []
    for timestep in range(8):
        progress = timestep / 7
        lane_rows.append(
            {
                "scenario_id": "scene-1",
                "track_id": "focal",
                "focal_track_id": "focal",
                "timestep": timestep,
                "velocity_x": 10.0,
                "velocity_y": 4.5,
                "position_x": timestep * 1.5,
                "position_y": progress * 3.2,
                "heading": progress * 0.08,
                "object_type": "vehicle",
            }
        )
    _write_scene(tmp_path, pd.DataFrame(lane_rows), scenario_id="lane-scene")

    cases = {
        "cut_in": "cut-in-scene",
        "stopped_vehicle_ahead": "stopped-scene",
        "lane_change": "lane-scene",
    }
    for event_type, scenario_id in cases.items():
        batch = AnalysisTools(tmp_path).run_analysis(
            operation="detect_events",
            event_type=event_type,
            scenario_ids=[scenario_id],
        )

        assert batch["summary"]["succeeded"] == 1
        assert batch["items"][0]["result"][0]["event_type"] == event_type


def test_analysis_tools_removed_old_public_wrappers() -> None:
    tools = AnalysisTools()

    assert not hasattr(tools, "detect_hard_braking")
    assert not hasattr(tools, "detect_close_following")
    assert not hasattr(tools, "compute_track_kinematics")
    assert not hasattr(tools, "compute_relative_motion")
    assert not hasattr(tools, "compare_velocity_position_evidence")


def test_analysis_tools_passes_map_overlay_to_cut_in_detector(tmp_path: Path) -> None:
    rows = []
    for timestep, lateral in enumerate([3.0, 2.2, 1.2, 0.8, 0.4]):
        rows.extend(
            [
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
                },
                {
                    "scenario_id": "scene-1",
                    "track_id": "actor",
                    "focal_track_id": "focal",
                    "timestep": timestep,
                    "velocity_x": 8.0,
                    "velocity_y": -2.0,
                    "position_x": timestep * 0.8 + 8.0,
                    "position_y": lateral,
                    "heading": 0.0,
                    "object_type": "vehicle",
                },
            ]
        )
    scene_path = _write_scene(tmp_path, pd.DataFrame(rows))
    map_path = scene_path.parent / "log_map_archive_scene-1.json"
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

    batch = AnalysisTools(tmp_path).run_analysis(
        operation="detect_events",
        event_type="cut_in",
        scenario_ids=["scene-1"],
    )

    assert batch["summary"]["succeeded"] == 1
    events = batch["items"][0]["result"]
    assert len(events) == 1
    assert events[0]["evidence"]["focal_start_lane_id"] == 0
    assert events[0]["evidence"]["actor_start_lane_id"] is None
