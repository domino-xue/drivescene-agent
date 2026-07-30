from pathlib import Path

import pandas as pd

from drivescene.evidence.index import build_event_index, build_scenario_index


def test_build_event_index_flattens_events_reviews_and_asset_paths(tmp_path: Path) -> None:
    queue = [
        {
            "review_id": "000001",
            "scenario_id": "scene-1",
            "city": "austin",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
            "min_acceleration_mps2": -5.5,
            "evidence": {
                "duration_s": 0.3,
                "speed_before_mps": 8.0,
                "speed_after_mps": 3.0,
                "min_velocity_acceleration_mps2": -5.5,
                "min_position_acceleration_mps2": -4.0,
            },
        },
        {
            "review_id": "000002",
            "scenario_id": "scene-2",
            "city": "miami",
            "event_type": "close_following",
            "track_id": "ego",
            "actor_id": "front",
            "start_timestep": 20,
            "end_timestep": 25,
            "min_front_distance_m": 4.2,
            "min_ttc_s": 0.8,
            "evidence": {
                "duration_s": 0.6,
                "min_lateral_offset_m": 0.5,
                "max_closing_speed_mps": 6.5,
            },
        },
    ]
    reviewed = [
        {
            **queue[0],
            "review_status": "reviewed",
            "review": {
                "is_valid_event": True,
                "correct_event_type": "hard_braking",
                "severity": "high",
                "correct_start_timestep": 10,
                "correct_end_timestep": 12,
                "false_positive_reason": None,
                "review_note": "strong braking",
            },
        },
        {
            **queue[1],
            "scenario_id": "old-scene",
            "review_status": "reviewed",
            "review": {
                "is_valid_event": False,
                "correct_event_type": "close_following",
                "severity": None,
                "correct_start_timestep": 20,
                "correct_end_timestep": 25,
                "false_positive_reason": "old_queue",
                "review_note": "",
            },
        },
    ]

    index = build_event_index(queue, reviewed, asset_root=tmp_path)

    assert index["review_id"].tolist() == ["000001", "000002"]
    assert index.loc[0, "event_identity"] == "scene-1|hard_braking|focal||10|12"
    assert index.loc[0, "is_valid_event"] is True
    assert index.loc[0, "severity"] == "high"
    assert index.loc[0, "min_velocity_acceleration_mps2"] == -5.5
    assert index.loc[0, "min_position_acceleration_mps2"] == -4.0
    assert index.loc[0, "animation_path"] == str(tmp_path / "000001" / "animation.gif")
    assert index.loc[1, "is_valid_event"] is None
    assert index.loc[1, "false_positive_reason"] is None
    assert index.loc[1, "min_ttc_s"] == 0.8
    assert index.loc[1, "max_closing_speed_mps"] == 6.5


def test_build_event_index_marks_asset_existence(tmp_path: Path) -> None:
    asset_dir = tmp_path / "000001"
    asset_dir.mkdir()
    (asset_dir / "animation.gif").write_bytes(b"gif")
    queue = [
        {
            "review_id": "000001",
            "scenario_id": "scene-1",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 1,
            "end_timestep": 2,
        }
    ]

    index = build_event_index(queue, [], asset_root=tmp_path)

    assert index.loc[0, "has_animation"] is True
    assert index.loc[0, "has_metrics"] is False


def test_build_event_index_matches_review_when_queue_id_changes(tmp_path: Path) -> None:
    queue = [
        {
            "review_id": "000001",
            "scenario_id": "scene-1",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
        }
    ]
    reviewed = [
        {
            **queue[0],
            "review_id": "000099",
            "review_status": "reviewed",
            "review": {
                "is_valid_event": True,
                "correct_event_type": "hard_braking",
                "severity": "high",
                "correct_start_timestep": 10,
                "correct_end_timestep": 12,
                "false_positive_reason": None,
                "review_note": "old queue id",
            },
        }
    ]

    index = build_event_index(queue, reviewed, asset_root=tmp_path)

    assert index.loc[0, "review_id"] == "000001"
    assert index.loc[0, "is_valid_event"] is True
    assert index.loc[0, "review_status"] == "reviewed"
    assert index.loc[0, "review_note"] == "old queue id"


def test_build_scenario_index_summarizes_scene_files(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scene-1"
    scenario_dir.mkdir()
    scene_path = scenario_dir / "scenario_scene-1.parquet"
    map_path = scenario_dir / "log_map_archive_scene-1.json"
    pd.DataFrame(
        {
            "scenario_id": ["scene-1", "scene-1", "scene-1"],
            "track_id": ["focal", "front", "focal"],
            "focal_track_id": ["focal", "focal", "focal"],
            "object_type": ["vehicle", "pedestrian", "vehicle"],
            "object_category": [3, 3, 3],
            "timestep": [0, 0, 1],
            "city": ["austin", "austin", "austin"],
        }
    ).to_parquet(scene_path)
    map_path.write_text("{}", encoding="utf-8")

    index = build_scenario_index([scene_path])

    assert index.loc[0, "scenario_id"] == "scene-1"
    assert index.loc[0, "city"] == "austin"
    assert index.loc[0, "num_tracks"] == 2
    assert index.loc[0, "num_rows"] == 3
    assert index.loc[0, "num_timestamps"] == 2
    assert index.loc[0, "object_types"] == "pedestrian,vehicle"
    assert index.loc[0, "has_map"] is True
    assert index.loc[0, "scenario_path"] == str(scene_path)
