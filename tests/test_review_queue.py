from pathlib import Path

from drivescene.review.queue import build_review_queue, load_jsonl, write_jsonl


def test_build_review_queue_combines_and_ranks_events(tmp_path: Path) -> None:
    hard = tmp_path / "hard.jsonl"
    close = tmp_path / "close.jsonl"
    write_jsonl(
        hard,
        [
            {
                "scenario_id": "hard-1",
                "event_type": "hard_braking",
                "track_id": "focal",
                "min_acceleration_mps2": -6.0,
            }
        ],
    )
    write_jsonl(
        close,
        [
            {
                "scenario_id": "close-1",
                "event_type": "close_following",
                "track_id": "focal",
                "actor_id": "front",
                "min_distance_m": 3.0,
                "min_front_distance_m": 9.0,
                "min_ttc_s": 0.8,
            },
            {
                "scenario_id": "close-2",
                "event_type": "close_following",
                "track_id": "focal",
                "actor_id": "front",
                "min_distance_m": 4.0,
                "min_front_distance_m": 4.0,
                "min_ttc_s": 0.8,
            }
        ],
    )

    queue = build_review_queue([hard, close], per_type_limit=1)

    assert len(queue) == 2
    assert {item["review_status"] for item in queue} == {"unreviewed"}
    assert {item["review"]["is_valid_event"] for item in queue} == {None}
    assert queue[0]["review_id"] == "000001"
    assert queue[1]["review_id"] == "000002"
    assert queue[0]["scenario_id"] == "close-2"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"

    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert load_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_build_review_queue_includes_new_scene_labels(tmp_path: Path) -> None:
    cut_in = tmp_path / "cut_in.jsonl"
    stopped = tmp_path / "stopped.jsonl"
    lane = tmp_path / "lane.jsonl"
    write_jsonl(
        cut_in,
        [
            {
                "scenario_id": "cut-1",
                "event_type": "cut_in",
                "track_id": "focal",
                "actor_id": "actor",
                "min_front_distance_m": 8.0,
                "min_ttc_s": 2.0,
            }
        ],
    )
    write_jsonl(
        stopped,
        [
            {
                "scenario_id": "stop-1",
                "event_type": "stopped_vehicle_ahead",
                "track_id": "focal",
                "actor_id": "stopped",
                "min_front_distance_m": 12.0,
                "min_ttc_s": 1.5,
            }
        ],
    )
    write_jsonl(
        lane,
        [
            {
                "scenario_id": "lane-1",
                "event_type": "lane_change",
                "track_id": "focal",
                "lateral_displacement_m": 3.2,
            }
        ],
    )

    queue = build_review_queue([cut_in, stopped, lane], per_type_limit=10)

    assert {item["event_type"] for item in queue} == {
        "cut_in",
        "stopped_vehicle_ahead",
        "lane_change",
    }
    assert {item["review_status"] for item in queue} == {"unreviewed"}


def test_build_review_queue_preserves_existing_review_by_event_identity(tmp_path: Path) -> None:
    hard = tmp_path / "hard.jsonl"
    write_jsonl(
        hard,
        [
            {
                "scenario_id": "hard-1",
                "event_type": "hard_braking",
                "track_id": "focal",
                "actor_id": "",
                "start_timestep": 10,
                "end_timestep": 12,
                "min_acceleration_mps2": -6.0,
            }
        ],
    )
    reviewed = [
        {
            "review_id": "000099",
            "scenario_id": "hard-1",
            "event_type": "hard_braking",
            "track_id": "focal",
            "actor_id": "",
            "start_timestep": 10,
            "end_timestep": 12,
            "review_status": "reviewed",
            "review": {
                "is_valid_event": True,
                "correct_event_type": "hard_braking",
                "severity": "high",
                "correct_start_timestep": 10,
                "correct_end_timestep": 12,
                "false_positive_reason": None,
                "review_note": "kept after rebuild",
            },
        }
    ]

    queue = build_review_queue([hard], reviewed_items=reviewed)

    assert queue[0]["review_id"] == "000001"
    assert queue[0]["review_status"] == "reviewed"
    assert queue[0]["review"]["is_valid_event"] is True
    assert queue[0]["review"]["review_note"] == "kept after rebuild"
