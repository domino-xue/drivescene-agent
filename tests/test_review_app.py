from scripts.review_app import (
    build_reviewed_lookup,
    clamp_review_index,
    next_review_index_after_save,
    upsert_reviewed_item,
)


def test_upsert_reviewed_item_replaces_existing_review_by_identity() -> None:
    reviewed = [
        {
            "review_id": "000001",
            "scenario_id": "scene-1",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
            "review_status": "reviewed",
            "review": {"severity": "low"},
        },
        {
            "review_id": "000002",
            "scenario_id": "scene-2",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 20,
            "end_timestep": 22,
            "review_status": "reviewed",
            "review": {"severity": "medium"},
        },
    ]
    updated_item = {
        "review_id": "000099",
        "scenario_id": "scene-1",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 10,
        "end_timestep": 12,
        "review_status": "reviewed",
        "review": {"severity": "high"},
    }

    result = upsert_reviewed_item(reviewed, updated_item)

    assert len(result) == 2
    assert result[0]["review_id"] == "000099"
    assert result[0]["review"]["severity"] == "high"
    assert result[1]["review_id"] == "000002"


def test_upsert_reviewed_item_appends_new_review() -> None:
    reviewed = [
        {
            "review_id": "000001",
            "scenario_id": "scene-1",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
            "review_status": "reviewed",
        }
    ]
    new_item = {
        "review_id": "000002",
        "scenario_id": "scene-2",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 20,
        "end_timestep": 22,
        "review_status": "reviewed",
    }

    result = upsert_reviewed_item(reviewed, new_item)

    assert [item["review_id"] for item in result] == ["000001", "000002"]


def test_upsert_reviewed_item_does_not_replace_different_event_with_same_review_id() -> None:
    reviewed = [
        {
            "review_id": "000001",
            "scenario_id": "old-scene",
            "event_type": "cut_in",
            "track_id": "focal",
            "actor_id": "old",
            "start_timestep": 1,
            "end_timestep": 2,
            "review_status": "reviewed",
            "review": {"is_valid_event": False},
        }
    ]
    new_item = {
        "review_id": "000001",
        "scenario_id": "new-scene",
        "event_type": "cut_in",
        "track_id": "focal",
        "actor_id": "new",
        "start_timestep": 1,
        "end_timestep": 2,
        "review_status": "reviewed",
        "review": {"is_valid_event": True},
    }

    result = upsert_reviewed_item(reviewed, new_item)

    assert len(result) == 2
    assert result[0]["scenario_id"] == "old-scene"
    assert result[1]["scenario_id"] == "new-scene"


def test_build_reviewed_lookup_ignores_stale_review_id_for_different_event() -> None:
    queue_item = {
        "review_id": "000001",
        "scenario_id": "new-scene",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 10,
        "end_timestep": 12,
    }
    reviewed = [
        {
            "review_id": "000001",
            "scenario_id": "old-scene",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
            "review_status": "reviewed",
        }
    ]

    lookup = build_reviewed_lookup([queue_item], reviewed)

    assert lookup == {}


def test_build_reviewed_lookup_keeps_matching_event_review() -> None:
    queue_item = {
        "review_id": "000001",
        "scenario_id": "scene-1",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 10,
        "end_timestep": 12,
    }
    reviewed = [{**queue_item, "review_status": "reviewed"}]

    lookup = build_reviewed_lookup([queue_item], reviewed)

    assert lookup == {"000001": reviewed[0]}


def test_build_reviewed_lookup_matches_review_when_queue_id_changes() -> None:
    queue_item = {
        "review_id": "000001",
        "scenario_id": "scene-1",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 10,
        "end_timestep": 12,
    }
    reviewed = [
        {
            **queue_item,
            "review_id": "000099",
            "review_status": "reviewed",
            "review": {"is_valid_event": True},
        }
    ]

    lookup = build_reviewed_lookup([queue_item], reviewed)

    assert lookup == {"000001": reviewed[0]}


def test_clamp_review_index_stays_inside_queue_bounds() -> None:
    assert clamp_review_index(-1, queue_size=3) == 0
    assert clamp_review_index(1, queue_size=3) == 1
    assert clamp_review_index(3, queue_size=3) == 2
    assert clamp_review_index(0, queue_size=0) == 0


def test_next_review_index_after_save_advances_without_passing_end() -> None:
    assert next_review_index_after_save(0, queue_size=3) == 1
    assert next_review_index_after_save(1, queue_size=3) == 2
    assert next_review_index_after_save(2, queue_size=3) == 2
    assert next_review_index_after_save(0, queue_size=0) == 0
