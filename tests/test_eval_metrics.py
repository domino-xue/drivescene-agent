from drivescene.eval.review_metrics import compute_review_metrics, compute_review_metrics_for_queue


def test_compute_review_metrics_precision_and_false_positive_breakdown() -> None:
    reviewed = [
        {
            "event_type": "hard_braking",
            "review": {"is_valid_event": True, "false_positive_reason": None},
        },
        {
            "event_type": "hard_braking",
            "review": {"is_valid_event": False, "false_positive_reason": "low_speed_noise"},
        },
        {
            "event_type": "close_following",
            "review": {"is_valid_event": True, "false_positive_reason": None},
        },
    ]

    metrics = compute_review_metrics(reviewed, k=2)

    assert metrics["num_reviewed"] == 3
    assert metrics["precision"] == 2 / 3
    assert metrics["precision_at_2"] == 1 / 2
    assert metrics["valid_rate_by_event_type"]["hard_braking"] == 0.5
    assert metrics["false_positive_breakdown"]["low_speed_noise"] == 1


def test_compute_review_metrics_for_queue_ignores_stale_reviewed_items() -> None:
    queue = [
        {
            "review_id": "000001",
            "scenario_id": "current-scene",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
        },
        {
            "review_id": "000002",
            "scenario_id": "second-scene",
            "event_type": "close_following",
            "track_id": "focal",
            "actor_id": "front",
            "start_timestep": 20,
            "end_timestep": 22,
        },
    ]
    reviewed = [
        {
            "review_id": "000001",
            "scenario_id": "old-scene",
            "event_type": "hard_braking",
            "track_id": "focal",
            "start_timestep": 10,
            "end_timestep": 12,
            "review": {"is_valid_event": False, "false_positive_reason": "old_queue"},
        },
        {
            **queue[0],
            "review": {"is_valid_event": True, "false_positive_reason": None},
        },
        {
            **queue[1],
            "review": {"is_valid_event": None, "false_positive_reason": None},
        },
    ]

    metrics = compute_review_metrics_for_queue(queue, reviewed, k=20)

    assert metrics["num_reviewed"] == 1
    assert metrics["num_valid"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["num_stale_reviewed"] == 1
    assert metrics["num_labeled_in_queue"] == 2
    assert metrics["num_pending_or_unclear_in_queue"] == 1
    assert metrics["false_positive_breakdown"] == {}
