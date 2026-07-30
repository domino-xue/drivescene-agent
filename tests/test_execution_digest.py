from drivescene.agent.digest import ExecutionDigestBuilder
from drivescene.agent.plan_execute import PlanExecuteState, PlanStep, StepResult


def test_execution_digest_summarizes_event_search_and_artifact_copy() -> None:
    state = PlanExecuteState(
        user_request="查找 5 条急刹并复制证据",
        plan=[
            PlanStep("step_1", "search hard braking", "EvidenceStore.query_index"),
            PlanStep("step_2", "copy evidence", "ArtifactOps.manage_artifacts"),
        ],
        step_results=[
            StepResult(
                step_id="step_1",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "search",
                    "filters": {"event_type": "hard_braking"},
                },
                result={
                    "ok": True,
                    "operation": "search",
                    "summary": {"total": 5, "succeeded": 5, "failed": 0},
                    "items": [
                        {"ok": True, "result": {"review_id": f"{index:06d}"}}
                        for index in range(1, 6)
                    ],
                },
            ),
            StepResult(
                step_id="step_2",
                tool_name="ArtifactOps.manage_artifacts",
                args={"operation": "copy", "output_dir": "outputs/exports/hard_braking"},
                result={
                    "ok": True,
                    "operation": "copy",
                    "summary": {"total": 15, "succeeded": 15, "failed": 0},
                    "output_dir": "outputs/exports/hard_braking",
                    "copied_files": [f"outputs/exports/hard_braking/{index}.gif" for index in range(15)],
                    "items": [],
                },
            ),
        ],
    )

    digest = ExecutionDigestBuilder().build(state)

    assert digest.status == "completed"
    assert digest.events_found == 5
    assert digest.event_types == ["hard_braking"]
    assert digest.files_copied == 15
    assert digest.output_paths == ["outputs/exports/hard_braking"]
    assert digest.execution_summary == ["search hard braking", "copy evidence"]
    assert digest.to_dict()["files_copied"] == 15


def test_execution_digest_records_failed_items_without_losing_successes() -> None:
    state = PlanExecuteState(
        user_request="复制证据",
        plan=[PlanStep("step_1", "copy evidence", "ArtifactOps.manage_artifacts")],
        step_results=[
            StepResult(
                step_id="step_1",
                tool_name="ArtifactOps.manage_artifacts",
                args={"operation": "copy", "output_dir": "outputs/export"},
                result={
                    "ok": False,
                    "operation": "copy",
                    "summary": {"total": 2, "succeeded": 1, "failed": 1},
                    "output_dir": "outputs/export",
                    "copied_files": ["outputs/export/a.gif"],
                    "items": [
                        {"input": {"path": "a.gif"}, "ok": True, "result": {}},
                        {
                            "input": {"path": "missing.gif"},
                            "ok": False,
                            "error_type": "FileNotFoundError",
                            "error": "missing.gif",
                        },
                    ],
                },
            )
        ],
    )

    digest = ExecutionDigestBuilder().build(state)

    assert digest.status == "partial"
    assert digest.files_copied == 1
    assert digest.failed_items == [
        {
            "step_id": "step_1",
            "tool_name": "ArtifactOps.manage_artifacts",
            "input": {"path": "missing.gif"},
            "error_type": "FileNotFoundError",
            "error": "missing.gif",
        }
    ]


def test_execution_digest_keeps_summary_counts_and_event_samples() -> None:
    state = PlanExecuteState(
        user_request="目前已人工标注的急刹事件有多少个？",
        plan=[
            PlanStep("step_1", "count reviewed hard braking events", "EvidenceStore.query_index"),
            PlanStep("step_2", "search one hard braking event", "EvidenceStore.query_index"),
        ],
        step_results=[
            StepResult(
                step_id="step_1",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "summary",
                    "filters": {"event_type": "hard_braking", "review_status": "reviewed"},
                },
                result={
                    "ok": True,
                    "operation": "summary",
                    "summary": {"total": 1, "succeeded": 1, "failed": 0},
                    "items": [
                        {
                            "input": {"filters": {"event_type": "hard_braking"}},
                            "ok": True,
                            "result": {
                                "num_events": 12,
                                "num_valid": 5,
                                "event_type_counts": {"hard_braking": 12},
                                "filters": {
                                    "event_type": "hard_braking",
                                    "review_status": "reviewed",
                                },
                            },
                        }
                    ],
                },
            ),
            StepResult(
                step_id="step_2",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "search",
                    "filters": {"event_type": "hard_braking"},
                    "limit": 1,
                },
                result={
                    "ok": True,
                    "operation": "search",
                    "summary": {"total": 1, "succeeded": 1, "failed": 0},
                    "items": [
                        {
                            "input": {"filters": {"event_type": "hard_braking"}},
                            "ok": True,
                            "result": {
                                "review_id": "000123",
                                "event_type": "hard_braking",
                                "animation_path": "outputs/review_assets/000123/animation.gif",
                            },
                        }
                    ],
                },
            ),
        ],
    )

    digest = ExecutionDigestBuilder().build(state)

    assert digest.events_found == 13
    assert digest.event_summary["num_events"] == 12
    assert digest.event_summary["filters"]["review_status"] == "reviewed"
    assert digest.events[0]["review_id"] == "000123"
    assert digest.output_paths == ["outputs/review_assets/000123/animation.gif"]


def test_execution_digest_keeps_hard_braking_acceleration_metric() -> None:
    state = PlanExecuteState(
        user_request="给我刹停加速度最大的急刹事件",
        plan=[PlanStep("step_1", "search hardest braking", "EvidenceStore.query_index")],
        step_results=[
            StepResult(
                step_id="step_1",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "search",
                    "filters": {"event_type": "hard_braking"},
                    "sort_by": "min_velocity_acceleration_mps2",
                    "ascending": True,
                    "limit": 1,
                },
                result={
                    "ok": True,
                    "operation": "search",
                    "summary": {"total": 1, "succeeded": 1, "failed": 0},
                    "items": [
                        {
                            "ok": True,
                            "result": {
                                "review_id": "000123",
                                "event_type": "hard_braking",
                                "min_velocity_acceleration_mps2": -7.5,
                            },
                        }
                    ],
                },
            )
        ],
    )

    digest = ExecutionDigestBuilder().build(state)

    assert digest.events[0]["review_id"] == "000123"
    assert digest.events[0]["min_velocity_acceleration_mps2"] == -7.5
