from pathlib import Path

import pandas as pd

from drivescene.agent.digest import ExecutionDigest
from drivescene.agent.evaluator import (
    TaskCompletionEvaluator,
    infer_task_requirements,
)
from drivescene.agent.plan_execute import PlanAndExecuteAgent, PlanStep
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


class State:
    def __init__(self, request: str, *, blackboard=None, plan=None) -> None:
        self.user_request = request
        self.pending_confirmation = None
        self.blackboard = blackboard or {}
        self.plan = plan or []


class SearchThenExportPlanner:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.requests: list[str] = []

    def create_plan(self, user_request: str) -> list[PlanStep]:
        self.requests.append(user_request)
        if len(self.requests) == 1:
            return [
                PlanStep(
                    step_id="step_1",
                    goal="search event",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "hard_braking"},
                        "limit": 1,
                    },
                )
            ]
        return [
            PlanStep(
                step_id="step_2",
                goal="export evidence files",
                tool_name="ArtifactOps.manage_artifacts",
                args={
                    "operation": "copy",
                    "paths": {"$from_state": "evidence_paths"},
                    "output_dir": self.output_dir,
                },
            )
        ]


def _digest(**kwargs) -> ExecutionDigest:
    return ExecutionDigest(task="test", **kwargs)


def _registry(tmp_path: Path):
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    asset_dir = tmp_path / "review_assets" / "000001"
    asset_dir.mkdir(parents=True)
    animation_path = asset_dir / "animation.gif"
    metrics_path = asset_dir / "metrics.png"
    trajectory_path = asset_dir / "trajectory_window.png"
    animation_path.write_bytes(b"gif")
    metrics_path.write_bytes(b"png")
    trajectory_path.write_bytes(b"png")
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "is_valid_event": True,
                "animation_path": str(animation_path),
                "metrics_path": str(metrics_path),
                "trajectory_path": str(trajectory_path),
            }
        ]
    ).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    return build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )


def test_requirement_inference_extracts_count_and_evidence_kind() -> None:
    requirements = infer_task_requirements("找 5 个有效急刹案例并给出动画路径")

    assert requirements.event_type == "hard_braking"
    assert requirements.min_events == 5
    assert requirements.require_animation is True


def test_evaluator_requests_replan_when_animation_is_missing() -> None:
    evaluator = TaskCompletionEvaluator()
    digest = _digest(
        events_found=1,
        event_types=["hard_braking"],
        events=[{"review_id": "000001", "event_type": "hard_braking"}],
    )

    result = evaluator.evaluate(State("找 1 个急刹案例并给出动画路径"), digest)

    assert result.needs_replan is True
    assert any("动画路径" in item for item in result.missing_requirements)


def test_evaluator_accepts_requested_event_and_animation() -> None:
    evaluator = TaskCompletionEvaluator()
    digest = _digest(
        events_found=1,
        event_types=["hard_braking"],
        events=[
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "animation_path": "animation.gif",
            }
        ],
    )

    result = evaluator.evaluate(State("找 1 个急刹案例并给出动画路径"), digest)

    assert result.status == "completed"
    assert result.completed is True


def test_evaluator_requires_exported_files_not_only_output_folder() -> None:
    evaluator = TaskCompletionEvaluator()

    result = evaluator.evaluate(
        State("导出急刹事件证据"),
        _digest(output_paths=["outputs/exports/demo"]),
    )

    assert result.needs_replan is True
    assert any("需要实际导出文件" in item for item in result.missing_requirements)


def test_default_evaluator_triggers_real_evidence_replan(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "exports")
    planner = SearchThenExportPlanner(output_dir)
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=planner,
        max_replans=1,
    )

    state = agent.run("找 1 个急刹案例并导出动画证据")

    assert len(planner.requests) == 2
    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.replans_used == 1
    assert state.evaluation["status"] == "completed"
    assert state.execution_digest["files_copied"] == 3
