from pathlib import Path

import pandas as pd

from drivescene.agent.evaluator import EvaluationResult
from drivescene.agent.plan_execute import PlanAndExecuteAgent, PlanStep
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


class TwoStagePlanner:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.requests: list[str] = []

    def create_plan(self, user_request: str) -> list[PlanStep]:
        self.requests.append(user_request)
        if len(self.requests) == 1:
            return [
                PlanStep(
                    step_id="step_1",
                    goal="search hard braking events",
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
                goal="create export folder",
                tool_name="ArtifactOps.manage_artifacts",
                args={"operation": "create_folder", "output_dir": self.output_dir},
            )
        ]


class NeedsExportFolderEvaluator:
    def evaluate(self, state, digest):
        if not digest.output_paths:
            return EvaluationResult(
                status="needs_replan",
                completed=False,
                needs_replan=True,
                missing_requirements=["需要创建导出文件夹"],
            )
        return EvaluationResult(status="completed", completed=True)


def _registry(tmp_path: Path):
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame([{"review_id": "000001", "event_type": "hard_braking"}]).to_parquet(
        event_index
    )
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    return build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )


def test_agent_evaluator_can_trigger_one_replan_cycle(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "exports")
    planner = TwoStagePlanner(output_dir)
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=planner,
        evaluator=NeedsExportFolderEvaluator(),
        max_replans=1,
    )

    state = agent.run("查找急刹并导出")

    assert len(planner.requests) == 2
    assert "missing_requirements" in planner.requests[1]
    assert [step.step_id for step in state.plan] == ["step_1", "step_2"]
    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.evaluation["status"] == "completed"
    assert state.execution_digest["output_paths"] == [output_dir]


def test_streaming_agent_can_trigger_one_replan_cycle(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "exports")
    planner = TwoStagePlanner(output_dir)
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=planner,
        evaluator=NeedsExportFolderEvaluator(),
        max_replans=1,
    )

    events = list(agent.stream("find hard braking and export"))
    state = events[-1].state

    assert len(planner.requests) == 2
    assert "missing_requirements" in planner.requests[1]
    assert [event.step.step_id for event in events if event.type == "step_completed"] == [
        "step_1",
        "step_2",
    ]
    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.evaluation["status"] == "completed"
    assert state.execution_digest["output_paths"] == [output_dir]
