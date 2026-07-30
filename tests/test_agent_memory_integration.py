from pathlib import Path

import pandas as pd

from drivescene.agent.plan_execute import PlanAndExecuteAgent, PlanStep
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.memory.context import ContextManager
from drivescene.memory.store import MemoryStore
from drivescene.memory.writer import AgentMemoryWriter
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


class StaticPlanner:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    def create_plan(self, user_request: str) -> list[PlanStep]:
        return self.steps


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


def test_agent_records_artifact_tool_outputs_into_thread_memory(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory.sqlite")
    user = memory_store.create_user("alice", "secret")
    thread = memory_store.create_thread(user.id, "Exports")
    output_dir = tmp_path / "hard_braking_cases"
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="create hard braking export folder",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "create_folder", "output_dir": str(output_dir)},
                )
            ]
        ),
        memory_writer=AgentMemoryWriter(memory_store),
    )

    agent.run(
        "帮我创建急刹导出文件夹",
        memory_user_id=user.id,
        memory_thread_id=thread.id,
    )

    matches = memory_store.query_memory(
        user_id=user.id,
        thread_id=thread.id,
        memory_type="path",
        object_kind="folder",
        semantic_tags=["hard_braking"],
    )
    assert [item.value for item in matches] == [str(output_dir)]


def test_agent_uses_resolved_context_paths_for_delete_plan(tmp_path: Path) -> None:
    memory_store = MemoryStore(tmp_path / "memory.sqlite")
    user = memory_store.create_user("alice", "secret")
    thread = memory_store.create_thread(user.id, "Exports")
    target = tmp_path / "hard_braking_cases"
    memory_store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value=str(target),
        object_kind="folder",
        semantic_tags=["hard_braking", "export"],
        description="Hard braking export folder",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )
    context_pack = ContextManager(memory_store).build_context_pack(
        user.id,
        thread.id,
        "将这个文件夹删除",
    )
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("将这个文件夹删除", context_pack=context_pack)

    assert state.plan[0].tool_name == "ArtifactOps.manage_artifacts"
    assert state.plan[0].args["operation"] == "delete"
    assert state.plan[0].args["paths"] == [str(target)]
