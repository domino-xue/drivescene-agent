import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from drivescene.agent.agent import (
    build_stateful_tool_call_graph,
    build_tool_call_graph,
    registry_to_langchain_tools,
)
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


class FakeToolCallingModel:
    def __init__(self) -> None:
        self.bound_tools: list[Any] = []
        self.num_invocations = 0

    def bind_tools(self, tools: list[Any]) -> "FakeToolCallingModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.num_invocations += 1
        if self.num_invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceStore__query_index",
                        "args": {"target": "events", "operation": "summary"},
                        "id": "call_summary",
                    }
                ],
            )
        return AIMessage(content="I summarized the event index.")


class FakeStatefulToolCallingModel:
    def __init__(self, output_dir: str) -> None:
        self.bound_tools: list[Any] = []
        self.num_invocations = 0
        self.output_dir = output_dir

    def bind_tools(self, tools: list[Any]) -> "FakeStatefulToolCallingModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.num_invocations += 1
        if self.num_invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceStore__query_index",
                        "args": {
                            "target": "events",
                            "operation": "search",
                            "filters": {"event_type": "close_following"},
                            "sort_by": "min_front_distance_m",
                            "ascending": True,
                            "limit": 1,
                        },
                        "id": "call_search",
                    }
                ],
            )
        if self.num_invocations == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceStore__query_index",
                        "args": {"target": "events", "operation": "evidence"},
                        "id": "call_evidence",
                    }
                ],
            )
        if self.num_invocations == 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ArtifactOps__manage_artifacts",
                        "args": {
                            "operation": "copy",
                            "output_dir": self.output_dir,
                        },
                        "id": "call_copy",
                    }
                ],
            )
        return AIMessage(content="已完成复制，并找到了最近跟车事件。")


class FakeStateReferenceToolCallingModel:
    def __init__(self, output_dir: str) -> None:
        self.bound_tools: list[Any] = []
        self.num_invocations = 0
        self.output_dir = output_dir

    def bind_tools(self, tools: list[Any]) -> "FakeStateReferenceToolCallingModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.num_invocations += 1
        if self.num_invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceStore__query_index",
                        "args": {
                            "target": "events",
                            "operation": "search",
                            "filters": {"event_type": "close_following"},
                            "sort_by": "min_front_distance_m",
                            "ascending": True,
                            "limit": 1,
                        },
                        "id": "call_search",
                    }
                ],
            )
        if self.num_invocations == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "EvidenceStore__query_index",
                        "args": {
                            "target": "events",
                            "operation": "evidence",
                            "review_ids": {"$from_state": "review_ids"},
                        },
                        "id": "call_evidence",
                    }
                ],
            )
        if self.num_invocations == 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ArtifactOps__manage_artifacts",
                        "args": {
                            "operation": "copy",
                            "paths": {"$from_state": "evidence_paths"},
                            "output_dir": self.output_dir,
                        },
                        "id": "call_copy",
                    }
                ],
            )
        return AIMessage(content="done")


class FakeIntentModel:
    def __init__(self) -> None:
        self.num_invocations = 0

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.num_invocations += 1
        return AIMessage(
            content=json.dumps(
                {
                    "event_type": "close_following",
                    "sort_by": "min_front_distance_m",
                    "limit": 1,
                    "deliverables": ["copy_evidence", "report_top_event"],
                },
                ensure_ascii=False,
            )
        )


def _registry(tmp_path: Path):
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {"review_id": "000001", "event_type": "hard_braking", "is_valid_event": True},
            {"review_id": "000002", "event_type": "close_following", "is_valid_event": False},
        ]
    ).to_parquet(event_index)
    pd.DataFrame(
        [
            {
                "scenario_id": "scene-1",
                "city": "austin",
                "object_types": "vehicle",
                "has_map": True,
            }
        ]
    ).to_parquet(scenario_index)
    return build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )


def test_registry_to_langchain_tools_uses_llm_safe_tool_names(tmp_path: Path) -> None:
    tools = registry_to_langchain_tools(
        _registry(tmp_path),
        tool_names=["EvidenceStore.query_index"],
    )

    assert [tool.name for tool in tools] == ["EvidenceStore__query_index"]
    assert "Input:" in tools[0].description
    assert "EvidenceStore.query_index" in tools[0].description


def test_tool_call_graph_executes_model_requested_tool(tmp_path: Path) -> None:
    model = FakeToolCallingModel()
    graph = build_tool_call_graph(
        model=model,
        registry=_registry(tmp_path),
        tool_names=["EvidenceStore.query_index"],
    )

    state = graph.invoke({"messages": [HumanMessage(content="Summarize this batch.")]})

    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    assert model.bound_tools[0].name == "EvidenceStore__query_index"
    assert model.num_invocations == 2
    assert len(tool_messages) == 1

    tool_payload = json.loads(tool_messages[0].content)
    summary = tool_payload["items"][0]["result"]
    assert summary["num_events"] == 2
    assert summary["num_valid"] == 1
    assert summary["event_type_counts"] == {"hard_braking": 1, "close_following": 1}


def test_stateful_tool_call_graph_updates_blackboard_and_binds_missing_args(
    tmp_path: Path,
) -> None:
    animation = tmp_path / "assets" / "000002" / "animation.gif"
    metrics = tmp_path / "assets" / "000002" / "metrics.png"
    trajectory = tmp_path / "assets" / "000002" / "trajectory_window.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    trajectory.write_bytes(b"trajectory")

    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000002",
                "event_type": "close_following",
                "min_front_distance_m": 3.5,
                "animation_path": str(animation),
                "metrics_path": str(metrics),
                "trajectory_path": str(trajectory),
            }
        ]
    ).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    registry = build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    output_dir = tmp_path / "exports"
    model = FakeStatefulToolCallingModel(str(output_dir))
    graph = build_stateful_tool_call_graph(model=model, registry=registry)

    state = graph.invoke(
        {
            "messages": [HumanMessage(content="复制最近的近距离跟车证据文件")],
            "task_frame": {
                "event_type": "close_following",
                "sort_by": "min_front_distance_m",
                "limit": 1,
                "deliverables": ["copy_evidence", "report_top_event"],
            },
            "blackboard": {},
            "diagnostics": [],
        }
    )

    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    assert model.num_invocations == 4
    assert len(tool_messages) == 3
    assert state["blackboard"]["review_ids"] == ["000002"]
    assert state["blackboard"]["top_event"]["review_id"] == "000002"
    assert state["blackboard"]["top_event"]["min_front_distance_m"] == 3.5
    assert state["blackboard"]["evidence_paths"] == [
        str(animation),
        str(metrics),
        str(trajectory),
    ]
    assert state["blackboard"]["copied_files"] == [
        str(output_dir / "animation.gif"),
        str(output_dir / "metrics.png"),
        str(output_dir / "trajectory_window.png"),
    ]
    assert state["diagnostics"] == []


def test_stateful_tool_call_graph_resolves_state_references_without_recovery_diagnostics(
    tmp_path: Path,
) -> None:
    animation = tmp_path / "assets" / "000002" / "animation.gif"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000002",
                "event_type": "close_following",
                "min_front_distance_m": 3.5,
                "animation_path": str(animation),
            }
        ]
    ).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    registry = build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    output_dir = tmp_path / "exports"
    model = FakeStateReferenceToolCallingModel(str(output_dir))
    graph = build_stateful_tool_call_graph(model=model, registry=registry)

    state = graph.invoke(
        {
            "messages": [HumanMessage(content="copy close following evidence")],
            "task_frame": {},
            "blackboard": {},
            "diagnostics": [],
        }
    )

    assert model.num_invocations == 4
    assert state["blackboard"]["review_ids"] == ["000002"]
    assert state["blackboard"]["copied_files"] == [str(output_dir / "animation.gif")]
    assert state["diagnostics"] == []


def test_stateful_tool_call_graph_can_fill_task_frame_with_intent_model(
    tmp_path: Path,
) -> None:
    model = FakeStatefulToolCallingModel(str(tmp_path / "exports"))
    intent_model = FakeIntentModel()
    graph = build_stateful_tool_call_graph(
        model=model,
        registry=_registry(tmp_path),
        tool_names=["EvidenceStore.query_index"],
        intent_model=intent_model,
    )

    state = graph.invoke(
        {
            "messages": [HumanMessage(content="复制最近的近距离跟车证据文件")],
            "blackboard": {},
            "diagnostics": [],
        }
    )

    assert intent_model.num_invocations == 1
    assert state["task_frame"] == {
        "event_type": "close_following",
        "sort_by": "min_front_distance_m",
        "limit": 1,
        "deliverables": ["copy_evidence", "report_top_event"],
    }
