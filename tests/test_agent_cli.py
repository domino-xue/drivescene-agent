from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from drivescene.agent.plan_execute import LLMJsonPlanner, PlanEvent, PlanExecuteState, PlanStep
from drivescene.agent.reporting import LLMReporter
from scripts.run_agent_cli import (
    build_planner,
    build_registry_from_paths,
    build_reporter,
    build_stateful_tool_graph,
    format_stream_event,
    parse_confirmation,
)


class FakeStateGraphModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="state graph done")


def test_parse_confirmation_accepts_yes_only() -> None:
    assert parse_confirmation("yes") is True
    assert parse_confirmation(" YES ") is True
    assert parse_confirmation("y") is False
    assert parse_confirmation("no") is False
    assert parse_confirmation("") is False


def test_build_registry_from_paths_loads_event_index(tmp_path: Path) -> None:
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame([{"review_id": "000001", "event_type": "hard_braking"}]).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1", "object_types": "vehicle"}]).to_parquet(
        scenario_index
    )

    registry = build_registry_from_paths(
        event_index=event_index,
        scenario_index=scenario_index,
        scenario_root=tmp_path,
    )

    result = registry["EvidenceStore.query_index"].func(target="events", operation="summary")

    assert result["items"][0]["result"]["num_events"] == 1


def test_build_planner_uses_model_config_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        'model="test-model", api_key="test-key", base_url="https://example.test/v1",',
        encoding="utf-8",
    )
    captured_kwargs = {}

    def fake_model_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    planner = build_planner(
        registry={},
        config_path=config_path,
        model_factory=fake_model_factory,
        use_heuristic_planner=False,
    )

    assert isinstance(planner, LLMJsonPlanner)
    assert captured_kwargs == {
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "temperature": 0,
    }


def test_build_reporter_uses_model_config_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        'model="report-model", api_key="report-key", base_url="https://example.test/v1",',
        encoding="utf-8",
    )
    captured_kwargs = {}
    model = object()

    def fake_model_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return model

    reporter = build_reporter(
        config_path=config_path,
        model_factory=fake_model_factory,
        use_heuristic_planner=False,
    )

    assert isinstance(reporter, LLMReporter)
    assert reporter.model is model
    assert captured_kwargs == {
        "model": "report-model",
        "api_key": "report-key",
        "base_url": "https://example.test/v1",
        "temperature": 0,
    }


def test_build_reporter_uses_deterministic_fallback_for_heuristic_planner() -> None:
    def fake_model_factory(**kwargs):
        raise AssertionError("heuristic mode should not create a reporter model")

    reporter = build_reporter(
        model_factory=fake_model_factory,
        use_heuristic_planner=True,
    )

    assert isinstance(reporter, LLMReporter)
    assert reporter.model is None


def test_build_stateful_tool_graph_uses_model_config(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        'model="graph-model", api_key="graph-key", base_url="https://example.test/v1",',
        encoding="utf-8",
    )
    registry = build_registry_from_paths(
        event_index=_write_event_index(tmp_path),
        scenario_index=_write_scenario_index(tmp_path),
        scenario_root=tmp_path,
    )
    created_models: list[FakeStateGraphModel] = []

    def fake_model_factory(**kwargs):
        model = FakeStateGraphModel(**kwargs)
        created_models.append(model)
        return model

    graph = build_stateful_tool_graph(
        registry=registry,
        config_path=config_path,
        model_factory=fake_model_factory,
    )
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="hello")],
            "blackboard": {},
            "diagnostics": [],
        }
    )

    assert created_models[0].kwargs == {
        "model": "graph-model",
        "api_key": "graph-key",
        "base_url": "https://example.test/v1",
        "temperature": 0,
    }
    assert created_models[0].bound_tools
    assert state["messages"][-1].content == "state graph done"


def test_format_stream_event_describes_step_started() -> None:
    state = PlanExecuteState(
        user_request="汇总",
        plan=[PlanStep(step_id="step_1", goal="汇总", tool_name="EvidenceStore.query_index")],
    )
    event = PlanEvent(type="step_started", state=state, step=state.plan[0], message="汇总")

    text = format_stream_event(event)

    assert "step_1" in text
    assert "开始" in text
    assert "EvidenceStore.query_index" in text


def _write_event_index(tmp_path: Path) -> Path:
    event_index = tmp_path / "event_index.parquet"
    pd.DataFrame([{"review_id": "000001", "event_type": "hard_braking"}]).to_parquet(
        event_index
    )
    return event_index


def _write_scenario_index(tmp_path: Path) -> Path:
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame([{"scenario_id": "scene-1", "object_types": "vehicle"}]).to_parquet(
        scenario_index
    )
    return scenario_index
