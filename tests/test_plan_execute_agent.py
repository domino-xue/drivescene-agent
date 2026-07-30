from pathlib import Path

import pandas as pd
import pytest

from drivescene.agent.plan_execute import (
    PlanAndExecuteAgent,
    PlanEvent,
    PlanExecuteState,
    PlanStep,
    _replan_request,
    resolve_references,
)
from drivescene.agent.evaluator import EvaluationResult
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
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
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "is_valid_event": True,
                "animation_path": "outputs/review_assets/000001/animation.gif",
            },
            {
                "review_id": "000002",
                "event_type": "close_following",
                "is_valid_event": False,
                "animation_path": "outputs/review_assets/000002/animation.gif",
            },
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


def test_plan_and_execute_runs_retrieval_steps_in_order(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="summarize event index",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "summary"},
                ),
                PlanStep(
                    step_id="step_2",
                    goal="find valid hard braking events",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "hard_braking", "is_valid_event": True},
                        "limit": 1,
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("summarize events and find one valid hard braking case")

    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert [result.step_id for result in state.step_results] == ["step_1", "step_2"]
    assert state.step_results[0].result["items"][0]["result"]["num_events"] == 2
    assert state.step_results[1].result["items"][0]["result"]["review_id"] == "000001"
    assert state.steps["step_2"]["output"]["items"][0]["result"]["review_id"] == "000001"


def test_resolve_reference_selects_values_from_step_state() -> None:
    state = PlanExecuteState(
        user_request="copy hard braking evidence",
        plan=[],
        steps={
            "step_1": {
                "tool": "EvidenceStore.query_index",
                "output": {
                    "ok": True,
                    "operation": "search",
                    "summary": {"total": 2, "succeeded": 2, "failed": 0},
                    "items": [
                        {"input": {}, "ok": True, "result": {"review_id": "000001"}},
                        {"input": {}, "ok": True, "result": {"review_id": "000002"}},
                    ],
                },
            }
        },
    )

    args = resolve_references(
        {
            "review_ids": {
                "$from_step": "step_1",
                "$select": "output.items[*].result.review_id",
            }
        },
        state,
    )

    assert args["review_ids"] == ["000001", "000002"]


def test_resolve_reference_supports_index_and_wildcard_path_fields() -> None:
    state = PlanExecuteState(
        user_request="copy evidence and write manifest",
        plan=[],
        steps={
            "step_1": {
                "tool": "EvidenceStore.query_index",
                "output": {
                    "ok": True,
                    "operation": "evidence",
                    "items": [
                        {
                            "ok": True,
                            "result": {
                                "review_id": "000001",
                                "animation_path": "assets/000001/animation.gif",
                                "metrics_path": "assets/000001/metrics.png",
                                "trajectory_path": "assets/000001/trajectory.png",
                                "asset_dir": "assets/000001",
                            },
                        },
                        {
                            "ok": True,
                            "result": {
                                "review_id": "000002",
                                "animation_path": "assets/000002/animation.gif",
                                "metrics_path": None,
                                "trajectory_path": "assets/000002/trajectory.png",
                            },
                        },
                    ],
                },
            }
        },
    )

    args = resolve_references(
        {
            "paths": {
                "$from_step": "step_1",
                "$select": "output.items[*].result.*_path",
            },
            "top_event": {
                "$from_step": "step_1",
                "$select": "output.items[0].result",
            },
        },
        state,
    )

    assert args["paths"] == [
        "assets/000001/animation.gif",
        "assets/000001/metrics.png",
        "assets/000001/trajectory.png",
        "assets/000002/animation.gif",
        "assets/000002/trajectory.png",
    ]
    assert args["top_event"]["review_id"] == "000001"


def test_wildcard_path_reference_skips_batch_items_without_matching_fields() -> None:
    state = PlanExecuteState(
        user_request="copy evidence paths",
        plan=[],
        steps={
            "step_1": {
                "tool": "EvidenceStore.query_index",
                "output": {
                    "ok": True,
                    "operation": "evidence",
                    "items": [
                        {"ok": True, "result": {"review_id": "000001"}},
                        {
                            "ok": True,
                            "result": {
                                "review_id": "000002",
                                "animation_path": "assets/000002/animation.gif",
                            },
                        },
                    ],
                },
            }
        },
    )

    args = resolve_references(
        {
            "paths": {
                "$from_step": "step_1",
                "$select": "output.items[*].result.*_path",
            }
        },
        state,
    )

    assert args["paths"] == ["assets/000002/animation.gif"]


def test_resolve_reference_can_read_typed_state_slots() -> None:
    state = PlanExecuteState(
        user_request="copy evidence",
        plan=[],
        blackboard={
            "review_ids": ["000001", "000002"],
            "evidence_paths": ["assets/000001/animation.gif"],
        },
    )

    args = resolve_references(
        {
            "review_ids": {"$from_state": "review_ids"},
            "paths": {"$from_state": "evidence_paths"},
        },
        state,
    )

    assert args == {
        "review_ids": ["000001", "000002"],
        "paths": ["assets/000001/animation.gif"],
    }


def test_executor_resolves_structured_reference_before_tool_call(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
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
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence for searched events",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {
                            "$from_step": "step_1",
                            "$select": "output.items[*].result.review_id",
                        },
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("search then load evidence")

    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.step_results[1].args["review_ids"] == ["000001"]
    assert state.steps["step_2"]["args"]["review_ids"] == ["000001"]
    assert state.step_results[1].result["items"][0]["result"]["review_id"] == "000001"


def test_executor_updates_blackboard_and_resolves_state_reference(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
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
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence from typed state",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("search then load evidence")

    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.step_results[1].args["review_ids"] == ["000001"]
    assert state.blackboard["review_ids"] == ["000001"]
    assert state.blackboard["evidence_paths"] == ["outputs/review_assets/000001/animation.gif"]


def test_executor_can_copy_evidence_paths_selected_by_wildcard_reference(tmp_path: Path) -> None:
    animation = tmp_path / "assets" / "000001" / "animation.gif"
    metrics = tmp_path / "assets" / "000001" / "metrics.png"
    trajectory = tmp_path / "assets" / "000001" / "trajectory.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    trajectory.write_bytes(b"trajectory")
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
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
    output_dir = tmp_path / "copied"
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search hard braking",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "hard_braking"},
                        "limit": 1,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {
                            "$from_step": "step_1",
                            "$select": "output.items[*].result.review_id",
                        },
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="copy evidence",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {
                            "$from_step": "step_2",
                            "$select": "output.items[*].result.*_path",
                        },
                        "output_dir": str(output_dir),
                    },
                    depends_on=["step_2"],
                ),
            ]
        ),
    )

    state = agent.run("copy evidence")

    assert state.plan[-1].status == "completed"
    assert state.step_results[-1].result["num_copied"] == 3
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "animation.gif",
        "metrics.png",
        "trajectory.png",
    ]


def test_executor_binds_copy_paths_from_typed_evidence_state(tmp_path: Path) -> None:
    animation = tmp_path / "assets" / "000001" / "animation.gif"
    metrics = tmp_path / "assets" / "000001" / "metrics.png"
    trajectory = tmp_path / "assets" / "000001" / "trajectory.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    trajectory.write_bytes(b"trajectory")
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
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
    output_dir = tmp_path / "copied"
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search hard braking",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "hard_braking"},
                        "limit": 1,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="copy evidence using typed state",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "output_dir": str(output_dir),
                    },
                    depends_on=["step_2"],
                ),
            ]
        ),
    )

    state = agent.run("copy searched evidence")

    assert state.plan[-1].status == "completed"
    assert state.step_results[-1].args["paths"] == [
        str(animation),
        str(metrics),
        str(trajectory),
    ]
    assert state.step_results[-1].result["num_copied"] == 3
    assert state.blackboard["copied_files"] == [
        str(output_dir / "animation.gif"),
        str(output_dir / "metrics.png"),
        str(output_dir / "trajectory.png"),
    ]


def test_executor_recovers_copy_paths_from_state_when_planner_selector_is_wrong(
    tmp_path: Path,
) -> None:
    animation = tmp_path / "assets" / "000001" / "animation.gif"
    metrics = tmp_path / "assets" / "000001" / "metrics.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "animation_path": str(animation),
                "metrics_path": str(metrics),
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
    output_dir = tmp_path / "copied"
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search hard braking",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "hard_braking"},
                        "limit": 1,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="copy evidence with bad planner selector",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {
                            "$from_step": "step_2",
                            "$select": "output.items[*].result.no_such_path",
                        },
                        "output_dir": str(output_dir),
                    },
                    depends_on=["step_2"],
                ),
            ]
        ),
    )

    state = agent.run("copy evidence despite bad selector")

    assert state.plan[-1].status == "completed"
    assert state.step_results[-1].args["paths"] == [str(animation), str(metrics)]
    assert state.step_results[-1].result["num_copied"] == 2
    assert state.diagnostics[-1]["error"].startswith(
        "Recovered from invalid reference using typed state:"
    )


def test_executor_recovers_manifest_from_state_when_planner_selector_is_wrong(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "manifest.json"
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search close following",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "close_following"},
                        "sort_by": "min_front_distance_m",
                        "ascending": True,
                        "limit": 1,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="write manifest with bad item selector",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "write_manifest",
                        "output_path": str(output_path),
                        "manifest": {
                            "events": {"$from_step": "step_1", "$select": "items"},
                            "top_event": {"$from_step": "step_1", "$select": "items[0]"},
                        },
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("write close following manifest")

    assert state.plan[-1].status == "completed"
    assert state.step_results[-1].args["manifest"]["top_event"]["review_id"] == "000002"
    assert state.step_results[-1].result["manifest_path"] == str(output_path)
    assert state.diagnostics[-1]["error"].startswith(
        "Recovered from invalid reference using typed state:"
    )


def test_executor_recovers_copy_paths_and_output_dir_from_state_when_planner_selectors_are_wrong(
    tmp_path: Path,
) -> None:
    animation = tmp_path / "assets" / "000001" / "animation.gif"
    metrics = tmp_path / "assets" / "000001" / "metrics.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    output_dir = tmp_path / "exports" / "close_following_top5"
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "close_following",
                "min_front_distance_m": 1.2,
                "animation_path": str(animation),
                "metrics_path": str(metrics),
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
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search closest close following events",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "close_following"},
                        "sort_by": "min_front_distance_m",
                        "ascending": True,
                        "limit": 5,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="create export folder",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "create_folder", "output_dir": str(output_dir)},
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="load evidence paths",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_4",
                    goal="copy evidence with bad planner selectors",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {"$from_step": "step_3", "$select": "items[0]"},
                        "output_dir": {"$from_step": "step_2", "$select": "items[0]"},
                    },
                    depends_on=["step_2", "step_3"],
                ),
            ]
        ),
    )

    state = agent.run("copy the closest close_following evidence")

    assert [step.status for step in state.plan] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert state.step_results[-1].args["paths"] == [str(animation), str(metrics)]
    assert state.step_results[-1].args["output_dir"] == str(output_dir)
    assert state.step_results[-1].result["num_copied"] == 2
    assert state.blackboard["copied_files"] == [
        str(output_dir / "animation.gif"),
        str(output_dir / "metrics.png"),
    ]


def test_executor_recovers_copy_and_manifest_from_bare_bad_selectors(
    tmp_path: Path,
) -> None:
    animation = tmp_path / "assets" / "000001" / "animation.gif"
    metrics = tmp_path / "assets" / "000001" / "metrics.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"metrics")
    output_dir = tmp_path / "exports" / "close_following_top7"
    manifest_path = output_dir / "manifest.json"
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "close_following",
                "min_front_distance_m": 1.2,
                "animation_path": str(animation),
                "metrics_path": str(metrics),
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
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search closest close following events",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "close_following"},
                        "sort_by": "min_front_distance_m",
                        "ascending": True,
                        "limit": 7,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence paths",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="create export folder",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "create_folder", "output_dir": str(output_dir)},
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_4",
                    goal="copy evidence with bare bad path selector",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {"$from_step": "step_2", "$select": "*_path"},
                        "output_dir": str(output_dir),
                    },
                    depends_on=["step_2", "step_3"],
                ),
                PlanStep(
                    step_id="step_5",
                    goal="write manifest with bad items selector",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "write_manifest",
                        "output_path": str(manifest_path),
                        "manifest": {
                            "events": {"$from_state": "events"},
                            "top_event": {"$from_step": "step_1", "$select": "items[0]"},
                        },
                    },
                    depends_on=["step_1", "step_3"],
                ),
            ]
        ),
    )

    state = agent.run("copy close following evidence and write manifest")

    assert [step.status for step in state.plan] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert state.step_results[3].args["paths"] == [str(animation), str(metrics)]
    assert state.step_results[3].result["num_copied"] == 2
    assert state.step_results[4].args["manifest"]["top_event"]["review_id"] == "000001"
    assert state.step_results[4].result["manifest_path"] == str(manifest_path)


def test_executor_preserves_event_structure_for_batch_evidence_copy_without_planner_flag(
    tmp_path: Path,
) -> None:
    rows = []
    for review_id in ["000001", "000002"]:
        event_dir = tmp_path / "review_assets" / review_id
        event_dir.mkdir(parents=True)
        animation = event_dir / "animation.gif"
        metrics = event_dir / "metrics.png"
        trajectory = event_dir / "trajectory_window.png"
        animation.write_bytes(f"gif-{review_id}".encode("utf-8"))
        metrics.write_bytes(f"metrics-{review_id}".encode("utf-8"))
        trajectory.write_bytes(f"trajectory-{review_id}".encode("utf-8"))
        rows.append(
            {
                "review_id": review_id,
                "event_type": "close_following",
                "min_front_distance_m": 1.0 if review_id == "000001" else 2.0,
                "animation_path": str(animation),
                "metrics_path": str(metrics),
                "trajectory_path": str(trajectory),
            }
        )
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(rows).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    registry = build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    output_dir = tmp_path / "exports" / "close_following_2_cases"
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search close following",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "close_following"},
                        "sort_by": "min_front_distance_m",
                        "ascending": True,
                        "limit": 2,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="load evidence",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_state": "review_ids"},
                    },
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    goal="create export folder",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "create_folder", "output_dir": str(output_dir)},
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_4",
                    goal="copy evidence without preserve_structure flag",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {"$from_state": "evidence_paths"},
                        "output_dir": str(output_dir),
                    },
                    depends_on=["step_2", "step_3"],
                ),
            ]
        ),
    )

    state = agent.run("copy two close following evidence bundles")

    assert state.plan[-1].status == "completed"
    assert state.step_results[-1].args["preserve_structure"] is True
    assert state.step_results[-1].args["base_dir"] == str(tmp_path / "review_assets")
    assert state.step_results[-1].result["num_copied"] == 6
    copied_files = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*.*"))
    assert copied_files == [
        "000001/animation.gif",
        "000001/metrics.png",
        "000001/trajectory_window.png",
        "000002/animation.gif",
        "000002/metrics.png",
        "000002/trajectory_window.png",
    ]


def test_executor_recovers_top_review_id_from_state_when_planner_uses_items_zero(
    tmp_path: Path,
) -> None:
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "close_following",
                "min_front_distance_m": 1.0,
            },
            {
                "review_id": "000002",
                "event_type": "close_following",
                "min_front_distance_m": 2.0,
            },
        ]
    ).to_parquet(event_index)
    pd.DataFrame([{"scenario_id": "scene-1"}]).to_parquet(scenario_index)
    registry = build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="search close following sorted by distance",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {"event_type": "close_following"},
                        "sort_by": "min_front_distance_m",
                        "ascending": True,
                    },
                ),
                PlanStep(
                    step_id="step_2",
                    goal="return closest event evidence",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "evidence",
                        "review_ids": {"$from_step": "step_1", "$select": "items[0]"},
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("return the closest close_following file")

    assert [step.status for step in state.plan] == ["completed", "completed"]
    assert state.step_results[-1].args["review_ids"] == ["000001"]
    assert state.step_results[-1].result["items"][0]["input"] == {"review_id": "000001"}


def test_replan_does_not_append_diagnostic_no_tool_steps(tmp_path: Path) -> None:
    class DiagnosticReplanPlanner:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def create_plan(self, user_request: str) -> list[PlanStep]:
            self.requests.append(user_request)
            if len(self.requests) == 1:
                return [
                    PlanStep(
                        step_id="step_1",
                        goal="copy paths from bad selector",
                        tool_name="ArtifactOps.manage_artifacts",
                        args={
                            "operation": "copy",
                            "paths": {"$from_step": "missing", "$select": "items[0]"},
                            "output_dir": str(tmp_path / "exports"),
                        },
                    )
                ]
            return [
                PlanStep(
                    step_id="replan_step_4",
                    goal=(
                        "工具执行失败：ArtifactOps.manage_artifacts，错误信息："
                        "Reference selector field not found: items[0]"
                    ),
                    tool_name=None,
                )
            ]

    class AlwaysReplanEvaluator:
        def evaluate(self, state, digest):
            if len(getattr(state, "plan", [])) == 1:
                return EvaluationResult(
                    status="needs_replan",
                    completed=False,
                    needs_replan=True,
                    missing_requirements=["copy evidence paths"],
                )
            return EvaluationResult(status="failed", completed=False)

    planner = DiagnosticReplanPlanner()
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=planner,
        evaluator=AlwaysReplanEvaluator(),
        max_replans=1,
    )

    state = agent.run("copy evidence")

    assert len(planner.requests) == 2
    assert [step.step_id for step in state.plan] == ["step_1"]
    assert all(step.tool_name is not None for step in state.plan)
    assert state.diagnostics[-1]["phase"] == "replan"
    assert state.diagnostics[-1]["error_type"] == "IgnoredNoToolReplanStep"


def test_failed_reference_records_diagnostic_without_fake_completed_replan_step(
    tmp_path: Path,
) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="summarize events",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "summary"},
                ),
                PlanStep(
                    step_id="step_2",
                    goal="copy paths that do not exist in summary output",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={
                        "operation": "copy",
                        "paths": {
                            "$from_step": "step_1",
                            "$select": "output.items[*].result.*_path",
                        },
                        "output_dir": str(tmp_path / "copied"),
                    },
                    depends_on=["step_1"],
                ),
            ]
        ),
    )

    state = agent.run("copy evidence from a bad selector")

    assert [step.step_id for step in state.plan] == ["step_1", "step_2"]
    assert [step.status for step in state.plan] == ["completed", "failed"]
    assert state.diagnostics == [
        {
            "step_id": "step_2",
            "tool_name": "ArtifactOps.manage_artifacts",
            "phase": "resolve_or_validate",
            "error_type": "ValueError",
            "error": "Reference step_1.output.items[*].result.*_path resolved to no values",
        }
    ]
    assert state.steps["step_2"]["output"]["ok"] is False


def test_step_with_failed_dependency_is_skipped_without_stalling(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="invalid legacy placeholder",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "evidence", "review_ids": "$"},
                ),
                PlanStep(
                    step_id="step_2",
                    goal="dependent copy",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "copy", "output_dir": str(tmp_path / "copied")},
                    depends_on=["step_1"],
                ),
            ]
        ),
    )
    state = agent.start("run dependency failure")

    agent._execute_current_step(state)
    agent._execute_current_step(state)

    assert state.current_step == 2
    assert [step.status for step in state.plan] == ["failed", "skipped"]
    assert state.diagnostics[-1] == {
        "step_id": "step_2",
        "tool_name": "ArtifactOps.manage_artifacts",
        "phase": "dependencies",
        "error_type": "DependencyNotCompleted",
        "error": "Step step_2 skipped because dependencies are not completed: step_1",
    }


def test_replan_request_exposes_blackboard_and_diagnostics_separately() -> None:
    state = PlanExecuteState(
        user_request="copy evidence",
        plan=[
            PlanStep(
                step_id="step_1",
                goal="copy bad selector",
                tool_name="ArtifactOps.manage_artifacts",
                status="failed",
            )
        ],
        blackboard={"review_ids": ["000001"], "evidence_paths": ["assets/000001/a.gif"]},
        diagnostics=[
            {
                "step_id": "step_1",
                "tool_name": "ArtifactOps.manage_artifacts",
                "phase": "resolve_or_validate",
                "error_type": "ValueError",
                "error": "bad selector",
            }
        ],
    )

    prompt = _replan_request(
        state,
        EvaluationResult(
            status="needs_replan",
            completed=False,
            needs_replan=True,
            missing_requirements=["需要复制 evidence_paths"],
        ),
        {"events_found": 1},
    )

    assert '"blackboard"' in prompt
    assert '"evidence_paths": ["assets/000001/a.gif"]' in prompt
    assert '"diagnostics"' in prompt
    assert '"allowed_state_references"' in prompt
    assert "Do not reference diagnostic or failed no-tool steps as data sources" in prompt


def test_executor_rejects_legacy_string_placeholder_before_tool_call(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="invalid legacy placeholder",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "evidence", "review_ids": "$"},
                )
            ]
        ),
    )

    state = agent.run("invalid legacy placeholder")

    assert state.plan[0].status == "failed"
    assert state.step_results[0].args["review_ids"] == "$"
    assert "structured reference" in state.step_results[0].error
    assert state.steps["step_1"]["output"]["ok"] is False


def test_stream_emits_plan_step_and_final_answer_events(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="summarize event index",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "summary"},
                )
            ]
        ),
    )

    events = list(agent.stream("summarize event index"))

    assert [event.type for event in events] == [
        "plan_created",
        "step_started",
        "step_completed",
        "final_answer",
    ]
    assert isinstance(events[0], PlanEvent)
    assert events[1].step.step_id == "step_1"
    assert events[2].result.result["items"][0]["result"]["num_events"] == 2
    assert "已完成" in events[-1].state.final_answer
    assert "summarize event index" in events[-1].state.final_answer


def test_heuristic_planner_uses_generic_search_tool(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.run("find 1 valid hard_braking case")

    assert state.plan[0].tool_name == "EvidenceStore.query_index"
    assert state.plan[0].args["target"] == "events"
    assert state.plan[0].args["operation"] == "search"
    assert state.plan[0].args["filters"]["event_type"] == "hard_braking"


def test_heuristic_planner_treats_one_case_as_limit_one(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("给我一个急刹的事件")

    assert state.plan[0].tool_name == "EvidenceStore.query_index"
    assert state.plan[0].args["operation"] == "search"
    assert state.plan[0].args["limit"] == 1


def test_heuristic_planner_searches_new_scene_labels_with_generic_tool(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("find 3 cut_in candidates")

    assert state.plan[0].tool_name == "EvidenceStore.query_index"
    assert state.plan[0].args["target"] == "events"
    assert state.plan[0].args["operation"] == "search"
    assert state.plan[0].args["filters"]["event_type"] == "cut_in"
    assert state.plan[0].args["limit"] == 3


def test_agent_preplans_reviewed_event_count_as_filtered_summary(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("目前已人工标注的急刹事件有多少个？")

    assert len(state.plan) == 1
    assert state.plan[0].tool_name == "EvidenceStore.query_index"
    assert state.plan[0].args == {
        "target": "events",
        "operation": "summary",
        "filters": {"event_type": "hard_braking", "review_status": "reviewed"},
    }


def test_agent_preplans_copy_top_hard_braking_evidence_workflow(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("将5个急刹事件复制到一个新的文件夹吧，并给我其中一个刹停加速度最大的一个事件")

    assert [step.tool_name for step in state.plan] == [
        "EvidenceStore.query_index",
        "EvidenceStore.query_index",
        "ArtifactOps.manage_artifacts",
        "ArtifactOps.manage_artifacts",
        "ArtifactOps.manage_artifacts",
    ]
    assert state.plan[0].args == {
        "target": "events",
        "operation": "search",
        "filters": {"event_type": "hard_braking"},
        "sort_by": "min_velocity_acceleration_mps2",
        "ascending": True,
        "limit": 5,
    }
    assert state.plan[1].args["review_ids"] == {"$from_state": "review_ids"}
    assert state.plan[3].args["paths"] == {"$from_state": "evidence_paths"}
    assert state.plan[3].args["preserve_structure"] is True
    assert state.plan[3].args["base_dir"] == "outputs/review_assets"
    assert state.plan[4].args["manifest"]["events"] == {"$from_state": "events"}
    assert state.plan[4].args["manifest"]["top_event"] == {"$from_state": "top_event"}


def test_agent_preplans_copy_closest_close_following_evidence_workflow(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("将7个近距离跟车事件复制到一个新的文件夹，并给我跟车距离最近的那个文件")

    assert [step.tool_name for step in state.plan] == [
        "EvidenceStore.query_index",
        "EvidenceStore.query_index",
        "ArtifactOps.manage_artifacts",
        "ArtifactOps.manage_artifacts",
        "ArtifactOps.manage_artifacts",
    ]
    assert state.plan[0].args == {
        "target": "events",
        "operation": "search",
        "filters": {"event_type": "close_following"},
        "sort_by": "min_front_distance_m",
        "ascending": True,
        "limit": 7,
    }
    assert state.plan[1].args["review_ids"] == {"$from_state": "review_ids"}
    assert state.plan[3].args["paths"] == {"$from_state": "evidence_paths"}
    assert "close_following_top7" in state.plan[3].args["output_dir"]


def test_agent_migrates_old_state_without_blackboard_before_execution(tmp_path: Path) -> None:
    state = PlanExecuteState(
        user_request="search then evidence",
        plan=[
            PlanStep(
                step_id="step_1",
                goal="search close following",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "search",
                    "filters": {"event_type": "close_following"},
                    "limit": 1,
                },
            ),
            PlanStep(
                step_id="step_2",
                goal="load evidence",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "evidence",
                    "review_ids": {"$from_state": "review_ids"},
                },
                depends_on=["step_1"],
            ),
        ],
    )
    delattr(state, "blackboard")
    delattr(state, "diagnostics")
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path), planner=StaticPlanner([]))

    state = agent.run_until_pause_or_done(state)

    assert state.blackboard["review_ids"] == ["000002"]
    assert state.diagnostics == []


def test_agent_preplans_evidence_query_from_resolved_event_context(tmp_path: Path) -> None:
    context_pack = type(
        "ContextPack",
        (),
        {
            "resolved_context": {"review_ids": ["000001"]},
            "unresolved_references": [],
            "structured_memory": [],
            "thread_summary": "",
        },
    )()
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.start("给我它的文件地址", context_pack=context_pack)

    assert len(state.plan) == 1
    assert state.plan[0].tool_name == "EvidenceStore.query_index"
    assert state.plan[0].args == {
        "target": "events",
        "operation": "evidence",
        "review_ids": ["000001"],
    }


def test_agent_preplans_direct_context_answer(tmp_path: Path) -> None:
    context_pack = type(
        "ContextPack",
        (),
        {
            "resolved_context": {"direct_answer": "你最近主要咨询的是 hard_braking（急刹）事件。"},
            "unresolved_references": [],
            "structured_memory": [],
            "thread_summary": "",
        },
    )()
    agent = PlanAndExecuteAgent(registry=_registry(tmp_path))

    state = agent.run("我咨询你的主要是关于哪类事件？", context_pack=context_pack)

    assert len(state.plan) == 1
    assert state.plan[0].tool_name is None
    assert "hard_braking" in state.final_answer


def test_unknown_tool_name_is_rejected_before_execution(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="try unknown tool",
                    tool_name="os.remove",
                    args={"path": "outputs"},
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="Unknown tool"):
        agent.start("delete outputs")


def test_high_risk_step_pauses_for_confirmation_without_executing(tmp_path: Path) -> None:
    target = tmp_path / "delete_me.txt"
    target.write_text("keep until confirmed", encoding="utf-8")
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="delete temporary file",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "delete", "paths": [str(target)]},
                )
            ]
        ),
    )

    state = agent.run_until_pause_or_done(agent.start("delete temporary file"))

    assert target.exists()
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.step_id == "step_1"
    assert state.plan[0].status == "blocked"
    assert state.step_results == []


def test_stream_emits_confirmation_required_and_pauses(tmp_path: Path) -> None:
    target = tmp_path / "delete_me.txt"
    target.write_text("keep until confirmed", encoding="utf-8")
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="delete temporary file",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "delete", "paths": [str(target)]},
                )
            ]
        ),
    )

    events = list(agent.stream("delete temporary file"))

    assert target.exists()
    assert [event.type for event in events] == [
        "plan_created",
        "step_started",
        "confirmation_required",
    ]
    assert events[-1].pending_confirmation is not None
    assert events[-1].pending_confirmation.step_id == "step_1"


def test_stream_emits_step_skipped_once_when_dependency_is_not_completed(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="wait for missing dependency",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "summary"},
                    depends_on=["missing_step"],
                )
            ]
        ),
    )

    events = list(agent.stream("run plan with missing dependency"))

    assert [event.type for event in events] == [
        "plan_created",
        "step_started",
        "step_skipped",
        "final_answer",
    ]
    assert events[2].step.step_id == "step_1"


def test_confirming_high_risk_step_executes_it(tmp_path: Path) -> None:
    target = tmp_path / "delete_me.txt"
    target.write_text("delete after confirmation", encoding="utf-8")
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="delete temporary file",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "delete", "paths": [str(target)]},
                )
            ]
        ),
    )

    state = agent.run_until_pause_or_done(agent.start("delete temporary file"))
    state = agent.resolve_confirmation(state, approved=True)
    state = agent.run_until_pause_or_done(state)

    assert not target.exists()
    assert state.pending_confirmation is None
    assert state.plan[0].status == "completed"
    assert state.step_results[0].result["num_deleted"] == 1
    assert state.steps["step_1"]["output"]["summary"]["succeeded"] == 1


def test_rejecting_high_risk_step_skips_it_and_finishes(tmp_path: Path) -> None:
    target = tmp_path / "keep_me.txt"
    target.write_text("do not delete", encoding="utf-8")
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="delete temporary file",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "delete", "paths": [str(target)]},
                )
            ]
        ),
    )

    state = agent.run_until_pause_or_done(agent.start("delete temporary file"))
    state = agent.resolve_confirmation(state, approved=False)
    state = agent.run_until_pause_or_done(state)

    assert target.exists()
    assert state.pending_confirmation is None
    assert state.plan[0].status == "skipped"
    assert state.evaluation["status"] == "cancelled"
    assert "取消高风险操作" in state.final_answer


def test_tool_structured_error_is_stored_without_crashing_executor(tmp_path: Path) -> None:
    agent = PlanAndExecuteAgent(
        registry=_registry(tmp_path),
        planner=StaticPlanner(
            [
                PlanStep(
                    step_id="step_1",
                    goal="query missing event",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "detail", "review_ids": ["missing"]},
                )
            ]
        ),
    )

    state = agent.run("query missing event")

    assert state.plan[0].status == "completed"
    assert state.step_results[0].result["ok"] is False
    assert state.step_results[0].result["items"][0]["error_type"] == "KeyError"
    assert state.steps["step_1"]["output"]["items"][0]["input"] == {"review_id": "missing"}
