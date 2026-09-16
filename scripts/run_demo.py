from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI

from drivescene.agent.model_config import load_model_config
from drivescene.agent.plan_execute import HeuristicPlanner, LLMJsonPlanner, PlanAndExecuteAgent
from drivescene.agent.reporting import LLMReporter
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


_DEMO_ROOT: Path | None = None


def build_demo_agent(
    *,
    online: bool = False,
    model_config: Path | str = "config/model.yml",
) -> tuple[PlanAndExecuteAgent, Path, str]:
    global _DEMO_ROOT
    if _DEMO_ROOT is None:
        _DEMO_ROOT = Path(tempfile.mkdtemp(prefix="drivescene-demo-"))
        _write_demo_indexes(_DEMO_ROOT)
    registry = build_tool_registry(
        evidence_store=EvidenceStore(
            _DEMO_ROOT / "event_index.parquet",
            scenario_index_path=_DEMO_ROOT / "scenario_index.parquet",
        ),
        analysis_tools=AnalysisTools(_DEMO_ROOT / "scenarios"),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    planner = HeuristicPlanner()
    reporter = LLMReporter()
    runtime_mode = "offline · deterministic planner"
    if online:
        config = load_model_config(model_config)
        if not config.api_key:
            raise RuntimeError(
                "Online demo requires OPENAI_API_KEY. Set it in the current shell, "
                "then rerun with --online."
            )
        model = ChatOpenAI(**config.to_chat_openai_kwargs())
        planner = LLMJsonPlanner(model, registry)
        reporter = LLMReporter(model=model)
        runtime_mode = f"online · {config.model}"

    return (
        PlanAndExecuteAgent(
            registry=registry,
            planner=planner,
            reporter=reporter,
            max_replans=1,
        ),
        _DEMO_ROOT,
        runtime_mode,
    )


def run_demo(
    question: str,
    *,
    online: bool = False,
    model_config: Path | str = "config/model.yml",
) -> dict[str, Any]:
    agent, root, runtime_mode = build_demo_agent(
        online=online,
        model_config=model_config,
    )
    state = agent.run(question)
    return {
        "question": question,
        "runtime_mode": runtime_mode,
        "root": str(root),
        "final_answer": state.final_answer,
        "plan": [
            {
                "step_id": step.step_id,
                "goal": step.goal,
                "tool_name": step.tool_name,
                "status": step.status,
            }
            for step in state.plan
        ],
        "execution_digest": state.execution_digest,
        "evaluation": state.evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DriveScene Agent with built-in sample indexes and no AV2 download."
    )
    parser.add_argument("--question", default="汇总当前事件索引")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use the configured LLM for planning and reporting; requires OPENAI_API_KEY.",
    )
    parser.add_argument("--model-config", default="config/model.yml")
    parser.add_argument("--json", action="store_true", help="Print the complete structured result.")
    args = parser.parse_args()
    try:
        result = run_demo(
            args.question,
            online=args.online,
            model_config=args.model_config,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    print("[DriveScene Demo]")
    print(f"runtime: {result['runtime_mode']}")
    print(f"question: {result['question']}")
    for step in result["plan"]:
        print(f"[{step['status']}] {step['step_id']}: {step['goal']} -> {step['tool_name']}")
    print("\n[final_answer]")
    print(result["final_answer"])


def _write_demo_indexes(root: Path) -> None:
    events = pd.DataFrame(
        [
            _event("000001", "hard_braking", True, -4.8),
            _event("000002", "close_following", True, -2.1),
            _event("000003", "hard_braking", True, -2.7),
        ]
    )
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "demo-scene-1",
                "city": "austin",
                "num_rows": 220,
                "num_tracks": 8,
                "num_timestamps": 110,
                "min_timestep": 0,
                "max_timestep": 109,
                "object_types": "vehicle,pedestrian",
                "has_map": True,
                "scenario_path": "demo-scene-1/scenario.parquet",
                "map_path": "demo-scene-1/map.json",
            }
        ]
    )
    events.to_parquet(root / "event_index.parquet")
    scenarios.to_parquet(root / "scenario_index.parquet")


def _event(review_id: str, event_type: str, valid: bool, acceleration: float) -> dict[str, Any]:
    asset_dir = f"demo/review_assets/{review_id}"
    return {
        "review_id": review_id,
        "event_identity": f"demo-scene-1|{event_type}|{review_id}",
        "scenario_id": "demo-scene-1",
        "city": "austin",
        "event_type": event_type,
        "track_id": "focal",
        "actor_id": "actor-1",
        "start_timestep": 20,
        "end_timestep": 28,
        "duration_s": 0.8,
        "min_distance_m": 8.4,
        "min_front_distance_m": 8.1,
        "min_ttc_s": 1.7,
        "min_lateral_offset_m": 0.4,
        "max_closing_speed_mps": 3.2,
        "min_heading_alignment": 0.98,
        "max_front_angle_deg": 8.0,
        "min_acceleration_mps2": acceleration,
        "speed_before_mps": 12.0,
        "speed_after_mps": 8.0,
        "min_velocity_acceleration_mps2": acceleration,
        "min_position_acceleration_mps2": acceleration + 0.1,
        "is_valid_event": valid,
        "correct_event_type": event_type,
        "severity": "medium",
        "correct_start_timestep": 20,
        "correct_end_timestep": 28,
        "false_positive_reason": None,
        "review_note": "demo fixture",
        "review_status": "reviewed",
        "asset_dir": asset_dir,
        "animation_path": f"{asset_dir}/animation.gif",
        "metrics_path": f"{asset_dir}/metrics.png",
        "trajectory_path": f"{asset_dir}/trajectory_window.png",
        "has_animation": True,
        "has_metrics": True,
        "has_trajectory": True,
    }


if __name__ == "__main__":
    main()
