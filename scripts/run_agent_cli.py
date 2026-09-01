from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from drivescene.agent.agent import build_stateful_tool_call_graph
from drivescene.agent.model_config import load_model_config
from drivescene.agent.plan_execute import (
    HeuristicPlanner,
    LLMJsonPlanner,
    PlanAndExecuteAgent,
    PlanEvent,
)
from drivescene.agent.reporting import LLMReporter
from drivescene.agent.tool_registry import ToolRegistry, build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


def parse_confirmation(value: str) -> bool:
    return value.strip().lower() == "yes"


def build_registry_from_paths(
    event_index: Path | str,
    scenario_index: Path | str,
    scenario_root: Path | str,
) -> ToolRegistry:
    return build_tool_registry(
        evidence_store=EvidenceStore(event_index, scenario_index_path=scenario_index),
        analysis_tools=AnalysisTools(scenario_root),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )


def build_planner(
    registry: ToolRegistry,
    config_path: Path | str = "config/model.yml",
    model_factory=ChatOpenAI,
    use_heuristic_planner: bool = False,
    planner_prompt_profile: str = "full",
    planner_max_attempts: int = 2,
):
    if use_heuristic_planner:
        return HeuristicPlanner()
    model_config = load_model_config(config_path)
    return LLMJsonPlanner(
        model_factory(**model_config.to_chat_openai_kwargs()),
        registry,
        prompt_profile=planner_prompt_profile,
        max_attempts=planner_max_attempts,
    )


def build_reporter(
    config_path: Path | str = "config/model.yml",
    model_factory=ChatOpenAI,
    use_heuristic_planner: bool = False,
) -> LLMReporter:
    if use_heuristic_planner:
        return LLMReporter()
    model_config = load_model_config(config_path)
    return LLMReporter(model=model_factory(**model_config.to_chat_openai_kwargs()))


def build_stateful_tool_graph(
    registry: ToolRegistry,
    config_path: Path | str = "config/model.yml",
    model_factory=ChatOpenAI,
    intent_model_factory=None,
):
    model_config = load_model_config(config_path)
    model = model_factory(**model_config.to_chat_openai_kwargs())
    intent_model = (
        intent_model_factory(**model_config.to_chat_openai_kwargs())
        if intent_model_factory is not None
        else None
    )
    return build_stateful_tool_call_graph(
        model=model,
        registry=registry,
        intent_model=intent_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveScene Plan-and-Execute agent CLI.")
    parser.add_argument("--question", required=True, help="中文用户问题。")
    parser.add_argument("--event-index", default="outputs/labeled_event_index.parquet")
    parser.add_argument("--scenario-index", default="outputs/scenario_index.parquet")
    parser.add_argument("--scenario-root", default="data/val")
    parser.add_argument("--model-config", default="config/model.yml")
    parser.add_argument(
        "--use-heuristic-planner",
        action="store_true",
        help="使用确定性 fallback planner，而不是 config/model.yml 中配置的模型 planner。",
    )
    parser.add_argument(
        "--use-state-graph",
        action="store_true",
        help="使用 messages + task_frame + blackboard 的 LangGraph tool-calling runtime。",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="Plan-and-Execute 模式允许的最大补充规划次数，默认 1。",
    )
    args = parser.parse_args()
    if args.max_replans < 0:
        parser.error("--max-replans must be non-negative")

    registry = build_registry_from_paths(
        event_index=args.event_index,
        scenario_index=args.scenario_index,
        scenario_root=args.scenario_root,
    )
    if args.use_state_graph:
        graph = build_stateful_tool_graph(
            registry=registry,
            config_path=args.model_config,
        )
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=args.question)],
                "blackboard": {},
                "diagnostics": [],
            }
        )
        print_state_graph_result(state)
        return

    planner = build_planner(
        registry=registry,
        config_path=args.model_config,
        use_heuristic_planner=args.use_heuristic_planner,
    )
    reporter = build_reporter(
        config_path=args.model_config,
        use_heuristic_planner=args.use_heuristic_planner,
    )
    agent = PlanAndExecuteAgent(
        registry=registry,
        planner=planner,
        reporter=reporter,
        max_replans=args.max_replans,
    )

    events = agent.stream(args.question)
    state = None
    while True:
        for event in events:
            state = event.state
            print(format_stream_event(event), flush=True)
            if event.type == "confirmation_required":
                pending = event.pending_confirmation
                print("\n需要人工确认的高风险工具调用：", flush=True)
                print(f"step_id: {pending.step_id}", flush=True)
                print(f"tool: {pending.tool_name}", flush=True)
                print(f"args: {pending.args}", flush=True)
                print(f"risk: {pending.risk['risk_level']} - {pending.risk['reason']}", flush=True)
                approved = parse_confirmation(input("输入 yes 执行，其他任意输入取消："))
                state = agent.resolve_confirmation(state, approved=approved)
                events = agent.stream_from_state(state)
                break
        else:
            break


def print_state_graph_result(state: dict) -> None:
    messages = state.get("messages", [])
    if messages:
        print("\n[final_message]")
        print(getattr(messages[-1], "content", messages[-1]))
    if state.get("task_frame"):
        print("\n[task_frame]")
        print(state["task_frame"])
    if state.get("blackboard"):
        print("\n[blackboard]")
        print(state["blackboard"])
    if state.get("diagnostics"):
        print("\n[diagnostics]")
        print(state["diagnostics"])


def format_stream_event(event: PlanEvent) -> str:
    if event.type == "plan_created":
        return "\n[plan_created]\n" + "\n".join(
            f"- {step.step_id} [{step.status}] {step.goal} -> {step.tool_name or '无工具'} args={step.args}"
            for step in event.state.plan
        )
    if event.type == "step_started" and event.step is not None:
        return (
            f"\n[step_started] 开始 {event.step.step_id}: {event.step.goal} "
            f"-> {event.step.tool_name or '无工具'} args={event.step.args}"
        )
    if event.type == "step_completed" and event.result is not None:
        return f"\n[step_completed] {event.result.step_id}: {event.result.result}"
    if event.type == "step_failed" and event.result is not None:
        return f"\n[step_failed] {event.result.step_id}: {event.result.error}"
    if event.type == "step_skipped" and event.step is not None:
        return f"\n[step_skipped] {event.step.step_id}: dependency not completed"
    if event.type == "confirmation_required":
        return f"\n[confirmation_required]\n{event.message}"
    if event.type == "final_answer":
        return f"\n[final_answer]\n{event.message}"
    return f"\n[{event.type}]\n{event.message}"


def print_plan(plan) -> None:
    print("\n计划：")
    for step in plan:
        tool = step.tool_name or "无工具"
        print(f"- {step.step_id} [{step.status}] {step.goal} -> {tool} args={step.args}")


def print_results(results) -> None:
    if not results:
        return
    print("\n工具结果：")
    for result in results:
        if result.error:
            print(f"- {result.step_id} ERROR {result.error}")
        else:
            print(f"- {result.step_id} {result.tool_name}: {result.result}")


if __name__ == "__main__":
    main()
