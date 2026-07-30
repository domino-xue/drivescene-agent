from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from drivescene.agent.agent import build_tool_call_graph
from drivescene.agent.tool_registry import build_tool_registry
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


DEFAULT_TOOLS = [
    "EvidenceStore.query_index",
    "AnalysisTools.run_analysis",
    "ArtifactOps.manage_artifacts",
    "TableOps.transform_table",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test model-driven DriveScene tool calls.")
    parser.add_argument(
        "--event-index",
        default="outputs/labeled_event_index.parquet",
        help="Path to event index parquet.",
    )
    parser.add_argument(
        "--scenario-index",
        default="outputs/scenario_index.parquet",
        help="Path to scenario index parquet.",
    )
    parser.add_argument(
        "--scenario-root",
        default="data/val",
        help="Root folder containing Argoverse scenario parquet folders.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DRIVESCENE_AGENT_MODEL", "deepseek-v3.2"),
        help="Chat model name passed to ChatOpenAI.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL. Can also be set with OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--question",
        default=(
            "Summarize the current event index. Use the available tool instead of guessing, "
            "then explain the result briefly."
        ),
        help="User question sent to the agent graph.",
    )
    args = parser.parse_args()

    registry = build_tool_registry(
        evidence_store=EvidenceStore(args.event_index, scenario_index_path=args.scenario_index),
        analysis_tools=AnalysisTools(args.scenario_root),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )
    model_kwargs = {"model": args.model, "temperature": 0}
    if args.base_url:
        model_kwargs["base_url"] = args.base_url
    model = ChatOpenAI(**model_kwargs)
    graph = build_tool_call_graph(model=model, registry=registry, tool_names=DEFAULT_TOOLS)

    state = graph.invoke({"messages": [HumanMessage(content=args.question)]})
    for message in state["messages"]:
        role = message.__class__.__name__
        print(f"\n[{role}]")
        if getattr(message, "tool_calls", None):
            print("tool_calls:", message.tool_calls)
        if isinstance(message, ToolMessage):
            print("tool_name:", message.name)
        print(message.content)

    used_tools = [message.name for message in state["messages"] if isinstance(message, ToolMessage)]
    if not used_tools:
        raise SystemExit("No tool call was observed. Try a more explicit question or check model tool support.")
    print(f"\nObserved tool calls: {used_tools}")


if __name__ == "__main__":
    main()
