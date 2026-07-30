from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


PROJECT_ROOT = Path(__file__).resolve().parent
EVENT_INDEX_PATH = PROJECT_ROOT / "outputs" / "labeled_event_index.parquet"
FALLBACK_EVENT_INDEX_PATH = PROJECT_ROOT / "outputs" / "event_index.parquet"


def get_event_index_path() -> Path:
    if EVENT_INDEX_PATH.exists():
        return EVENT_INDEX_PATH
    return FALLBACK_EVENT_INDEX_PATH


@tool
def summarize_event_index() -> str:
    """当用户想了解当前自动驾驶事件索引的整体情况时调用，例如事件总数、有效事件数量和事件类型分布。"""
    path = get_event_index_path()
    df = pd.read_parquet(path)
    valid_count = int((df["is_valid_event"] == True).sum()) if "is_valid_event" in df.columns else 0
    event_type_counts = df["event_type"].value_counts().to_dict() if "event_type" in df.columns else {}
    result = {
        "index_path": str(path),
        "num_events": int(len(df)),
        "num_valid": valid_count,
        "valid_rate": valid_count / len(df) if len(df) else 0.0,
        "event_type_counts": event_type_counts,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def search_events(event_type: str = "", limit: int = 3) -> str:
    """当用户想按事件类型查找具体案例时调用。event_type 可传 hard_braking 或 close_following。"""
    path = get_event_index_path()
    df = pd.read_parquet(path)
    if event_type and "event_type" in df.columns:
        df = df[df["event_type"] == event_type]
    columns = [
        column
        for column in [
            "review_id",
            "scenario_id",
            "event_type",
            "track_id",
            "actor_id",
            "is_valid_event",
            "severity",
            "animation_path",
        ]
        if column in df.columns
    ]
    records = df.head(limit)[columns].where(pd.notnull(df.head(limit)[columns]), None).to_dict(
        orient="records"
    )
    return json.dumps(records, ensure_ascii=False)


tools = [summarize_event_index, search_events]
tool_node = ToolNode(tools)
bound_model = None


def should_continue(state: MessagesState) -> Literal["action", "__end__"]:
    """如果模型消息里包含 tool_calls，就进入工具节点；否则结束。"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"


def call_model(state: MessagesState):
    """调用已经绑定工具的模型，让模型自行决定是否调用工具。"""
    if bound_model is None:
        raise RuntimeError("模型尚未初始化，请先调用 build_demo_app()。")
    response = bound_model.invoke(state["messages"])
    return {"messages": response}


from langchain_openai import ChatOpenAI

def build_demo_app():
    """创建真实模型、绑定工具，并编译一个最小 ReAct 风格 LangGraph。"""
    global bound_model

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the real model demo.")

    model = ChatOpenAI(
        model="gpt-5.4",
        api_key=api_key,
        base_url="https://ai.vale.xin/v1",
        temperature=0,
    )
    bound_model = model.bind_tools(tools)

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["action", END])
    workflow.add_edge("action", "agent")
    return workflow.compile(checkpointer=MemorySaver())


def dry_run_tools() -> None:
    print("可用工具：")
    for item in tools:
        print(f"- {item.name}: {item.description}")
    print("\n直接执行 summarize_event_index 工具结果：")
    print(summarize_event_index.invoke({}))


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run_tools()
        raise SystemExit(0)

    # if not os.getenv("OPENAI_API_KEY"):
    #     raise RuntimeError("请先设置 OPENAI_API_KEY，再运行真实模型 demo。")

    app = build_demo_app()
    config = {"configurable": {"thread_id": "demo-1"}}
    input_message = HumanMessage(
        content="给我5个急刹"
    )

    for chunk in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
