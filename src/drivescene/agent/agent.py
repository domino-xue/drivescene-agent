from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from drivescene.agent.plan_execute import (
    PlanExecuteState,
    PlanStep,
    bind_tool_arguments,
    ensure_state_runtime_fields,
    recover_tool_arguments_from_state,
    resolve_references,
    validate_tool_arguments,
    _update_blackboard_from_step,
)
from drivescene.agent.tool_registry import ToolRegistry, execute_registered_tool, preflight_tool_call
from drivescene.ops.contracts import validation_error_result


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_frame: dict[str, Any]
    blackboard: dict[str, Any]
    diagnostics: list[dict[str, Any]]


def registry_tool_name_to_llm_name(tool_name: str) -> str:
    """将 registry 中的工具名转换为更适合 LLM function calling 的安全名称。"""
    return tool_name.replace(".", "__")


def llm_tool_name_to_registry_name(tool_name: str) -> str:
    """将 LLM 调用时使用的安全工具名还原为 registry 中的原始工具名。"""
    return tool_name.replace("__", ".")


def registry_to_langchain_tools(
    registry: ToolRegistry,
    tool_names: Iterable[str] | None = None,
) -> list[StructuredTool]:
    """把 ToolRegistry 中的工具规格转换为 LangChain 可绑定的 StructuredTool 列表。"""
    selected_tool_names = list(tool_names) if tool_names is not None else list(registry)
    tools: list[StructuredTool] = []
    for registry_name in selected_tool_names:
        if registry_name not in registry:
            raise KeyError(f"Unknown tool: {registry_name}")
        spec = registry[registry_name]
        tools.append(
            StructuredTool.from_function(
                func=_json_result_wrapper(spec.func),
                name=registry_tool_name_to_llm_name(registry_name),
                description=f"{spec.description}\n\nRegistry tool name: {registry_name}",
            )
        )
    return tools


def build_tool_call_graph(
    model: Any,
    registry: ToolRegistry,
    tool_names: Iterable[str] | None = None,
):
    """构建最小 LangGraph：模型先判断是否调用工具，工具执行后再回到模型生成回答。"""
    tools = registry_to_langchain_tools(registry, tool_names=tool_names)
    bound_model = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        """调用已绑定工具的模型，并把模型回复追加到消息状态中。"""
        response = bound_model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_stateful_tool_call_graph(
    model: Any,
    registry: ToolRegistry,
    tool_names: Iterable[str] | None = None,
    intent_model: Any | None = None,
):
    """Build a LangGraph tool-calling loop with typed state slots.

    The LLM still selects tools through AIMessage.tool_calls. The runtime owns
    argument binding, validation, tool execution, diagnostics, and blackboard
    updates, so the model does not need brittle output selectors for common
    handoffs such as review_ids and evidence_paths.
    """
    tools = registry_to_langchain_tools(registry, tool_names=tool_names)
    bound_model = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict[str, Any]:
        messages = _messages_with_state_context(state)
        response = bound_model.invoke(messages)
        return {"messages": [response]}

    def analyze_intent(state: AgentState) -> dict[str, Any]:
        if state.get("task_frame"):
            return {}
        if intent_model is None:
            return {"task_frame": {}}
        response = intent_model.invoke(state.get("messages", []))
        content = getattr(response, "content", response)
        return {"task_frame": _parse_task_frame(content)}

    def call_tools(state: AgentState) -> dict[str, Any]:
        runtime_state = _runtime_state_from_graph_state(state)
        messages: list[ToolMessage] = []
        for tool_call in _last_tool_calls(state.get("messages", [])):
            tool_message = _execute_graph_tool_call(registry, runtime_state, tool_call)
            messages.append(tool_message)
        return {
            "messages": messages,
            "blackboard": runtime_state.blackboard,
            "diagnostics": runtime_state.diagnostics,
        }

    graph = StateGraph(AgentState)
    graph.add_node("intent", analyze_intent)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "intent")
    graph.add_edge("intent", "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _parse_task_frame(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(str(content))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _messages_with_state_context(state: AgentState) -> list[BaseMessage]:
    messages = list(state.get("messages", []))
    task_frame = state.get("task_frame") or {}
    blackboard = state.get("blackboard") or {}
    diagnostics = state.get("diagnostics") or []
    if not task_frame and not blackboard and not diagnostics:
        return messages
    context = {
        "task_frame": task_frame,
        "blackboard": blackboard,
        "diagnostics": diagnostics,
        "instruction": (
            "Use tool calls for data operations. Prefer the typed blackboard for "
            "handoffs: review_ids for event evidence lookup, evidence_paths for copying, "
            "export_dirs for the latest created output folder, and top_event for the "
            "top ranked event."
        ),
    }
    return [
        *messages,
        AIMessage(
            content="[state_context]\n"
            + json.dumps(context, ensure_ascii=False, default=_json_default)
            + "\n[/state_context]"
        ),
    ]


def _runtime_state_from_graph_state(state: AgentState) -> PlanExecuteState:
    runtime_state = PlanExecuteState(
        user_request=_first_human_message_content(state.get("messages", [])),
        plan=[],
        blackboard=dict(state.get("blackboard") or {}),
        diagnostics=list(state.get("diagnostics") or []),
    )
    ensure_state_runtime_fields(runtime_state)
    return runtime_state


def _first_human_message_content(messages: list[BaseMessage]) -> str:
    for message in messages:
        if getattr(message, "type", "") == "human":
            return str(getattr(message, "content", ""))
    return ""


def _last_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    if not messages:
        return []
    last_message = messages[-1]
    return list(getattr(last_message, "tool_calls", []) or [])


def _execute_graph_tool_call(
    registry: ToolRegistry,
    runtime_state: PlanExecuteState,
    tool_call: dict[str, Any],
) -> ToolMessage:
    llm_tool_name = str(tool_call.get("name", ""))
    registry_name = llm_tool_name_to_registry_name(llm_tool_name)
    call_id = str(tool_call.get("id") or registry_name)
    step = PlanStep(
        step_id=call_id,
        goal=f"Tool call {registry_name}",
        tool_name=registry_name,
    )
    raw_args = dict(tool_call.get("args") or {})
    try:
        args = resolve_references(raw_args, runtime_state)
        args = bind_tool_arguments(
            registry_name,
            args,
            runtime_state,
        )
        args = validate_tool_arguments(registry_name, args)
        risk = preflight_tool_call(registry, {"tool": registry_name, "args": args})["risk"]
        if risk.get("requires_confirmation"):
            raise ValueError(f"Tool call requires confirmation: {risk.get('reason')}")
        result = execute_registered_tool(
            registry,
            {"tool": registry_name, "args": args},
        )["result"]
    except Exception as error:  # noqa: BLE001
        recovered_args = recover_tool_arguments_from_state(
            registry_name,
            raw_args,
            runtime_state,
            error,
        )
        if recovered_args is None:
            result = validation_error_result("tool_call", error, dict(tool_call.get("args") or {}))
            runtime_state.diagnostics.append(
                {
                    "step_id": call_id,
                    "tool_name": registry_name,
                    "phase": "tool_call",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            try:
                args = validate_tool_arguments(registry_name, recovered_args)
                risk = preflight_tool_call(registry, {"tool": registry_name, "args": args})[
                    "risk"
                ]
                if risk.get("requires_confirmation"):
                    raise ValueError(f"Tool call requires confirmation: {risk.get('reason')}")
                result = execute_registered_tool(
                    registry,
                    {"tool": registry_name, "args": args},
                )["result"]
                runtime_state.diagnostics.append(
                    {
                        "step_id": call_id,
                        "tool_name": registry_name,
                        "phase": "tool_call",
                        "error_type": type(error).__name__,
                        "error": f"Recovered from invalid tool args using typed state: {error}",
                    }
                )
            except Exception as recovered_error:  # noqa: BLE001
                result = validation_error_result(
                    "tool_call",
                    recovered_error,
                    recovered_args,
                )
                runtime_state.diagnostics.append(
                    {
                        "step_id": call_id,
                        "tool_name": registry_name,
                        "phase": "tool_call",
                        "error_type": type(recovered_error).__name__,
                        "error": str(recovered_error),
                    }
                )
    _update_blackboard_from_step(runtime_state, step, result)
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False, default=_json_default),
        tool_call_id=call_id,
        name=llm_tool_name,
    )


def _json_result_wrapper(func: Callable[..., Any]) -> Callable[..., str]:
    """包装工具函数，让工具结果稳定地以 JSON 字符串形式返回给模型。"""
    original_signature = signature(func)

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        """执行原始工具函数，并处理 Path、numpy 标量等 JSON 默认不支持的对象。"""
        return json.dumps(func(*args, **kwargs), ensure_ascii=False, default=_json_default)

    wrapped.__signature__ = original_signature  # type: ignore[attr-defined]
    return wrapped


def _json_default(value: Any) -> Any:
    """为 json.dumps 提供兜底序列化逻辑，避免工具结果因特殊对象而无法返回。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
