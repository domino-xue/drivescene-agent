from __future__ import annotations

import importlib
import inspect
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from drivescene.agent import plan_execute as plan_execute_module
from drivescene.agent.model_config import load_model_config
from drivescene.agent.plan_execute import (
    PendingConfirmation,
    PlanAndExecuteAgent,
    PlanEvent,
    PlanExecuteState,
    PlanStep,
    StepResult,
)
from drivescene.agent.reporting import LLMReporter
from drivescene.memory.context import ContextManager
from drivescene.memory.store import MemoryStore, Thread
from drivescene.memory.writer import AgentMemoryWriter
from scripts.run_agent_cli import build_planner, build_registry_from_paths

try:
    from scripts.run_agent_cli import build_stateful_tool_graph
except ImportError:  # pragma: no cover - compatibility with older CLI module in import tests
    build_stateful_tool_graph = None


AGENT_RUNTIME_VERSION = "state-binding-v2"


EXAMPLE_QUESTIONS = [
    "汇总当前事件索引",
    "找 5 个有效急刹案例并给出动画路径",
    "找 5 个有效 close_following 案例",
    "删除这个文件夹",
]


@dataclass
class ChatWorkspaceRender:
    question: str | None
    live_response: Any | None


def format_step_label(step: PlanStep) -> str:
    tool = step.tool_name or "无工具"
    return f"{step.step_id} [{step.status}] {step.goal} -> {tool}"


def format_pending_confirmation(pending: PendingConfirmation) -> str:
    return (
        f"{pending.step_id} 需要确认：{pending.tool_name}\n\n"
        f"参数：{pending.args}\n\n"
        f"风险：{pending.risk['risk_level']} - {pending.risk['reason']}"
    )


def format_stream_event_for_ui(event: PlanEvent) -> str:
    if event.type == "plan_created":
        return "计划已生成，正在准备执行..."
    if event.type == "step_started" and event.step is not None:
        return f"正在进行：{event.step.step_id} - {event.step.goal}"
    if event.type == "step_completed" and event.result is not None:
        return f"已完成：{event.result.step_id} - {event.result.tool_name}"
    if event.type == "step_completed" and event.step is not None:
        return f"已完成：{event.step.step_id} - {event.step.tool_name or '无工具'}"
    if event.type == "step_failed" and event.result is not None:
        return f"执行失败：{event.result.step_id} - {event.result.error}"
    if event.type == "step_blocked" and event.step is not None:
        return f"已暂停：{event.step.step_id} 被阻塞，等待处理。"
    if event.type == "step_skipped" and event.step is not None:
        return f"已跳过：{event.step.step_id} - 依赖步骤未完成。"
    if event.type == "confirmation_required":
        return "已暂停：需要人工确认后继续执行。"
    if event.type == "final_answer":
        return "已完成：最终回答已生成。"
    return event.message or event.type


def agent_cache_key(
    event_index: str,
    scenario_index: str,
    scenario_root: str,
    model_config: str,
    use_heuristic_planner: bool,
    use_state_graph: bool = False,
    memory_db: str | None = None,
) -> tuple[str, ...]:
    mode = "state_graph" if use_state_graph else ("heuristic" if use_heuristic_planner else "llm")
    key = (
        "agent",
        AGENT_RUNTIME_VERSION,
        str(Path(event_index)),
        str(Path(scenario_index)),
        str(Path(scenario_root)),
        str(Path(model_config)),
        mode,
    )
    if memory_db is not None:
        key = (*key, str(Path(memory_db)))
    return key


def clear_agent_runtime_state(session_state: Any) -> None:
    for key in ["agent", "agent_cache_key", "agent_state", "stream_status"]:
        session_state.pop(key, None)


def runtime_status_text(use_state_graph: bool) -> str:
    mode = "state_graph" if use_state_graph else "plan_execute"
    return f"Runtime: {AGENT_RUNTIME_VERSION} ({mode})"


def thread_title_from_question(question: str) -> str:
    title = " ".join(question.strip().split())
    if not title:
        return "新对话"
    return title[:40]


def chat_workspace_css() -> str:
    return """
<style>
    .block-container {
        padding-top: 4rem;
        padding-bottom: 1rem;
        max-width: 1120px;
    }

    .ds-chat-workspace {
        max-width: 920px;
        margin: 0 auto;
    }

    .ds-chat-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }

    .ds-chat-title {
        color: #1f2937;
        font-size: 1.35rem;
        font-weight: 650;
        line-height: 1.25;
    }

    .ds-chat-kicker {
        color: #64748b;
        font-size: 0.82rem;
        white-space: nowrap;
    }

    .ds-chat-history {
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #f7f8fa;
        padding: 0.5rem;
    }

    .ds-chat-history-marker {
        display: none;
    }

    div[data-testid="stVerticalBlock"]:has(.ds-chat-history-marker) {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #f7f8fa;
        padding: 0.55rem 0.45rem;
    }

    .ds-chat-empty {
        border: 1px dashed #d1d5db;
        border-radius: 8px;
        color: #64748b;
        background: #f8fafc;
        padding: 1.2rem;
        text-align: center;
        margin: 0.4rem 0 0.7rem;
    }

    .ds-message-row {
        display: flex;
        width: 100%;
        margin: 0.35rem 0;
    }

    .ds-message-user {
        justify-content: flex-end;
    }

    .ds-message-assistant {
        justify-content: flex-start;
    }

    .ds-message-bubble {
        max-width: 76%;
        border-radius: 8px;
        padding: 0.62rem 0.78rem;
        line-height: 1.55;
        word-break: break-word;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }

    .ds-message-user .ds-message-bubble {
        background: #d9fdd3;
        color: #172554;
    }

    .ds-message-assistant .ds-message-bubble {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        color: #111827;
    }

    div[data-testid="stForm"]:has(.ds-chat-form-marker) {
        border: 0;
        padding: 0.45rem 0 0;
        background: transparent;
    }

    .ds-chat-form-marker {
        display: none;
    }

    .ds-execution-shell {
        max-width: 920px;
        margin: 0.4rem auto 0;
    }
</style>
"""


def render_app_shell_styles() -> None:
    st.markdown(chat_workspace_css(), unsafe_allow_html=True)


def _chat_history_height(message_count: int) -> int:
    if message_count <= 0:
        return 0
    return min(620, max(220, message_count * 120 + 100))


def _message_bubble_html(role: str, content: str) -> str:
    message_class = "ds-message-user" if role == "user" else "ds-message-assistant"
    safe_content = html.escape(content).replace("\n", "<br>")
    return (
        f'<div class="ds-message-row {message_class}">'
        f'<div class="ds-message-bubble">{safe_content}</div>'
        "</div>"
    )


def _render_live_assistant_response(placeholder: Any | None, content: str) -> None:
    if placeholder is None or not content:
        return
    placeholder.markdown(_message_bubble_html("assistant", content), unsafe_allow_html=True)


def _tool_result_brief(result: StepResult) -> str:
    payload = result.result
    if not isinstance(payload, dict):
        return ""
    operation = payload.get("operation") or result.tool_name or "工具调用"
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return f"{operation}：{summary.strip()}"
    if not isinstance(summary, dict):
        return str(operation)

    keys = (
        "total",
        "succeeded",
        "failed",
        "num_events",
        "num_valid",
        "num_copied",
        "num_deleted",
        "num_processed",
    )
    details = [f"{key}={summary[key]}" for key in keys if key in summary]
    return f"{operation}：{', '.join(details)}" if details else str(operation)


def _partial_execution_summary(state: PlanExecuteState) -> str:
    completed = [result for result in state.step_results if not result.error]
    failed = [result for result in state.step_results if result.error]
    if not completed and not failed:
        return ""

    parts = [f"已完成 {len(completed)} 个步骤"]
    if failed:
        parts.append(f"失败 {len(failed)} 个步骤")
    latest = completed[-1] if completed else failed[-1]
    latest_brief = _tool_result_brief(latest)
    if latest_brief:
        parts.append(f"最近结果：{latest_brief}")
    return "。".join(parts) + "。"


def _confirmation_assistant_message(state: PlanExecuteState) -> str:
    parts = [_partial_execution_summary(state)]
    if state.final_answer.strip():
        parts.append(state.final_answer.strip())
    else:
        parts.append("此操作需要人工确认后才能继续执行。")
    parts.append("请在“执行状态与细节”中选择确认执行或取消。")
    return "\n\n".join(part for part in parts if part)


def _stream_event_message(event: PlanEvent) -> str:
    if event.type == "plan_created":
        return "正在理解任务，已生成执行计划。"
    if event.type == "step_started" and event.step is not None:
        return f"正在执行：{event.step.goal}"
    if event.type == "step_completed" and event.step is not None:
        details = _tool_result_brief(event.result) if event.result is not None else ""
        return f"已完成：{event.step.goal}" + (f"\n\n结果：{details}" if details else "")
    if event.type == "step_failed" and event.step is not None:
        error = event.result.error if event.result is not None else event.message
        return f"步骤失败：{event.step.goal}\n\n原因：{error or '未知错误'}"
    if event.type == "confirmation_required":
        return _confirmation_assistant_message(event.state)
    if event.type == "final_answer":
        return event.state.final_answer or event.message or "任务已完成。"
    return format_stream_event_for_ui(event)


def _assistant_message_for_state(state: PlanExecuteState) -> str:
    if state.pending_confirmation is not None:
        return _confirmation_assistant_message(state)
    return state.final_answer.strip() or _partial_execution_summary(state)


def _build_reporter_for_app(
    config_path: Path | str = "config/model.yml",
    model_factory=ChatOpenAI,
    use_heuristic_planner: bool = False,
) -> LLMReporter:
    if use_heuristic_planner:
        return LLMReporter()
    model_config = load_model_config(config_path)
    return LLMReporter(model=model_factory(**model_config.to_chat_openai_kwargs()))


def _create_plan_execute_agent_for_app(
    registry: Any,
    planner: Any,
    reporter: LLMReporter,
    memory_writer: Any,
) -> Any:
    agent_cls = plan_execute_module.PlanAndExecuteAgent
    parameters = inspect.signature(agent_cls.__init__).parameters
    if "reporter" not in parameters:
        agent_cls = importlib.reload(plan_execute_module).PlanAndExecuteAgent
        parameters = inspect.signature(agent_cls.__init__).parameters

    kwargs = {
        "registry": registry,
        "planner": planner,
        "memory_writer": memory_writer,
    }
    if "reporter" in parameters:
        kwargs["reporter"] = reporter
    if "max_replans" in parameters:
        kwargs["max_replans"] = 1

    agent = agent_cls(**kwargs)
    if "reporter" not in parameters:
        setattr(agent, "reporter", reporter)
    return agent


def main() -> None:
    st.set_page_config(page_title="DriveScene Agent", layout="wide")
    render_app_shell_styles()

    with st.sidebar:
        st.subheader("配置")
        event_index = st.text_input("Event index", "outputs/labeled_event_index.parquet")
        scenario_index = st.text_input("Scenario index", "outputs/scenario_index.parquet")
        scenario_root = st.text_input("Scenario root", "data/val")
        model_config = st.text_input("Model config", "config/model.yml")
        memory_db = st.text_input("Memory DB", "outputs/agent_memory.sqlite")
        use_state_graph = st.checkbox("使用 state graph tool-calling runtime", value=False)

        st.caption(runtime_status_text(use_state_graph))
        if st.button("重置 Agent Runtime", use_container_width=True):
            clear_agent_runtime_state(st.session_state)
            st.rerun()

    memory_store = _get_memory_store(memory_db)
    user = _render_auth(memory_store)
    if user is None:
        st.info("请先登录或注册，然后开始多轮对话。")
        return

    thread = _render_thread_sidebar(memory_store, user["id"])
    if thread is None:
        st.info("请创建或选择一个 thread。")
        return

    agent = _get_agent(
        event_index=event_index,
        scenario_index=scenario_index,
        scenario_root=scenario_root,
        model_config=model_config,
        use_heuristic_planner=False,
        memory_db=memory_db,
        use_state_graph=use_state_graph,
    )
    state: Any | None = st.session_state.get("agent_state")

    queued_turn = _queued_turn_for_thread(thread.id)
    workspace = _render_chat_workspace(
        memory_store,
        thread,
        reserve_live_response=queued_turn is not None,
    )
    if workspace.question:
        _queue_user_turn(memory_store, user_id=user["id"], thread_id=thread.id, question=workspace.question)
        st.rerun()

    if queued_turn is not None:
        st.session_state.pop("pending_agent_turn", None)
        state = _run_user_turn(
            memory_store,
            agent,
            user_id=queued_turn["user_id"],
            thread_id=queued_turn["thread_id"],
            question=queued_turn["question"],
            append_user_message=False,
            live_response=workspace.live_response,
        )

    with st.expander(
        "执行状态与细节",
        expanded=_state_requires_attention(state),
    ):
        _render_agent_state_panel(memory_store, agent, state, show_details_expander=False)


def _queued_turn_for_thread(thread_id: int) -> dict[str, Any] | None:
    pending = st.session_state.get("pending_agent_turn")
    if not isinstance(pending, dict):
        return None
    if pending.get("thread_id") != thread_id:
        return None
    if not isinstance(pending.get("question"), str) or not pending["question"].strip():
        return None
    if not isinstance(pending.get("user_id"), int):
        return None
    return pending


def _render_chat_workspace(
    memory_store: MemoryStore,
    thread: Thread,
    *,
    reserve_live_response: bool = False,
) -> ChatWorkspaceRender:
    messages = memory_store.list_messages(thread.id)
    live_response: Any | None = None
    st.markdown(
        (
            '<div class="ds-chat-workspace">'
            '<div class="ds-chat-header">'
            '<div class="ds-chat-title">DriveScene Agent</div>'
            '<div class="ds-chat-kicker">场景分析助手</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(st.session_state.get("stream_status", "准备就绪"))

    if messages:
        with st.container(
            height=_chat_history_height(len(messages)),
            border=False,
            key=f"chat_history_{thread.id}",
            autoscroll=True,
        ):
            st.markdown('<span class="ds-chat-history-marker"></span>', unsafe_allow_html=True)
            for message in messages:
                role = "assistant" if message.role == "assistant" else "user"
                st.markdown(
                    _message_bubble_html(role, message.content),
                    unsafe_allow_html=True,
                )
            if reserve_live_response:
                live_response = st.empty()
    else:
        st.markdown(
            '<div class="ds-chat-empty">可以直接输入场景分析任务。</div>',
            unsafe_allow_html=True,
        )

    if "pending_question" in st.session_state:
        return ChatWorkspaceRender(
            question=st.session_state.pop("pending_question"),
            live_response=live_response,
        )

    with st.form(
        key=f"chat_form_{thread.id}",
        clear_on_submit=True,
        border=False,
    ):
        st.markdown('<span class="ds-chat-form-marker"></span>', unsafe_allow_html=True)
        question = st.text_input(
            "输入场景分析任务",
            key=f"chat_text_{thread.id}",
            label_visibility="collapsed",
            placeholder="输入场景分析任务，按 Enter 或点击发送",
        )
        submitted = st.form_submit_button("发送", type="primary", use_container_width=True)
    return ChatWorkspaceRender(
        question=question.strip() if submitted and question.strip() else None,
        live_response=live_response,
    )


def _render_auth(memory_store: MemoryStore) -> dict[str, Any] | None:
    current = st.session_state.get("current_user")
    if current is not None:
        with st.sidebar:
            st.success(f"已登录：{current['username']}")
            if st.button("退出登录", use_container_width=True):
                for key in ["current_user", "current_thread_id", "agent_state"]:
                    st.session_state.pop(key, None)
                st.rerun()
        return current

    with st.sidebar:
        st.subheader("登录")
        tab_login, tab_register = st.tabs(["登录", "注册"])
        with tab_login:
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            if st.button("登录", use_container_width=True):
                user = memory_store.authenticate_user(username, password)
                if user is None:
                    st.error("用户名或密码错误。")
                else:
                    st.session_state["current_user"] = {"id": user.id, "username": user.username}
                    st.rerun()
        with tab_register:
            username = st.text_input("新用户名", key="register_username")
            password = st.text_input("新密码", type="password", key="register_password")
            if st.button("注册", use_container_width=True):
                try:
                    user = memory_store.create_user(username, password)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state["current_user"] = {"id": user.id, "username": user.username}
                    st.rerun()
    return None


def _render_thread_sidebar(memory_store: MemoryStore, user_id: int) -> Thread | None:
    with st.sidebar:
        st.subheader("Threads")
        if st.button("新建对话", use_container_width=True):
            thread = memory_store.create_thread(user_id, "新对话")
            st.session_state["current_thread_id"] = thread.id
            st.session_state.pop("agent_state", None)
            st.rerun()

        threads = memory_store.list_threads(user_id)
        if not threads:
            thread = memory_store.create_thread(user_id, "新对话")
            st.session_state["current_thread_id"] = thread.id
            return thread

        current_thread_id = st.session_state.get("current_thread_id") or threads[0].id
        selected: Thread | None = None
        for thread in threads:
            label = thread.title or f"Thread {thread.id}"
            if st.button(label, key=f"thread_{thread.id}", use_container_width=True):
                current_thread_id = thread.id
                st.session_state["current_thread_id"] = thread.id
                st.session_state.pop("agent_state", None)
                st.rerun()
            if thread.id == current_thread_id:
                selected = thread

        st.subheader("示例问题")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

    return selected or threads[0]


def _record_user_message(
    memory_store: MemoryStore,
    user_id: int,
    thread_id: int,
    question: str,
) -> None:
    thread = memory_store.get_thread(thread_id)
    if thread is not None and thread.title == "新对话":
        memory_store.update_thread_title(thread_id, thread_title_from_question(question))
    memory_store.append_message(thread_id, "user", question)


def _queue_user_turn(
    memory_store: MemoryStore,
    *,
    user_id: int,
    thread_id: int,
    question: str,
) -> None:
    _record_user_message(memory_store, user_id, thread_id, question)
    st.session_state["pending_agent_turn"] = {
        "user_id": user_id,
        "thread_id": thread_id,
        "question": question,
    }
    st.session_state["stream_status"] = "正在理解任务，准备执行。"


def _run_user_turn(
    memory_store: MemoryStore,
    agent: Any,
    user_id: int,
    thread_id: int,
    question: str,
    *,
    append_user_message: bool = True,
    live_response: Any | None = None,
) -> Any:
    if append_user_message:
        _record_user_message(memory_store, user_id, thread_id, question)
    _render_live_assistant_response(live_response, "正在理解任务，准备执行。")

    if _is_state_graph_agent(agent):
        previous_state = st.session_state.get("agent_state")
        previous_blackboard = (
            previous_state.get("blackboard", {}) if isinstance(previous_state, dict) else {}
        )
        previous_diagnostics = (
            previous_state.get("diagnostics", []) if isinstance(previous_state, dict) else []
        )
        state = agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "blackboard": previous_blackboard,
                "diagnostics": previous_diagnostics,
            }
        )
        final_answer = _graph_final_message(state)
        st.session_state["agent_state"] = state
        st.session_state["stream_status"] = "已完成：state graph 最终回答已生成。"
        if final_answer:
            memory_store.append_message(
                thread_id,
                "assistant",
                final_answer,
                metadata={"status": st.session_state["stream_status"], "runtime": "state_graph"},
            )
            _render_live_assistant_response(live_response, final_answer)
        return state

    context_pack = ContextManager(memory_store).build_context_pack(user_id, thread_id, question)
    stream_status = ""
    state: PlanExecuteState | None = None
    last_live_message = ""
    for event in agent.stream(
        question,
        context_pack=context_pack,
        memory_user_id=user_id,
        memory_thread_id=thread_id,
    ):
        state = event.state
        stream_status = format_stream_event_for_ui(event)
        live_message = _stream_event_message(event)
        if live_message and live_message != last_live_message:
            _render_live_assistant_response(live_response, live_message)
            last_live_message = live_message
        if event.type == "confirmation_required":
            break
    st.session_state["agent_state"] = state
    st.session_state["stream_status"] = stream_status
    if state is not None:
        assistant_message = _assistant_message_for_state(state)
        if state.pending_confirmation is not None:
            state.final_answer = assistant_message
        if assistant_message and assistant_message != last_live_message:
            _render_live_assistant_response(live_response, assistant_message)
        if assistant_message:
            memory_store.append_message(
                thread_id,
                "assistant",
                assistant_message,
                metadata={
                    "status": stream_status,
                    "pending_confirmation": state.pending_confirmation is not None,
                },
            )
    return state


def _render_agent_state_panel(
    memory_store: MemoryStore,
    agent: Any,
    state: Any | None,
    *,
    show_details_expander: bool = True,
) -> None:
    st.subheader("执行结果")
    if state is None:
        st.caption("运行一次问题后，这里会优先展示最终回答；plan 和工具结果会保留在执行细节中。")
        return
    if isinstance(state, dict):
        _render_graph_state_panel(state, show_details_expander=show_details_expander)
        return

    if state.final_answer:
        st.markdown(state.final_answer)
    elif state.pending_confirmation is not None:
        st.info(_confirmation_assistant_message(state))
    else:
        st.caption(st.session_state.get("stream_status", "任务正在执行。"))

    if state.pending_confirmation is not None:
        st.subheader("Human Confirmation")
        st.warning(format_pending_confirmation(state.pending_confirmation))
        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button("确认执行", type="primary", use_container_width=True):
                live_response = st.empty()
                _continue_after_confirmation(
                    memory_store,
                    agent,
                    state,
                    approved=True,
                    live_response=live_response,
                )
                st.rerun()
        with reject_col:
            if st.button("取消执行", use_container_width=True):
                live_response = st.empty()
                _continue_after_confirmation(
                    memory_store,
                    agent,
                    state,
                    approved=False,
                    live_response=live_response,
                )
                st.rerun()

    if show_details_expander:
        with st.expander("执行细节", expanded=False):
            _render_plan_execute_details(state)
    else:
        _render_plan_execute_details(state)


def _render_plan_execute_details(state: PlanExecuteState) -> None:
    st.markdown("**Plan**")
    for step in state.plan:
        st.write(format_step_label(step))

    st.markdown("**Tool Results**")
    for result in state.step_results:
        st.markdown(f"**{result.step_id}**")
        if result.error:
            st.error(result.error)
        else:
            st.json(result.result)

    if state.execution_digest:
        st.markdown("**Execution Digest**")
        st.json(state.execution_digest)
    if state.evaluation:
        st.markdown("**Evaluation**")
        st.json(state.evaluation)
    blackboard = state_runtime_field(state, "blackboard", {})
    diagnostics = state_runtime_field(state, "diagnostics", [])
    if blackboard:
        st.markdown("**Typed State Blackboard**")
        st.json(blackboard)
    if diagnostics:
        st.markdown("**Diagnostics**")
        st.json(diagnostics)


def _continue_after_confirmation(
    memory_store: MemoryStore,
    agent: PlanAndExecuteAgent,
    state: PlanExecuteState,
    approved: bool,
    *,
    live_response: Any | None = None,
) -> PlanExecuteState:
    state = agent.resolve_confirmation(state, approved=approved)
    decision = "已收到确认，正在继续执行。" if approved else "已取消高风险操作，正在整理结果。"
    _render_live_assistant_response(live_response, decision)
    stream_status = ""
    last_live_message = decision
    for event in agent.stream_from_state(state):
        state = event.state
        stream_status = format_stream_event_for_ui(event)
        live_message = _stream_event_message(event)
        if live_message and live_message != last_live_message:
            _render_live_assistant_response(live_response, live_message)
            last_live_message = live_message
        if event.type == "confirmation_required":
            break
    st.session_state["agent_state"] = state
    st.session_state["stream_status"] = stream_status
    assistant_message = _assistant_message_for_state(state)
    if state.pending_confirmation is not None:
        state.final_answer = assistant_message
    if assistant_message and assistant_message != last_live_message:
        _render_live_assistant_response(live_response, assistant_message)
    if state.memory_thread_id is not None and assistant_message:
        memory_store.append_message(
            state.memory_thread_id,
            "assistant",
            assistant_message,
            metadata={
                "status": stream_status,
                "confirmed": approved,
                "pending_confirmation": state.pending_confirmation is not None,
            },
        )
    return state


@st.cache_resource
def _get_memory_store(memory_db: str) -> MemoryStore:
    return MemoryStore(Path(memory_db))


def _get_agent(
    event_index: str,
    scenario_index: str,
    scenario_root: str,
    model_config: str,
    use_heuristic_planner: bool,
    *,
    memory_db: str,
    use_state_graph: bool = False,
) -> Any:
    key = agent_cache_key(
        event_index=event_index,
        scenario_index=scenario_index,
        scenario_root=scenario_root,
        model_config=model_config,
        use_heuristic_planner=use_heuristic_planner,
        use_state_graph=use_state_graph,
        memory_db=memory_db,
    )
    if st.session_state.get("agent_cache_key") != key:
        clear_agent_runtime_state(st.session_state)
        registry = build_registry_from_paths(event_index, scenario_index, scenario_root)
        if use_state_graph:
            if build_stateful_tool_graph is None:
                raise RuntimeError("State graph runtime is unavailable in this CLI module.")
            st.session_state["agent"] = build_stateful_tool_graph(
                registry=registry,
                config_path=model_config,
            )
            st.session_state["agent_cache_key"] = key
            return st.session_state["agent"]
        planner = build_planner(
            registry=registry,
            config_path=model_config,
            use_heuristic_planner=use_heuristic_planner,
        )
        reporter = _build_reporter_for_app(
            config_path=model_config,
            use_heuristic_planner=use_heuristic_planner,
        )
        memory_writer = AgentMemoryWriter(_get_memory_store(memory_db))
        st.session_state["agent"] = _create_plan_execute_agent_for_app(
            registry=registry,
            planner=planner,
            reporter=reporter,
            memory_writer=memory_writer,
        )
        st.session_state["agent_cache_key"] = key
    return st.session_state["agent"]


def state_runtime_field(state: Any, field_name: str, default: Any) -> Any:
    return getattr(state, field_name, default)


def _state_requires_attention(state: Any | None) -> bool:
    return isinstance(state, PlanExecuteState) and state.pending_confirmation is not None


def _is_state_graph_agent(agent: Any) -> bool:
    return hasattr(agent, "invoke") and not isinstance(agent, PlanAndExecuteAgent)


def _graph_final_message(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", messages[-1])).strip()


def _render_graph_state_panel(
    state: dict[str, Any],
    *,
    show_details_expander: bool = True,
) -> None:
    final_answer = _graph_final_message(state)
    if final_answer:
        st.markdown(final_answer)
    else:
        st.caption(st.session_state.get("stream_status", "任务正在执行。"))

    if show_details_expander:
        with st.expander("执行细节", expanded=False):
            _render_graph_details(state)
    else:
        _render_graph_details(state)


def _render_graph_details(state: dict[str, Any]) -> None:
    if state.get("task_frame"):
        st.markdown("**Task Frame**")
        st.json(state["task_frame"])
    if state.get("blackboard"):
        st.markdown("**Typed State Blackboard**")
        st.json(state["blackboard"])
    if state.get("diagnostics"):
        st.markdown("**Diagnostics**")
        st.json(state["diagnostics"])
    if state.get("messages"):
        st.markdown("**Messages**")
        for index, message in enumerate(state["messages"], start=1):
            st.markdown(f"**{index}. {getattr(message, 'type', type(message).__name__)}**")
            st.write(getattr(message, "content", message))


if __name__ == "__main__":
    main()
