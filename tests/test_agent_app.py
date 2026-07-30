import inspect
from pathlib import Path

from drivescene.agent.plan_execute import (
    PendingConfirmation,
    PlanEvent,
    PlanExecuteState,
    PlanStep,
    StepResult,
)
from scripts.agent_app import (
    AGENT_RUNTIME_VERSION,
    _get_agent,
    _chat_history_height,
    _confirmation_assistant_message,
    _message_bubble_html,
    _queue_user_turn,
    _render_chat_workspace,
    _run_user_turn,
    _stream_event_message,
    clear_agent_runtime_state,
    agent_cache_key,
    chat_workspace_css,
    format_pending_confirmation,
    format_step_label,
    format_stream_event_for_ui,
    render_app_shell_styles,
    runtime_status_text,
    state_runtime_field,
)


def test_format_step_label_includes_status_goal_and_tool() -> None:
    label = format_step_label(
        PlanStep(
            step_id="step_1",
            goal="汇总事件索引",
            tool_name="EvidenceStore.query_index",
            status="completed",
        )
    )

    assert "step_1" in label
    assert "completed" in label
    assert "汇总事件索引" in label
    assert "EvidenceStore.query_index" in label


def test_format_pending_confirmation_shows_risk_reason() -> None:
    text = format_pending_confirmation(
        PendingConfirmation(
            step_id="step_2",
            tool_name="ArtifactOps.manage_artifacts",
            args={"operation": "delete", "paths": ["outputs/old"]},
            risk={"risk_level": "high", "reason": "delete operation", "requires_confirmation": True},
        )
    )

    assert "step_2" in text
    assert "ArtifactOps.manage_artifacts" in text
    assert "delete operation" in text


def test_agent_cache_key_includes_model_config_and_planner_mode() -> None:
    key = agent_cache_key(
        event_index="outputs/labeled_event_index.parquet",
        scenario_index="outputs/scenario_index.parquet",
        scenario_root="data/val",
        model_config="config/model.yml",
        use_heuristic_planner=False,
    )

    assert any(part.endswith("model.yml") for part in key)
    assert "llm" in key
    assert AGENT_RUNTIME_VERSION in key


def test_ui_does_not_expose_heuristic_fallback() -> None:
    source = Path("scripts/agent_app.py").read_text(encoding="utf-8")

    assert 'st.checkbox("使用规则 fallback planner"' not in source
    assert "use_heuristic_planner=False," in source


def test_clear_agent_runtime_state_removes_cached_agent_and_old_outputs() -> None:
    session_state = {
        "agent": object(),
        "agent_cache_key": ("old",),
        "agent_state": object(),
        "stream_status": "old result",
        "current_user": {"id": 1},
        "current_thread_id": 2,
    }

    clear_agent_runtime_state(session_state)

    assert "agent" not in session_state
    assert "agent_cache_key" not in session_state
    assert "agent_state" not in session_state
    assert "stream_status" not in session_state
    assert session_state["current_user"] == {"id": 1}
    assert session_state["current_thread_id"] == 2


def test_runtime_status_text_shows_version_and_mode() -> None:
    assert AGENT_RUNTIME_VERSION in runtime_status_text(use_state_graph=False)
    assert "plan_execute" in runtime_status_text(use_state_graph=False)
    assert "state_graph" in runtime_status_text(use_state_graph=True)


def test_agent_factory_runtime_flags_are_keyword_only() -> None:
    signature = inspect.signature(_get_agent)

    assert signature.parameters["memory_db"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["use_state_graph"].kind is inspect.Parameter.KEYWORD_ONLY


def test_format_stream_event_for_ui_returns_readable_text() -> None:
    state = PlanExecuteState(
        user_request="汇总",
        plan=[PlanStep(step_id="step_1", goal="汇总", tool_name="EvidenceStore.query_index")],
    )
    event = PlanEvent(type="step_started", state=state, step=state.plan[0], message="汇总")

    text = format_stream_event_for_ui(event)

    assert "step_1" in text
    assert "正在进行" in text


def test_format_stream_event_for_ui_marks_completed() -> None:
    state = PlanExecuteState(
        user_request="汇总",
        plan=[PlanStep(step_id="step_1", goal="汇总", tool_name="EvidenceStore.query_index")],
    )
    event = PlanEvent(type="step_completed", state=state, step=state.plan[0], message="汇总")

    text = format_stream_event_for_ui(event)

    assert "已完成" in text


def test_state_runtime_field_returns_default_for_old_cached_state() -> None:
    class OldState:
        pass

    assert state_runtime_field(OldState(), "blackboard", {}) == {}


def test_chat_workspace_css_defines_scrollable_chat_area() -> None:
    css = chat_workspace_css()

    assert ".ds-chat-workspace" in css
    assert ".ds-chat-history" in css
    assert ".ds-message-user" in css
    assert ".ds-message-assistant" in css
    assert "overflow-y: auto" in css
    assert "padding-top: 4rem" in css
    assert "div[data-testid=\"stChatInput\"]" not in css


def test_chat_history_height_scales_without_empty_box() -> None:
    assert _chat_history_height(0) == 0
    assert _chat_history_height(1) == 220
    assert _chat_history_height(2) == 340
    assert _chat_history_height(10) == 620


def test_message_bubble_html_aligns_roles_and_escapes_content() -> None:
    user_html = _message_bubble_html("user", "<delete>\n急刹")
    assistant_html = _message_bubble_html("assistant", "已找到")

    assert "ds-message-user" in user_html
    assert "ds-message-assistant" in assistant_html
    assert "&lt;delete&gt;<br>急刹" in user_html
    assert "<delete>" not in user_html


def test_render_app_shell_styles_injects_css(monkeypatch) -> None:
    calls = []

    class FakeStreamlit:
        def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
            calls.append((body, unsafe_allow_html))

    monkeypatch.setattr("scripts.agent_app.st", FakeStreamlit())

    render_app_shell_styles()

    assert calls
    assert calls[0][1] is True
    assert ".ds-chat-workspace" in calls[0][0]


def test_render_chat_workspace_uses_bubbles_and_inline_form(monkeypatch) -> None:
    class FakeMessage:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    class FakeMemoryStore:
        def list_messages(self, thread_id: int):
            assert thread_id == 42
            return [
                FakeMessage("user", "给我一个急刹事件"),
                FakeMessage("assistant", "已找到 1 个急刹事件。"),
            ]

    class FakeThread:
        id = 42
        title = "给我一个急刹事件"

    class FakeContainer:
        def __init__(self, parent: "FakeStreamlit") -> None:
            self.parent = parent

        def __enter__(self) -> "FakeStreamlit":
            return self.parent

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeForm(FakeContainer):
        pass

    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {"stream_status": "已完成"}
            self.calls = []

        def markdown(self, body: str, **kwargs) -> None:
            self.calls.append(("markdown", body, kwargs))

        def caption(self, body: str) -> None:
            self.calls.append(("caption", body))

        def container(self, **kwargs) -> FakeContainer:
            self.calls.append(("container", kwargs))
            return FakeContainer(self)

        def form(self, key: str, **kwargs) -> FakeForm:
            self.calls.append(("form", key, kwargs))
            return FakeForm(self)

        def text_input(self, label: str, **kwargs) -> str:
            self.calls.append(("text_input", label, kwargs))
            return "继续查 close_following"

        def form_submit_button(self, label: str, **kwargs) -> bool:
            self.calls.append(("form_submit_button", label, kwargs))
            return True

    fake_st = FakeStreamlit()
    monkeypatch.setattr("scripts.agent_app.st", fake_st)

    workspace = _render_chat_workspace(FakeMemoryStore(), FakeThread())

    assert workspace.question == "继续查 close_following"
    assert ("caption", "已完成") in fake_st.calls
    assert not any(call[0] == "chat_input" for call in fake_st.calls)
    header_calls = [
        call for call in fake_st.calls if call[0] == "markdown" and "ds-chat-header" in call[1]
    ]
    assert header_calls
    assert "DriveScene Agent" in header_calls[0][1]
    assert "给我一个急刹事件" not in header_calls[0][1]
    assert any(
        call[0] == "markdown" and "ds-message-user" in call[1]
        for call in fake_st.calls
    )
    assert any(
        call[0] == "markdown" and "ds-message-assistant" in call[1]
        for call in fake_st.calls
    )
    container_calls = [call for call in fake_st.calls if call[0] == "container"]
    assert container_calls[0][1]["height"] == 340
    assert container_calls[0][1]["autoscroll"] is True


def test_render_chat_workspace_uses_compact_empty_state(monkeypatch) -> None:
    class FakeMemoryStore:
        def list_messages(self, thread_id: int):
            return []

    class FakeThread:
        id = 7
        title = "新对话"

    class FakeForm:
        def __init__(self, parent: "FakeStreamlit") -> None:
            self.parent = parent

        def __enter__(self) -> "FakeStreamlit":
            return self.parent

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {}
            self.calls = []

        def markdown(self, body: str, **kwargs) -> None:
            self.calls.append(("markdown", body, kwargs))

        def caption(self, body: str) -> None:
            self.calls.append(("caption", body))

        def container(self, **kwargs):
            self.calls.append(("container", kwargs))
            raise AssertionError("empty chats should not render a large history container")

        def form(self, key: str, **kwargs) -> FakeForm:
            self.calls.append(("form", key, kwargs))
            return FakeForm(self)

        def text_input(self, label: str, **kwargs) -> str:
            self.calls.append(("text_input", label, kwargs))
            return ""

        def form_submit_button(self, label: str, **kwargs) -> bool:
            self.calls.append(("form_submit_button", label, kwargs))
            return False

    fake_st = FakeStreamlit()
    monkeypatch.setattr("scripts.agent_app.st", fake_st)

    workspace = _render_chat_workspace(FakeMemoryStore(), FakeThread())

    assert workspace.question is None
    assert any(
        call[0] == "markdown" and "ds-chat-empty" in call[1]
        for call in fake_st.calls
    )


def test_queue_user_turn_persists_message_before_execution(monkeypatch) -> None:
    class FakeThread:
        title = "新对话"

    class FakeMemoryStore:
        def __init__(self) -> None:
            self.updated_titles = []
            self.messages = []

        def get_thread(self, thread_id: int):
            assert thread_id == 12
            return FakeThread()

        def update_thread_title(self, thread_id: int, title: str) -> None:
            self.updated_titles.append((thread_id, title))

        def append_message(self, thread_id: int, role: str, content: str, metadata=None) -> None:
            self.messages.append((thread_id, role, content, metadata))

    class FakeStreamlit:
        session_state = {}

    fake_store = FakeMemoryStore()
    monkeypatch.setattr("scripts.agent_app.st", FakeStreamlit())

    _queue_user_turn(fake_store, user_id=3, thread_id=12, question="查找急刹事件")

    assert fake_store.messages == [(12, "user", "查找急刹事件", None)]
    assert fake_store.updated_titles == [(12, "查找急刹事件")]
    assert FakeStreamlit.session_state["pending_agent_turn"] == {
        "user_id": 3,
        "thread_id": 12,
        "question": "查找急刹事件",
    }


def test_stream_event_message_includes_completed_tool_result() -> None:
    step = PlanStep(
        step_id="step_1",
        goal="检索急刹事件",
        tool_name="EvidenceStore.query_index",
    )
    state = PlanExecuteState(user_request="检索急刹", plan=[step])
    result = StepResult(
        step_id="step_1",
        tool_name="EvidenceStore.query_index",
        args={},
        result={
            "operation": "search_events",
            "summary": {"total": 5, "succeeded": 5, "failed": 0},
        },
    )

    message = _stream_event_message(
        PlanEvent(type="step_completed", state=state, step=step, result=result)
    )

    assert "已完成" in message
    assert "检索急刹事件" in message
    assert "total=5" in message


def test_confirmation_message_keeps_partial_results_visible() -> None:
    step = PlanStep(
        step_id="step_1",
        goal="删除导出文件夹",
        tool_name="ArtifactOps.manage_artifacts",
    )
    state = PlanExecuteState(
        user_request="删除文件夹",
        plan=[step],
        pending_confirmation=PendingConfirmation(
            step_id="step_1",
            tool_name="ArtifactOps.manage_artifacts",
            args={"operation": "delete", "paths": ["outputs/demo"]},
            risk={"risk_level": "high", "reason": "delete operation"},
        ),
        step_results=[
            StepResult(
                step_id="step_0",
                tool_name="EvidenceStore.query_index",
                args={},
                result={
                    "operation": "search_events",
                    "summary": {"total": 5, "succeeded": 5, "failed": 0},
                },
            )
        ],
        final_answer="此操作需要人工确认后才能继续执行。",
    )

    message = _confirmation_assistant_message(state)

    assert "已完成 1 个步骤" in message
    assert "total=5" in message
    assert "需要人工确认" in message


def test_run_user_turn_streams_progress_and_persists_final_answer(monkeypatch) -> None:
    step = PlanStep(
        step_id="step_1",
        goal="检索急刹事件",
        tool_name="EvidenceStore.query_index",
    )
    state = PlanExecuteState(user_request="检索急刹", plan=[step], final_answer="已找到 5 个急刹事件。")

    class FakeAgent:
        def stream(self, *args, **kwargs):
            yield PlanEvent(type="plan_created", state=state)
            yield PlanEvent(type="step_started", state=state, step=step)
            yield PlanEvent(type="final_answer", state=state, message=state.final_answer)

    class FakeContextManager:
        def __init__(self, store) -> None:
            self.store = store

        def build_context_pack(self, *args, **kwargs):
            return {"context": "ok"}

    class FakeMemoryStore:
        def __init__(self) -> None:
            self.messages = []

        def append_message(self, thread_id: int, role: str, content: str, metadata=None) -> None:
            self.messages.append((thread_id, role, content, metadata))

    class FakePlaceholder:
        def __init__(self) -> None:
            self.messages = []

        def markdown(self, body: str, **kwargs) -> None:
            self.messages.append(body)

    class FakeStreamlit:
        session_state = {}

    memory_store = FakeMemoryStore()
    placeholder = FakePlaceholder()
    monkeypatch.setattr("scripts.agent_app.ContextManager", FakeContextManager)
    monkeypatch.setattr("scripts.agent_app.st", FakeStreamlit())

    returned = _run_user_turn(
        memory_store,
        FakeAgent(),
        user_id=3,
        thread_id=12,
        question="检索急刹",
        append_user_message=False,
        live_response=placeholder,
    )

    assert returned is state
    assert len(placeholder.messages) >= 3
    assert "正在" in placeholder.messages[0]
    assert "已找到 5 个急刹事件。" in placeholder.messages[-1]
    assert memory_store.messages[0][1:3] == ("assistant", "已找到 5 个急刹事件。")
