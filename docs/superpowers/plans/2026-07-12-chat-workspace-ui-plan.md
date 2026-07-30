# Chat Workspace UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPT-like Streamlit chat workspace where message history scrolls inside the chat panel and execution details stay compact.

**Architecture:** Keep the existing sidebar, auth, thread, memory, and agent runtime paths. Add small UI helper functions in `scripts/agent_app.py` for CSS injection, chat workspace rendering, and execution detail placement, then update tests in `tests/test_agent_app.py`.

**Tech Stack:** Python 3.11, Streamlit, pytest.

## Global Constraints

- Only change app presentation behavior and tests.
- Do not change planner, executor, tools, memory schema, or runtime semantics.
- Keep State Graph optional and Plan-and-Execute as the default runtime.
- Keep execution details available but collapsed by default.
- Restore user-visible Chinese labels to readable UTF-8 text where touched.

---

### Task 1: App UI Helpers

**Files:**
- Modify: `scripts/agent_app.py`
- Test: `tests/test_agent_app.py`

**Interfaces:**
- Produces: `chat_workspace_css() -> str`
- Produces: `render_app_shell_styles() -> None`

- [ ] **Step 1: Write failing tests**

Add tests that assert `chat_workspace_css()` includes fixed chat workspace selectors and scroll behavior.

- [ ] **Step 2: Run focused app tests**

Run: `D:\app\envs\agent\python.exe -m pytest tests/test_agent_app.py -q`

Expected: fail because `chat_workspace_css` does not exist yet.

- [ ] **Step 3: Implement UI helper functions**

Add `chat_workspace_css()` returning scoped CSS and `render_app_shell_styles()` calling `st.markdown(..., unsafe_allow_html=True)`.

- [ ] **Step 4: Run focused app tests**

Run: `D:\app\envs\agent\python.exe -m pytest tests/test_agent_app.py -q`

Expected: pass.

### Task 2: Chat-First Main Layout

**Files:**
- Modify: `scripts/agent_app.py`
- Test: `tests/test_agent_app.py`

**Interfaces:**
- Produces: `_render_chat_workspace(memory_store: MemoryStore, thread: Thread) -> str | None`
- Consumes: existing `_run_user_turn(...)`

- [ ] **Step 1: Write tests for helper presence and safe defaults**

Add a test around a tiny fake memory store showing `_render_chat_workspace` can render without an agent state and returns the submitted question or `None`.

- [ ] **Step 2: Run focused app tests**

Run: `D:\app\envs\agent\python.exe -m pytest tests/test_agent_app.py -q`

Expected: fail until helper exists.

- [ ] **Step 3: Replace main two-column layout**

In `main()`, call `render_app_shell_styles()`, render `_render_chat_workspace(...)`, then render `_render_agent_state_panel(...)` inside a compact expander below the chat area.

- [ ] **Step 4: Run focused app tests**

Run: `D:\app\envs\agent\python.exe -m pytest tests/test_agent_app.py -q`

Expected: pass.

### Task 3: Manual Verification And Regression

**Files:**
- Modify: `scripts/agent_app.py`
- Test: existing test suite

**Interfaces:**
- Consumes: all helpers from Tasks 1 and 2.

- [ ] **Step 1: Run app tests**

Run: `D:\app\envs\agent\python.exe -m pytest tests/test_agent_app.py tests/test_agent_app_imports.py -q`

Expected: pass.

- [ ] **Step 2: Run full regression**

Run: `D:\app\envs\agent\python.exe -m pytest -q`

Expected: pass.

- [ ] **Step 3: Start Streamlit**

Run: `D:\app\envs\agent\python.exe -m streamlit run scripts\agent_app.py --server.port 8501 --server.headless true`

Expected: page loads at `http://localhost:8501` with a fixed-height chat panel and collapsed execution details.
