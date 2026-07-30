# UI LLM-Only Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the Streamlit user interface always uses the configured LLM planner and cannot select the unreliable heuristic fallback.

**Architecture:** Keep `HeuristicPlanner` and the CLI `--use-heuristic-planner` flag unchanged for explicit local diagnosis. Remove only the Streamlit control and hard-code the UI agent creation path to `use_heuristic_planner=False`; its cache key then rebuilds any stale heuristic session.

**Tech Stack:** Python 3.11, Streamlit, pytest.

## Global Constraints

- Do not delete or rewrite `HeuristicPlanner`.
- Do not add an automatic fallback after an LLM service failure.
- Do not modify tool registry, risk policy, state binding, or memory behavior.
- Keep CLI fallback activation explicit through `--use-heuristic-planner`.

---

### Task 1: Lock the UI to the LLM planner

**Files:**
- Modify: `E:/drivescene-agent/scripts/agent_app.py`
- Modify: `E:/drivescene-agent/tests/test_agent_app.py`
- Test: `E:/drivescene-agent/tests/test_agent_cli.py`

**Interfaces:**
- Consumes: `_get_agent(..., use_heuristic_planner: bool, ...)`.
- Produces: a Streamlit path that always calls `_get_agent(..., use_heuristic_planner=False, ...)`.
- Preserves: `build_planner(..., use_heuristic_planner=True)` and CLI `--use-heuristic-planner`.

- [ ] **Step 1: Write failing UI tests**

Add a test that reads `scripts/agent_app.py` and asserts the user-visible checkbox label is absent, and add a test asserting the UI source passes `use_heuristic_planner=False` to `_get_agent`.

```python
def test_ui_does_not_expose_heuristic_fallback() -> None:
    source = Path("scripts/agent_app.py").read_text(encoding="utf-8")
    assert 'st.checkbox("使用规则 fallback planner"' not in source
    assert "use_heuristic_planner=False," in source
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
& 'D:\app\envs\agent\python.exe' -m pytest tests/test_agent_app.py -q
```

Expected: fail because the current Streamlit sidebar still contains the fallback checkbox.

- [ ] **Step 3: Implement the minimal UI change**

Remove the checkbox assignment and change the `_get_agent` call in `main()` to this literal argument:

```python
use_heuristic_planner=False,
```

- [ ] **Step 4: Verify app and CLI behavior**

Run:

```powershell
& 'D:\app\envs\agent\python.exe' -m pytest tests/test_agent_app.py tests/test_agent_cli.py -q
```

Expected: pass; CLI heuristic planner tests remain green.

### Task 2: Regression and real-model evaluation

**Files:**
- Modify: no production files
- Verify: `E:/drivescene-agent/tests`

**Interfaces:**
- Consumes: configured LLM via `config/model.yml`.
- Produces: evidence for LLM planning, tool execution, batch copy, and high-risk confirmation.

- [ ] **Step 1: Run the full regression suite**

Run:

```powershell
& 'D:\app\envs\agent\python.exe' -m pytest -q
```

Expected: pass.

- [ ] **Step 2: Run an LLM service probe**

Invoke the configured `ChatOpenAI` model with `Reply with exactly: SERVICE_OK` and require that exact content.

- [ ] **Step 3: Run four non-destructive real-model cases**

Evaluate these prompts through `PlanAndExecuteAgent.stream`:

1. Summarize the event index using a tool.
2. Find three valid hard-braking events with evidence paths.
3. Copy evidence for two close-following events to a new `outputs/agent_eval_exports` directory.
4. Request deletion of that directory without approving confirmation.

Record event types, tools selected, step status, result summary, copied-file count, and confirmation state. Require task 4 to pause without deletion.
