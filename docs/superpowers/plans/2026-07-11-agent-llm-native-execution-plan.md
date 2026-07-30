# Agent LLM-Native Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add typed execution state and contract-aware argument binding so LLM planning can stay semantic while tool execution remains reliable.

**Architecture:** `PlanAndExecuteAgent` keeps a typed blackboard and diagnostics alongside raw step outputs. References resolve either from completed step output or from typed state slots. Failures are recorded as diagnostics and item-level tool errors, not as completed plan steps.

**Tech Stack:** Python dataclasses, existing pytest suite, no new runtime dependency.

## Global Constraints

- Do not add new agent-facing tools.
- Do not hard-code a single user request as the solution.
- Keep tool outputs as state-transform envelopes.
- Preserve batch item-level error behavior.
- Do not delete or rewrite unrelated workspace changes.

---

### Task 1: Typed State And Diagnostics

**Files:**
- Modify: `src/drivescene/agent/plan_execute.py`
- Test: `tests/test_plan_execute_agent.py`

**Interfaces:**
- Produces: `PlanExecuteState.blackboard: dict[str, Any]`
- Produces: `PlanExecuteState.diagnostics: list[dict[str, Any]]`

- [ ] Write failing tests showing failed tool calls do not append `replan_step_*`.
- [ ] Add `blackboard` and `diagnostics` to `PlanExecuteState`.
- [ ] Replace `_append_failure_replan_step` with diagnostic recording.
- [ ] Run focused plan executor tests.

### Task 2: Blackboard Extraction

**Files:**
- Modify: `src/drivescene/agent/plan_execute.py`
- Test: `tests/test_plan_execute_agent.py`

**Interfaces:**
- Produces: `_update_blackboard_from_step(state, step, result) -> None`
- Produces slots: `review_ids`, `events`, `top_event`, `evidence_assets`, `evidence_paths`, `export_dirs`, `manifest_paths`, `copied_files`

- [ ] Write failing tests for blackboard extraction from event search, evidence lookup, and artifact copy.
- [ ] Implement extraction from successful `EvidenceStore.query_index` outputs.
- [ ] Implement extraction from successful `ArtifactOps.manage_artifacts` outputs.
- [ ] Run focused tests.

### Task 3: State References And Binding

**Files:**
- Modify: `src/drivescene/agent/plan_execute.py`
- Modify: `src/drivescene/agent/tool_registry.py`
- Test: `tests/test_plan_execute_agent.py`

**Interfaces:**
- Produces: `{"$from_state": "evidence_paths"}` reference support.
- Produces: `bind_tool_arguments(tool_name, args, state) -> dict[str, Any]`

- [ ] Write failing tests for `$from_state` references.
- [ ] Write failing tests for artifact copy binding from `evidence_paths`.
- [ ] Implement `$from_state` resolver.
- [ ] Implement contract-aware binding for evidence `review_ids` and copy `paths`.
- [ ] Update tool examples to prefer typed references.

### Task 4: Replan Context

**Files:**
- Modify: `src/drivescene/agent/plan_execute.py`
- Test: `tests/test_plan_execute_agent.py`

**Interfaces:**
- Modifies: `_replan_request(state, evaluation, digest) -> str`

- [ ] Write failing tests that replan request includes blackboard and diagnostics separately.
- [ ] Remove fake no-tool diagnostic steps from replan-visible data.
- [ ] Include allowed state references in replan prompt.
- [ ] Run replan tests.

### Task 5: Verification

**Files:**
- Test: all related tests

- [ ] Run `pytest tests/test_plan_execute_agent.py tests/test_tool_registry.py -q`.
- [ ] Run `pytest -q`.
- [ ] Run the real copy hard-braking request through the app-compatible LLM planner path when available, or through a fake LLM planner reproducing the selector failure.
