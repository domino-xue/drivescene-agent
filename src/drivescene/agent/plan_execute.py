from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
import json
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from drivescene.agent.digest import ExecutionDigestBuilder
from drivescene.agent.evaluator import SimpleEvaluator
from drivescene.agent.reporting import LLMReporter
from drivescene.agent.tool_registry import ToolRegistry, execute_registered_tool, preflight_tool_call
from drivescene.ops.contracts import validation_error_result


StepStatus = str


@dataclass
class PlanStep:
    step_id: str
    goal: str
    tool_name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"


@dataclass
class StepResult:
    step_id: str
    tool_name: str | None
    args: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class PendingConfirmation:
    step_id: str
    tool_name: str
    args: dict[str, Any]
    risk: dict[str, Any]


@dataclass
class PlanExecuteState:
    user_request: str
    plan: list[PlanStep]
    current_step: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_confirmation: PendingConfirmation | None = None
    final_answer: str = ""
    memory_user_id: int | None = None
    memory_thread_id: int | None = None
    execution_digest: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    replans_used: int = 0
    blackboard: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    denied_confirmations: list[str] = field(default_factory=list)


@dataclass
class PlanEvent:
    type: str
    state: PlanExecuteState
    step: PlanStep | None = None
    result: StepResult | None = None
    pending_confirmation: PendingConfirmation | None = None
    message: str = ""


class Planner(Protocol):
    def create_plan(self, user_request: str) -> list[PlanStep]:
        ...


class HeuristicPlanner:
    def create_plan(self, user_request: str) -> list[PlanStep]:
        planner_request = user_request
        user_request = _strip_context_pack(user_request)
        normalized = user_request.lower()
        steps: list[PlanStep] = []

        new_event_type = _new_scene_label_from_request(user_request, normalized)
        if new_event_type is not None:
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal=f"Search indexed {new_event_type} candidates",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {
                            "event_type": new_event_type,
                            "is_valid_event": True if _asks_for_valid_events(user_request) else None,
                        },
                        "limit": _extract_limit(user_request, default=5),
                    },
                    depends_on=_previous_step_ids(steps),
                )
            )

        if any(keyword in user_request for keyword in ["汇总", "总结", "统计", "概览"]):
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal="汇总当前事件索引",
                    tool_name="EvidenceStore.query_index",
                    args={"target": "events", "operation": "summary"},
                )
            )

        if any(keyword in normalized for keyword in ["hard_braking", "急刹"]):
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal="查找急刹事件案例",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {
                            "event_type": "hard_braking",
                            "is_valid_event": True if "有效" in user_request else None,
                        },
                        "limit": _extract_limit(user_request, default=5),
                    },
                    depends_on=_previous_step_ids(steps),
                )
            )
        elif any(keyword in normalized for keyword in ["close_following", "跟车", "近距离"]):
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal="查找近距离跟车事件案例",
                    tool_name="EvidenceStore.query_index",
                    args={
                        "target": "events",
                        "operation": "search",
                        "filters": {
                            "event_type": "close_following",
                            "is_valid_event": True if "有效" in user_request else None,
                        },
                        "limit": _extract_limit(user_request, default=5),
                    },
                    depends_on=_previous_step_ids(steps),
                )
            )

        if any(keyword in user_request for keyword in ["证据", "动画", "路径"]):
            review_id = _extract_review_id(user_request)
            if review_id is not None:
                steps.append(
                    PlanStep(
                        step_id=_next_step_id(steps),
                        goal="获取事件证据资源路径",
                        tool_name="EvidenceStore.query_index",
                        args={
                            "target": "events",
                            "operation": "evidence",
                            "review_ids": [review_id],
                        },
                        depends_on=_previous_step_ids(steps),
                    )
                )

        if any(keyword in user_request for keyword in ["复制", "导出"]):
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal="创建导出文件夹",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "create_folder", "output_dir": "outputs/exports/demo_cases"},
                    depends_on=_previous_step_ids(steps),
                )
            )

        if "删除" in user_request or "delete" in normalized:
            delete_paths = _extract_resolved_context_paths(planner_request) or [
                "outputs/exports/demo_cases"
            ]
            steps.append(
                PlanStep(
                    step_id=_next_step_id(steps),
                    goal="删除用户指定的文件或文件夹",
                    tool_name="ArtifactOps.manage_artifacts",
                    args={"operation": "delete", "paths": delete_paths},
                    depends_on=_previous_step_ids(steps),
                )
            )

        if not steps:
            steps.append(
                PlanStep(
                    step_id="step_1",
                    goal="解释当前 agent 能力并请求更具体的场景数据任务",
                    tool_name=None,
                )
            )

        return steps


class LLMJsonPlanner:
    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        *,
        max_attempts: int = 2,
        max_steps: int = 12,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model = model
        self.registry = registry
        self.max_attempts = max_attempts
        self.max_steps = max_steps

    def create_plan(self, user_request: str) -> list[PlanStep]:
        base_prompt = _planner_prompt(user_request, self.registry)
        allowed_external_dependencies = _existing_step_ids_from_replan_request(user_request)
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            prompt = (
                base_prompt
                if not errors
                else _planner_retry_prompt(base_prompt, errors[-1], attempt)
            )
            response = self.model.invoke(prompt)
            content = getattr(response, "content", response)
            try:
                payload = _parse_planner_payload(content)
                plan = _plan_steps_from_payload(payload)
                _validate_plan_structure(
                    plan,
                    self.registry,
                    max_steps=self.max_steps,
                    allowed_external_dependencies=allowed_external_dependencies,
                )
                return plan
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))

        joined_errors = " | ".join(errors)
        raise ValueError(
            f"Planner failed to produce a valid plan after {self.max_attempts} attempts: "
            f"{joined_errors}"
        )


class PlanAndExecuteAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        planner: Planner | None = None,
        memory_writer: Any | None = None,
        digest_builder: Any | None = None,
        evaluator: Any | None = None,
        reporter: Any | None = None,
        max_replans: int = 0,
    ) -> None:
        self.registry = registry
        self._uses_default_planner = planner is None
        self.planner = planner or HeuristicPlanner()
        self.memory_writer = memory_writer
        self.digest_builder = digest_builder or ExecutionDigestBuilder()
        self.evaluator = evaluator or SimpleEvaluator()
        self.reporter = reporter or LLMReporter()
        self.max_replans = max_replans

    def start(
        self,
        user_request: str,
        context_pack: Any | None = None,
        memory_user_id: int | None = None,
        memory_thread_id: int | None = None,
    ) -> PlanExecuteState:
        planner_request = _planner_request_with_context(user_request, context_pack)
        plan = (
            _deterministic_plan_override(planner_request)
            if self._uses_default_planner
            else None
        )
        if plan is None:
            plan = self.planner.create_plan(planner_request)
        self._validate_plan(plan)
        return PlanExecuteState(
            user_request=user_request,
            plan=plan,
            memory_user_id=memory_user_id,
            memory_thread_id=memory_thread_id,
        )

    def run(
        self,
        user_request: str,
        context_pack: Any | None = None,
        memory_user_id: int | None = None,
        memory_thread_id: int | None = None,
    ) -> PlanExecuteState:
        state = self.start(
            user_request,
            context_pack=context_pack,
            memory_user_id=memory_user_id,
            memory_thread_id=memory_thread_id,
        )
        return self.run_until_pause_or_done(state)

    def run_until_pause_or_done(self, state: PlanExecuteState) -> PlanExecuteState:
        ensure_state_runtime_fields(state)
        while True:
            while state.current_step < len(state.plan) and state.pending_confirmation is None:
                self._execute_current_step(state)
            evaluation = self._evaluate_state(state)
            if (
                state.pending_confirmation is not None
                or not evaluation.needs_replan
                or state.replans_used >= self.max_replans
            ):
                break
            replan_steps = self.planner.create_plan(
                _replan_request(state, evaluation, state.execution_digest)
            )
            self._append_replan_steps(state, replan_steps)
            state.replans_used += 1
        state.final_answer = self._build_reporter_final_answer(state)
        return state

    def stream(
        self,
        user_request: str,
        context_pack: Any | None = None,
        memory_user_id: int | None = None,
        memory_thread_id: int | None = None,
    ):
        state = self.start(
            user_request,
            context_pack=context_pack,
            memory_user_id=memory_user_id,
            memory_thread_id=memory_thread_id,
        )
        yield PlanEvent(type="plan_created", state=state, message="计划已生成")
        yield from self.stream_from_state(state)

    def stream_from_state(self, state: PlanExecuteState):
        ensure_state_runtime_fields(state)
        yield from self._stream_from_state_with_replan(state)

    def _stream_from_state_with_replan(self, state: PlanExecuteState):
        while True:
            blocked_step: PlanStep | None = None
            while state.current_step < len(state.plan) and state.pending_confirmation is None:
                step = state.plan[state.current_step]
                if step.status not in {"completed", "failed", "skipped"}:
                    yield PlanEvent(
                        type="step_started",
                        state=state,
                        step=step,
                        message=step.goal,
                    )
                before_results = len(state.step_results)
                self._execute_current_step(state)
                if state.pending_confirmation is not None:
                    state.final_answer = self._build_reporter_final_answer(state)
                    yield PlanEvent(
                        type="confirmation_required",
                        state=state,
                        step=step,
                        pending_confirmation=state.pending_confirmation,
                        message=state.final_answer,
                    )
                    return
                if step.status == "blocked":
                    blocked_step = step
                    yield PlanEvent(
                        type="step_blocked",
                        state=state,
                        step=step,
                        message=f"步骤 {step.step_id} 被阻塞，依赖尚未完成。",
                    )
                    break
                if step.status == "skipped":
                    yield PlanEvent(
                        type="step_skipped",
                        state=state,
                        step=step,
                        message=f"步骤 {step.step_id} 已跳过，因为依赖尚未完成。",
                    )
                    continue
                if len(state.step_results) > before_results:
                    result = state.step_results[-1]
                    event_type = "step_failed" if result.error else "step_completed"
                    yield PlanEvent(
                        type=event_type,
                        state=state,
                        step=step,
                        result=result,
                        message=result.error or step.goal,
                    )

            evaluation = self._evaluate_state(state)
            if (
                state.pending_confirmation is None
                and blocked_step is None
                and evaluation.needs_replan
                and state.replans_used < self.max_replans
            ):
                replan_steps = self.planner.create_plan(
                    _replan_request(state, evaluation, state.execution_digest)
                )
                self._append_replan_steps(state, replan_steps)
                state.replans_used += 1
                yield PlanEvent(
                    type="plan_updated",
                    state=state,
                    message="Planner added follow-up steps after evaluation.",
                )
                continue

            state.final_answer = self._build_reporter_final_answer(state)
            yield PlanEvent(type="final_answer", state=state, message=state.final_answer)
            return

    def resolve_confirmation(
        self,
        state: PlanExecuteState,
        approved: bool,
    ) -> PlanExecuteState:
        if state.pending_confirmation is None:
            return state
        ensure_state_runtime_fields(state)
        step = self._step_by_id(state, state.pending_confirmation.step_id)
        if approved:
            step.status = "confirmed"
        else:
            step.status = "skipped"
            state.denied_confirmations.append(step.step_id)
            state.current_step += 1
        state.pending_confirmation = None
        state.final_answer = self._build_reporter_final_answer(state)
        return state

    def _validate_plan(self, plan: list[PlanStep]) -> None:
        _validate_plan_structure(
            plan,
            self.registry,
            max_steps=100,
            require_ordered_dependencies=False,
        )

    def _execute_current_step(self, state: PlanExecuteState) -> None:
        ensure_state_runtime_fields(state)
        step = state.plan[state.current_step]
        if step.status in {"completed", "failed", "skipped"}:
            state.current_step += 1
            return
        if not self._dependencies_completed(state, step):
            step.status = "skipped"
            state.pending_confirmation = None
            missing = [
                dependency
                for dependency in step.depends_on
                if self._step_status(state, dependency) != "completed"
            ]
            _record_diagnostic(
                state,
                step,
                phase="dependencies",
                error_type="DependencyNotCompleted",
                error=(
                    f"Step {step.step_id} skipped because dependencies are not "
                    f"completed: {', '.join(missing)}"
                ),
            )
            state.current_step += 1
            return
        if step.tool_name is None:
            step.status = "completed"
            output = {
                "ok": True,
                "operation": "message",
                "summary": {"total": 1, "succeeded": 1, "failed": 0},
                "items": [{"input": {}, "ok": True, "result": step.goal, "error": None}],
            }
            state.step_results.append(
                StepResult(step_id=step.step_id, tool_name=None, args={}, result=output)
            )
            state.steps[step.step_id] = {"tool": None, "args": {}, "output": output}
            state.current_step += 1
            return

        try:
            resolved_args = resolve_references(step.args, state)
            resolved_args = bind_tool_arguments(step.tool_name, resolved_args, state)
            resolved_args = validate_tool_arguments(
                step.tool_name,
                resolved_args,
            )
        except ValueError as error:
            recovered_args = recover_tool_arguments_from_state(
                step.tool_name,
                step.args,
                state,
                error,
            )
            if recovered_args is not None:
                try:
                    resolved_args = validate_tool_arguments(
                        step.tool_name,
                        recovered_args,
                    )
                except ValueError as recovered_error:
                    error = recovered_error
                else:
                    _record_diagnostic(
                        state,
                        step,
                        phase="resolve_or_validate",
                        error_type=type(error).__name__,
                        error=f"Recovered from invalid reference using typed state: {error}",
                    )
                    return self._execute_resolved_tool_call(state, step, resolved_args)
            step.status = "failed"
            result = validation_error_result("resolve_references", error, step.args)
            state.step_results.append(
                StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    args=step.args,
                    result=result,
                    error=str(error),
                )
            )
            state.steps[step.step_id] = {
                "tool": step.tool_name,
                "args": step.args,
                "output": result,
            }
            _record_diagnostic(
                state,
                step,
                phase="resolve_or_validate",
                error_type=type(error).__name__,
                error=str(error),
            )
            state.current_step += 1
            return

        self._execute_resolved_tool_call(state, step, resolved_args)

    def _execute_resolved_tool_call(
        self,
        state: PlanExecuteState,
        step: PlanStep,
        resolved_args: dict[str, Any],
    ) -> None:
        risk = preflight_tool_call(
            self.registry,
            {"tool": step.tool_name, "args": resolved_args},
        )["risk"]
        if risk["requires_confirmation"] and step.status != "confirmed":
            step.status = "blocked"
            state.pending_confirmation = PendingConfirmation(
                step_id=step.step_id,
                tool_name=step.tool_name,
                args=resolved_args,
                risk=risk,
            )
            return

        step.status = "running"
        try:
            result = execute_registered_tool(
                self.registry,
                {"tool": step.tool_name, "args": resolved_args},
            )["result"]
        except Exception as error:  # noqa: BLE001
            step.status = "failed"
            result = validation_error_result("tool_execution", error, resolved_args)
            state.step_results.append(
                StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    args=resolved_args,
                    result=result,
                    error=str(error),
                )
            )
            state.steps[step.step_id] = {
                "tool": step.tool_name,
                "args": resolved_args,
                "output": result,
            }
            _record_diagnostic(
                state,
                step,
                phase="tool_execution",
                error_type=type(error).__name__,
                error=str(error),
            )
            state.current_step += 1
            return

        step.status = "completed"
        state.step_results.append(
            StepResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                args=resolved_args,
                result=result,
            )
        )
        state.steps[step.step_id] = {
            "tool": step.tool_name,
            "args": resolved_args,
            "output": result,
        }
        _update_blackboard_from_step(state, step, result)
        self._record_memory(state, step, result)
        state.current_step += 1

    def _record_memory(
        self,
        state: PlanExecuteState,
        step: PlanStep,
        result: Any,
    ) -> None:
        if (
            self.memory_writer is None
            or state.memory_user_id is None
            or state.memory_thread_id is None
        ):
            return
        self.memory_writer.record_tool_result(
            user_id=state.memory_user_id,
            thread_id=state.memory_thread_id,
            user_request=state.user_request,
            tool_name=step.tool_name,
            result=result,
        )

    def _dependencies_completed(self, state: PlanExecuteState, step: PlanStep) -> bool:
        status_by_id = {item.step_id: item.status for item in state.plan}
        return all(status_by_id.get(dep) == "completed" for dep in step.depends_on)

    def _step_status(self, state: PlanExecuteState, step_id: str) -> str | None:
        for item in state.plan:
            if item.step_id == step_id:
                return item.status
        return None

    def _step_by_id(self, state: PlanExecuteState, step_id: str) -> PlanStep:
        for step in state.plan:
            if step.step_id == step_id:
                return step
        raise KeyError(step_id)

    def _evaluate_state(self, state: PlanExecuteState) -> Any:
        ensure_state_runtime_fields(state)
        digest = self.digest_builder.build(state)
        state.execution_digest = digest.to_dict()
        evaluation = self.evaluator.evaluate(state, digest)
        state.evaluation = evaluation.to_dict()
        return evaluation

    def _append_replan_steps(
        self,
        state: PlanExecuteState,
        steps: list[PlanStep],
    ) -> None:
        existing = {step.step_id for step in state.plan}
        for step in steps:
            if step.tool_name is None and _is_diagnostic_replan_step(step):
                _record_diagnostic(
                    state,
                    step,
                    phase="replan",
                    error_type="IgnoredNoToolReplanStep",
                    error=step.goal,
                )
                continue
            if step.step_id in existing:
                step.step_id = _next_unique_step_id(existing)
            step.status = "pending"
            existing.add(step.step_id)
            state.plan.append(step)

    def _build_reporter_final_answer(self, state: PlanExecuteState) -> str:
        ensure_state_runtime_fields(state)
        if state.pending_confirmation is not None:
            pending = state.pending_confirmation
            return (
                f"步骤 {pending.step_id} 需要人工确认后才能继续执行：{pending.tool_name}。\n\n"
                f"风险等级：{pending.risk['risk_level']}\n"
                f"原因：{pending.risk['reason']}"
            )
        digest = self.digest_builder.build(state)
        state.execution_digest = digest.to_dict()
        evaluation = self.evaluator.evaluate(state, digest)
        state.evaluation = evaluation.to_dict()
        if evaluation.status == "cancelled":
            return "已按你的选择取消高风险操作，相关文件或表格未被修改。"
        return self.reporter.generate(digest)


def _next_step_id(steps: list[PlanStep]) -> str:
    return f"step_{len(steps) + 1}"


def _next_unique_step_id(existing: set[str]) -> str:
    index = 1
    while f"step_{index}" in existing:
        index += 1
    return f"step_{index}"


def _previous_step_ids(steps: list[PlanStep]) -> list[str]:
    return [steps[-1].step_id] if steps else []


def ensure_state_runtime_fields(state: Any) -> None:
    if not hasattr(state, "blackboard"):
        setattr(state, "blackboard", {})
    if not hasattr(state, "diagnostics"):
        setattr(state, "diagnostics", [])
    if not hasattr(state, "denied_confirmations"):
        setattr(state, "denied_confirmations", [])


def _replan_request(
    state: PlanExecuteState,
    evaluation: Any,
    digest: dict[str, Any],
) -> str:
    ensure_state_runtime_fields(state)
    payload = {
        "original_user_request": state.user_request,
        "missing_requirements": getattr(evaluation, "missing_requirements", []),
        "evaluation": evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation,
        "execution_digest": digest,
        "blackboard": state.blackboard,
        "diagnostics": state.diagnostics,
        "allowed_state_references": [
            {"$from_state": "review_ids"},
            {"$from_state": "events"},
            {"$from_state": "top_event"},
            {"$from_state": "evidence_assets"},
            {"$from_state": "evidence_paths"},
            {"$from_state": "export_dirs"},
            {"$from_state": "manifest_paths"},
            {"$from_state": "copied_files"},
        ],
        "existing_steps": [
            {
                "step_id": step.step_id,
                "goal": step.goal,
                "tool_name": step.tool_name,
                "status": step.status,
            }
            for step in state.plan
        ],
    }
    return (
        "The previous plan did not fully satisfy the task. "
        "Create only the additional pending steps needed to complete it. "
        "Do not repeat completed steps. "
        "Do not reference diagnostic or failed no-tool steps as data sources. "
        "Do not create no-tool steps that merely restate an error; every replan step "
        "must either call a real tool with corrected state references or be omitted. "
        "Prefer typed state references such as {'$from_state':'review_ids'} or "
        "{'$from_state':'evidence_paths'} over brittle output selectors. "
        "Return the same plan JSON schema.\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )


def _planner_request_with_context(user_request: str, context_pack: Any | None) -> str:
    if context_pack is None:
        return user_request
    payload = {
        "resolved_context": getattr(context_pack, "resolved_context", {}),
        "unresolved_references": getattr(context_pack, "unresolved_references", []),
        "structured_memory": getattr(context_pack, "structured_memory", []),
        "thread_summary": getattr(context_pack, "thread_summary", ""),
    }
    return (
        user_request
        + "\n\n[context_pack]\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n[/context_pack]"
    )


def _record_diagnostic(
    state: PlanExecuteState,
    step: PlanStep,
    phase: str,
    error_type: str,
    error: str,
) -> None:
    state.diagnostics.append(
        {
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "phase": phase,
            "error_type": error_type,
            "error": error,
        }
    )


def _update_blackboard_from_step(
    state: PlanExecuteState,
    step: PlanStep,
    result: Any,
) -> None:
    if not isinstance(result, dict):
        return
    if step.tool_name == "EvidenceStore.query_index":
        _update_blackboard_from_evidence_result(state, result)
    elif step.tool_name == "ArtifactOps.manage_artifacts":
        _update_blackboard_from_artifact_result(state, result)


def _update_blackboard_from_evidence_result(
    state: PlanExecuteState,
    result: dict[str, Any],
) -> None:
    operation = str(result.get("operation") or "")
    event_items: list[dict[str, Any]] = []
    for item in result.get("items", []):
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        item_result = item.get("result")
        if not isinstance(item_result, dict):
            continue
        if item_result.get("review_id"):
            _append_unique_state_value(
                state.blackboard,
                "review_ids",
                str(item_result["review_id"]),
            )
            event_items.append(item_result)
            _append_unique_state_dict(state.blackboard, "events", item_result)
        paths = _extract_path_values(item_result)
        if paths:
            asset = {
                "review_id": item_result.get("review_id"),
                "paths": paths,
            }
            _append_unique_state_dict(state.blackboard, "evidence_assets", asset)
            for path in paths:
                _append_unique_state_value(state.blackboard, "evidence_paths", path)

    if operation in {"search", "detail"} and event_items and not state.blackboard.get(
        "top_event"
    ):
        state.blackboard["top_event"] = dict(event_items[0])
    if operation == "evidence" and event_items and not state.blackboard.get("top_event"):
        state.blackboard["top_event"] = dict(event_items[0])


def _update_blackboard_from_artifact_result(
    state: PlanExecuteState,
    result: dict[str, Any],
) -> None:
    if result.get("output_dir"):
        _append_unique_state_value(state.blackboard, "export_dirs", str(result["output_dir"]))
    if result.get("manifest_path"):
        _append_unique_state_value(
            state.blackboard,
            "manifest_paths",
            str(result["manifest_path"]),
        )
    for copied_file in result.get("copied_files", []) or []:
        _append_unique_state_value(state.blackboard, "copied_files", str(copied_file))


def _extract_path_values(item_result: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key, value in item_result.items():
        if key.endswith("_path") and value is not None and value != "":
            paths.append(str(value))
    return paths


def _append_unique_state_value(
    blackboard: dict[str, Any],
    key: str,
    value: str,
) -> None:
    values = blackboard.setdefault(key, [])
    if value not in values:
        values.append(value)


def _append_unique_state_dict(
    blackboard: dict[str, Any],
    key: str,
    value: dict[str, Any],
) -> None:
    values = blackboard.setdefault(key, [])
    normalized = dict(value)
    if normalized not in values:
        values.append(normalized)


def _latest_state_value(blackboard: dict[str, Any], key: str) -> Any:
    value = blackboard.get(key)
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def _manifest_from_state(state: PlanExecuteState) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if state.blackboard.get("events"):
        manifest["events"] = list(state.blackboard["events"])
    if state.blackboard.get("top_event"):
        manifest["top_event"] = dict(state.blackboard["top_event"])
    if state.blackboard.get("evidence_assets"):
        manifest["evidence_assets"] = list(state.blackboard["evidence_assets"])
    if state.blackboard.get("copied_files"):
        manifest["copied_files"] = list(state.blackboard["copied_files"])
    return manifest


def _literal_tool_args(
    args: dict[str, Any],
    drop_keys: set[str],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in args.items()
        if key not in drop_keys and not _contains_unresolved_reference(value)
    }


def _literal_arg(args: dict[str, Any], key: str) -> Any:
    if key not in args:
        return None
    value = args[key]
    if _contains_unresolved_reference(value):
        return None
    return value


def _review_ids_from_state_for_args(
    state: PlanExecuteState,
    args: dict[str, Any],
) -> list[str]:
    top_review_id = _top_review_id(state.blackboard)
    if _references_top_item(args) and top_review_id:
        return [top_review_id]
    review_ids = state.blackboard.get("review_ids")
    if isinstance(review_ids, list) and review_ids:
        return [str(review_id) for review_id in review_ids]
    return [top_review_id] if top_review_id else []


def _bind_evidence_copy_layout(
    args: dict[str, Any],
    state: PlanExecuteState,
) -> None:
    paths = args.get("paths")
    if not isinstance(paths, list) or len(paths) < 2:
        return
    string_paths = [str(path) for path in paths if path]
    if not _copy_paths_need_structure(string_paths):
        return
    base_dir = args.get("base_dir") or _infer_evidence_base_dir(string_paths, state)
    if base_dir is None:
        return
    args["preserve_structure"] = True
    args["base_dir"] = str(base_dir)


def _copy_paths_need_structure(paths: list[str]) -> bool:
    basenames = [Path(path).name for path in paths]
    return len(basenames) != len(set(basenames))


def _infer_evidence_base_dir(
    paths: list[str],
    state: PlanExecuteState,
) -> str | None:
    review_ids = {
        str(review_id)
        for review_id in state.blackboard.get("review_ids", [])
        if review_id is not None
    }
    candidate_dirs: list[Path] = []
    for path in paths:
        source = Path(path)
        parent = source.parent
        if parent.name in review_ids:
            candidate_dirs.append(parent.parent)
            continue
        parts = list(source.parts)
        if "review_assets" in parts:
            index = parts.index("review_assets")
            candidate_dirs.append(Path(*parts[: index + 1]))
    if not candidate_dirs:
        return None
    first = candidate_dirs[0]
    if all(candidate == first for candidate in candidate_dirs):
        return str(first)
    return None


def _top_review_id(blackboard: dict[str, Any]) -> str | None:
    top_event = blackboard.get("top_event")
    if isinstance(top_event, dict) and top_event.get("review_id"):
        return str(top_event["review_id"])
    review_ids = blackboard.get("review_ids")
    if isinstance(review_ids, list) and review_ids:
        return str(review_ids[0])
    return None


def _references_top_item(value: Any) -> bool:
    if isinstance(value, dict):
        selector = value.get("$select")
        if isinstance(selector, str) and _selector_targets_top_item(selector):
            return True
        if value.get("$from_state") == "top_event":
            return True
        return any(_references_top_item(inner) for inner in value.values())
    if isinstance(value, list):
        return any(_references_top_item(item) for item in value)
    return False


def _selector_targets_top_item(selector: str) -> bool:
    normalized = selector.replace(" ", "")
    return (
        "[0]" in normalized
        or normalized in {"items[0]", "output.items[0]", "output.items[0].result"}
    )


def _is_diagnostic_replan_step(step: PlanStep) -> bool:
    text = step.goal.lower()
    diagnostic_markers = [
        "工具执行失败",
        "错误信息",
        "reference selector",
        "tool execution failed",
        "failed:",
        "无工具",
    ]
    return any(marker in text for marker in diagnostic_markers)


def _deterministic_plan_override(planner_request: str) -> list[PlanStep] | None:
    user_request = _strip_context_pack(planner_request)
    normalized = user_request.lower()
    direct_answer = _extract_resolved_context_value(planner_request, "direct_answer")
    if isinstance(direct_answer, str) and direct_answer:
        return [
            PlanStep(
                step_id="step_1",
                goal=direct_answer,
                tool_name=None,
            )
        ]

    copy_plan = _copy_event_artifacts_plan(user_request, normalized)
    if copy_plan is not None:
        return copy_plan

    review_ids = _extract_resolved_context_review_ids(planner_request)
    explicit_review_id = _extract_review_id(user_request)
    if explicit_review_id is not None:
        review_ids = [explicit_review_id]
    if review_ids and _asks_for_event_evidence(user_request, normalized):
        return [
            PlanStep(
                step_id="step_1",
                goal="获取事件证据资源路径",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "evidence",
                    "review_ids": review_ids,
                },
            )
        ]

    if _asks_for_event_count(user_request, normalized):
        filters = _event_count_filters(user_request, normalized)
        return [
            PlanStep(
                step_id="step_1",
                goal="统计当前事件索引",
                tool_name="EvidenceStore.query_index",
                args={
                    "target": "events",
                    "operation": "summary",
                    "filters": filters,
                },
            )
        ]
    return None


def _copy_event_artifacts_plan(user_request: str, normalized: str) -> list[PlanStep] | None:
    event_type = _event_type_from_request(user_request, normalized)
    if event_type is None:
        return None
    asks_copy = any(keyword in user_request for keyword in ["复制", "拷贝"]) or any(
        keyword in normalized for keyword in ["copy"]
    )
    asks_export = "导出" in user_request or "export" in normalized
    mentions_artifacts = any(
        keyword in user_request
        for keyword in ["事件", "证据", "动画", "案例", "文件", "路径"]
    ) or any(keyword in normalized for keyword in ["event", "evidence", "file", "path"])
    if not (asks_copy or (asks_export and mentions_artifacts)):
        return None

    limit = _extract_limit(user_request, default=5)
    output_dir = _unique_export_dir(f"outputs/exports/{event_type}_top{limit}_copy")
    search_args: dict[str, Any] = {
        "target": "events",
        "operation": "search",
        "filters": {"event_type": event_type},
        "limit": limit,
    }
    sort_by = _event_copy_sort_column(event_type, user_request, normalized)
    if sort_by is not None:
        search_args.update({"sort_by": sort_by, "ascending": True})

    event_label = _event_type_label(event_type)
    rank_label = _event_rank_label(event_type, sort_by)

    return [
        PlanStep(
            step_id="step_1",
            goal=f"查找{rank_label} {limit} 个{event_label}事件",
            tool_name="EvidenceStore.query_index",
            args=search_args,
        ),
        PlanStep(
            step_id="step_2",
            goal=f"获取{event_label}事件证据资源路径",
            tool_name="EvidenceStore.query_index",
            args={
                "target": "events",
                "operation": "evidence",
                "review_ids": {"$from_state": "review_ids"},
            },
            depends_on=["step_1"],
        ),
        PlanStep(
            step_id="step_3",
            goal=f"创建{event_label}事件导出文件夹",
            tool_name="ArtifactOps.manage_artifacts",
            args={"operation": "create_folder", "output_dir": output_dir},
        ),
        PlanStep(
            step_id="step_4",
            goal=f"复制{event_label}事件证据文件",
            tool_name="ArtifactOps.manage_artifacts",
            args={
                "operation": "copy",
                "paths": {"$from_state": "evidence_paths"},
                "output_dir": output_dir,
                "preserve_structure": True,
                "base_dir": "outputs/review_assets",
            },
            depends_on=["step_2", "step_3"],
        ),
        PlanStep(
            step_id="step_5",
            goal=f"写出{event_label}事件导出清单",
            tool_name="ArtifactOps.manage_artifacts",
            args={
                "operation": "write_manifest",
                "output_path": f"{output_dir}/manifest.json",
                "manifest": {
                    "events": {"$from_state": "events"},
                    "top_event": {"$from_state": "top_event"},
                },
            },
            depends_on=["step_1", "step_3"],
        ),
    ]


def _event_copy_sort_column(
    event_type: str,
    user_request: str,
    normalized: str,
) -> str | None:
    if event_type == "hard_braking" and _asks_for_hardest_braking(user_request, normalized):
        return "min_velocity_acceleration_mps2"
    if event_type == "close_following" and _asks_for_closest_following(user_request, normalized):
        return "min_front_distance_m"
    return None


def _asks_for_closest_following(user_request: str, normalized: str) -> bool:
    return any(
        keyword in user_request
        for keyword in ["跟车距离最近", "距离最近", "最近", "最短", "距离最小"]
    ) or any(keyword in normalized for keyword in ["closest", "nearest", "shortest"])


def _event_type_label(event_type: str) -> str:
    labels = {
        "hard_braking": "急刹",
        "close_following": "近距离跟车",
        "cut_in": "cut-in",
        "stopped_vehicle_ahead": "前方静止车辆",
        "lane_change": "lane-change",
    }
    return labels.get(event_type, event_type)


def _event_rank_label(event_type: str, sort_by: str | None) -> str:
    if event_type == "hard_braking" and sort_by == "min_velocity_acceleration_mps2":
        return "刹停加速度最大的"
    if event_type == "close_following" and sort_by == "min_front_distance_m":
        return "跟车距离最近的"
    return "匹配的"


def _unique_export_dir(base_dir: str) -> str:
    base_path = Path(base_dir)
    if not base_path.exists():
        return str(base_path)
    index = 2
    while True:
        candidate = base_path.with_name(f"{base_path.name}_{index}")
        if not candidate.exists():
            return str(candidate)
        index += 1


def _extract_resolved_context_paths(user_request: str) -> list[str]:
    paths = _extract_resolved_context_value(user_request, "paths")
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if path]


def _extract_resolved_context_review_ids(user_request: str) -> list[str]:
    review_ids = _extract_resolved_context_value(user_request, "review_ids")
    if not isinstance(review_ids, list):
        return []
    return [str(review_id) for review_id in review_ids if review_id]


def _extract_resolved_context_value(user_request: str, key: str) -> Any:
    payload = _extract_context_payload(user_request)
    if not payload:
        return None
    return payload.get("resolved_context", {}).get(key)


def _extract_context_payload(user_request: str) -> dict[str, Any]:
    match = re.search(r"\[context_pack\]\s*(.+?)\s*\[/context_pack\]", user_request, re.S)
    if match is None:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strip_context_pack(user_request: str) -> str:
    return re.sub(r"\n*\[context_pack\].+?\[/context_pack\]\s*", "", user_request, flags=re.S)


def _new_scene_label_from_request(user_request: str, normalized: str) -> str | None:
    if "cut_in" in normalized or "cut-in" in normalized or "切入" in user_request:
        return "cut_in"
    if (
        "stopped_vehicle_ahead" in normalized
        or "stopped vehicle" in normalized
        or "前方静止" in user_request
        or "静止车辆" in user_request
        or "低速车辆" in user_request
    ):
        return "stopped_vehicle_ahead"
    if "lane_change" in normalized or "lane change" in normalized or "换道" in user_request:
        return "lane_change"
    return None


def _asks_for_valid_events(user_request: str) -> bool:
    return "有效" in user_request or "valid" in user_request.lower()


def _extract_limit(text: str, default: int) -> int:
    match = re.search(r"(\d+)\s*(个|条|例)?", text)
    if match:
        return int(match.group(1))
    if re.search(r"\b(one|a|an)\b", text.lower()) or any(
        token in text for token in ["一个", "一条", "一例", "1个", "1条", "1例"]
    ):
        return 1
    return default


def _extract_review_id(text: str) -> str | None:
    match = re.search(r"\b\d{6}\b", text)
    if match:
        return match.group(0)
    return None


def _asks_for_event_evidence(user_request: str, normalized: str) -> bool:
    return any(
        keyword in user_request
        for keyword in ["证据", "动画", "路径", "地址", "文件地址", "文件位置", "详情"]
    ) or any(keyword in normalized for keyword in ["evidence", "path", "file", "address"])


def _asks_for_event_count(user_request: str, normalized: str) -> bool:
    asks_count = any(keyword in user_request for keyword in ["多少", "几个", "数量", "总数"])
    asks_count = asks_count or any(keyword in normalized for keyword in ["how many", "count"])
    mentions_events = "事件" in user_request or "标注" in user_request or bool(
        _event_type_from_request(user_request, normalized)
    )
    return asks_count and mentions_events


def _asks_for_hardest_braking(user_request: str, normalized: str) -> bool:
    mentions_braking_strength = any(
        keyword in user_request for keyword in ["刹停加速度", "加速度", "减速度", "制动"]
    ) or any(keyword in normalized for keyword in ["acceleration", "deceleration"])
    asks_extreme = any(keyword in user_request for keyword in ["最大", "最强", "最严重", "最高"]) or any(
        keyword in normalized for keyword in ["max", "largest", "highest", "hardest"]
    )
    return mentions_braking_strength and asks_extreme


def _event_count_filters(user_request: str, normalized: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    event_type = _event_type_from_request(user_request, normalized)
    if event_type is not None:
        filters["event_type"] = event_type
    if any(keyword in user_request for keyword in ["已标注", "人工标注", "人工打标", "复核"]):
        filters["review_status"] = "reviewed"
    if _asks_for_valid_events(user_request):
        filters["is_valid_event"] = True
    return filters


def _event_type_from_request(user_request: str, normalized: str) -> str | None:
    if "hard_braking" in normalized or "急刹" in user_request or "急煞" in user_request:
        return "hard_braking"
    if "close_following" in normalized or "近距离跟车" in user_request or "跟车" in user_request:
        return "close_following"
    new_scene_label = _new_scene_label_from_request(user_request, normalized)
    if new_scene_label is not None:
        return new_scene_label
    return None


def resolve_references(value: Any, state: PlanExecuteState) -> Any:
    ensure_state_runtime_fields(state)
    if _is_reference(value):
        return _resolve_reference(value, state)
    if isinstance(value, dict):
        _reject_legacy_reference_strings(value)
        return {key: resolve_references(inner, state) for key, inner in value.items()}
    if isinstance(value, list):
        return [resolve_references(item, state) for item in value]
    if _is_legacy_reference_string(value):
        raise ValueError(
            "Planner used a legacy string placeholder; use a structured reference "
            "like {'$from_step': 'step_1', '$select': 'output.items[*].result.review_id'}."
        )
    return value


def validate_tool_arguments(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("tool args must be a JSON object")
    if any(_contains_unresolved_reference(value) for value in args.values()):
        raise ValueError("tool args contain an unresolved structured reference")

    if tool_name == "EvidenceStore.query_index":
        _require_type(args, "target", str)
        _require_type(args, "operation", str)
        operation = args["operation"]
        if operation in {"detail", "evidence"}:
            _require_list(args, "review_ids")
        _optional_type(args, "filters", dict)
        _optional_type(args, "sort_by", str)
        _optional_type(args, "ascending", bool)
        _optional_type(args, "limit", int)
    elif tool_name == "ArtifactOps.manage_artifacts":
        _require_type(args, "operation", str)
        operation = args["operation"]
        if operation == "create_folder":
            _require_present(args, "output_dir")
        elif operation == "copy":
            _require_present(args, "output_dir")
            _require_list(args, "paths")
        elif operation == "delete":
            _require_list(args, "paths")
        elif operation == "write_manifest":
            _require_present(args, "output_path")
            _optional_type(args, "manifest", dict)
        _optional_type(args, "preserve_structure", bool)
        _optional_type(args, "overwrite", bool)
        _optional_type(args, "missing_ok", bool)
    elif tool_name == "AnalysisTools.run_analysis":
        _require_type(args, "operation", str)
    elif tool_name == "TableOps.transform_table":
        _require_type(args, "operation", str)
        _require_present(args, "input_path")
        _require_present(args, "output_path")
        _optional_type(args, "columns", list)
        _optional_type(args, "filters", dict)
        _optional_type(args, "overwrite", bool)
    return args


def bind_tool_arguments(
    tool_name: str,
    args: dict[str, Any],
    state: PlanExecuteState,
) -> dict[str, Any]:
    bound = dict(args)
    if tool_name == "EvidenceStore.query_index":
        if (
            bound.get("target") == "events"
            and bound.get("operation") in {"detail", "evidence"}
            and not bound.get("review_ids")
            and state.blackboard.get("review_ids")
        ):
            bound["review_ids"] = list(state.blackboard["review_ids"])
    if tool_name == "ArtifactOps.manage_artifacts":
        if bound.get("operation") == "copy":
            if not bound.get("paths") and state.blackboard.get("evidence_paths"):
                bound["paths"] = list(state.blackboard["evidence_paths"])
            if not bound.get("output_dir"):
                output_dir = _latest_state_value(state.blackboard, "export_dirs")
                if output_dir:
                    bound["output_dir"] = output_dir
            _bind_evidence_copy_layout(bound, state)
        if bound.get("operation") == "write_manifest":
            if not bound.get("manifest"):
                manifest = _manifest_from_state(state)
                if manifest:
                    bound["manifest"] = manifest
            if not bound.get("output_path"):
                output_dir = _latest_state_value(state.blackboard, "export_dirs")
                if output_dir:
                    bound["output_path"] = str(Path(output_dir) / "manifest.json")
    return bound


def recover_tool_arguments_from_state(
    tool_name: str,
    args: dict[str, Any],
    state: PlanExecuteState,
    error: Exception,
) -> dict[str, Any] | None:
    if tool_name == "ArtifactOps.manage_artifacts" and args.get("operation") == "copy":
        recovered = _literal_tool_args(args, drop_keys={"paths", "output_dir"})
        literal_paths = _literal_arg(args, "paths")
        if literal_paths is not None:
            recovered["paths"] = literal_paths
        elif state.blackboard.get("evidence_paths"):
            recovered["paths"] = list(state.blackboard["evidence_paths"])

        literal_output_dir = _literal_arg(args, "output_dir")
        if literal_output_dir is not None:
            recovered["output_dir"] = literal_output_dir
        else:
            output_dir = _latest_state_value(state.blackboard, "export_dirs")
            if output_dir:
                recovered["output_dir"] = output_dir

        if recovered.get("paths"):
            return bind_tool_arguments(tool_name, recovered, state)
    if tool_name == "ArtifactOps.manage_artifacts" and args.get("operation") == "write_manifest":
        manifest = _manifest_from_state(state)
        if manifest:
            recovered = _literal_tool_args(args, drop_keys={"manifest", "output_path"})
            literal_output_path = _literal_arg(args, "output_path")
            if literal_output_path is not None:
                recovered["output_path"] = literal_output_path
            recovered["manifest"] = manifest
            return bind_tool_arguments(tool_name, recovered, state)
    if tool_name == "EvidenceStore.query_index" and args.get("operation") in {
        "detail",
        "evidence",
    }:
        review_ids = _review_ids_from_state_for_args(state, args)
        if review_ids:
            recovered = _literal_tool_args(args, drop_keys={"review_ids"})
            recovered["review_ids"] = review_ids
            return bind_tool_arguments(tool_name, recovered, state)
    return None


def _resolve_reference(reference: Mapping[str, Any], state: PlanExecuteState) -> Any:
    if "$from_state" in reference:
        return _resolve_state_reference(reference, state)
    step_id = reference.get("$from_step")
    selector = reference.get("$select")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("structured reference requires a non-empty $from_step")
    if not isinstance(selector, str) or not selector:
        raise ValueError("structured reference requires a non-empty $select")
    if step_id not in state.steps:
        raise ValueError(f"Cannot resolve reference: step {step_id!r} is not in state")

    selected = _select_path(state.steps[step_id], selector.split("."))
    if selected == []:
        raise ValueError(f"Reference {step_id}.{selector} resolved to no values")
    return selected


def _resolve_state_reference(reference: Mapping[str, Any], state: PlanExecuteState) -> Any:
    slot = reference.get("$from_state")
    selector = reference.get("$select")
    if not isinstance(slot, str) or not slot:
        raise ValueError("state reference requires a non-empty $from_state")
    if slot not in state.blackboard:
        raise ValueError(f"Cannot resolve state reference: {slot!r} is not in blackboard")
    selected = state.blackboard[slot]
    if selector is not None:
        if not isinstance(selector, str) or not selector:
            raise ValueError("state reference $select must be a non-empty string")
        selected = _select_path({"value": selected}, ["value", *selector.split(".")])
    if selected == [] or selected is None:
        raise ValueError(f"State reference {slot} resolved to no values")
    return selected


def _select_path(current: Any, tokens: list[str]) -> Any:
    if not tokens:
        return current
    token = tokens[0]
    rest = tokens[1:]
    indexed = re.fullmatch(r"(?P<field>[^\[]*)\[(?P<index>\d+)\]", token)
    if indexed is not None:
        field = indexed.group("field")
        values = _get_field(current, field) if field else current
        if not isinstance(values, list):
            raise ValueError(f"Reference selector expected list at {token}")
        index = int(indexed.group("index"))
        if index >= len(values):
            raise ValueError(f"Reference selector index out of range: {token}")
        return _select_path(values[index], rest)
    if token.endswith("[*]"):
        field = token[:-3]
        values = _get_field(current, field)
        if not isinstance(values, list):
            raise ValueError(f"Reference selector expected list at {field}[*]")
        results: list[Any] = []
        for item in values:
            try:
                selected = _select_path(item, rest)
            except ValueError as error:
                if _is_missing_selector_value(error):
                    continue
                raise
            if isinstance(selected, list):
                results.extend(selected)
            elif selected is not None:
                results.append(selected)
        return results
    return _select_path(_get_field(current, token), rest)


def _get_field(current: Any, field: str) -> Any:
    if isinstance(current, dict):
        if "*" in field:
            values = [
                value
                for key, value in current.items()
                if fnmatch(str(key), field) and value is not None and value != ""
            ]
            if not values:
                raise ValueError(f"Reference selector field not found: {field}")
            return values
        if field not in current:
            raise ValueError(f"Reference selector field not found: {field}")
        return current[field]
    raise ValueError(f"Reference selector cannot read field {field!r} from {type(current).__name__}")


def _is_missing_selector_value(error: ValueError) -> bool:
    message = str(error)
    return (
        message.startswith("Reference selector field not found:")
        or message.startswith("Reference selector cannot read field")
    )


def _is_reference(value: Any) -> bool:
    return _is_step_reference(value) or _is_state_reference(value)


def _is_step_reference(value: Any) -> bool:
    return isinstance(value, dict) and "$from_step" in value and "$select" in value


def _is_state_reference(value: Any) -> bool:
    return isinstance(value, dict) and "$from_state" in value


def _contains_unresolved_reference(value: Any) -> bool:
    if _is_reference(value):
        return True
    if isinstance(value, dict):
        return any(_contains_unresolved_reference(inner) for inner in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved_reference(item) for item in value)
    return False


def _reject_legacy_reference_strings(value: dict[str, Any]) -> None:
    for inner in value.values():
        if _is_legacy_reference_string(inner):
            raise ValueError(
                "Planner used a legacy string placeholder; use a structured reference "
                "like {'$from_step': 'step_1', '$select': 'output.items[*].result.review_id'}."
            )


def _is_legacy_reference_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return normalized in {"$", "previous_result"} or (
        normalized.startswith("${") and normalized.endswith("}")
    )


def _require_present(args: dict[str, Any], key: str) -> None:
    if key not in args or args[key] is None:
        raise ValueError(f"{key} is required")


def _require_type(args: dict[str, Any], key: str, expected: type) -> None:
    _require_present(args, key)
    if not isinstance(args[key], expected):
        raise ValueError(f"{key} must be {expected.__name__}")


def _optional_type(args: dict[str, Any], key: str, expected: type) -> None:
    if key in args and args[key] is not None and not isinstance(args[key], expected):
        raise ValueError(f"{key} must be {expected.__name__}")


def _require_list(args: dict[str, Any], key: str) -> None:
    _require_present(args, key)
    if not isinstance(args[key], list):
        raise ValueError(f"{key} must be list")


def _parse_planner_payload(content: Any) -> Any:
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        if content and all(isinstance(block, Mapping) and "text" in block for block in content):
            content = "".join(str(block["text"]) for block in content)
        else:
            return content
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    if not isinstance(content, str):
        raise ValueError(
            f"Planner response must be JSON text, object, or array; got {type(content).__name__}"
        )

    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced is not None:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return payload
        raise ValueError(f"Planner did not return valid JSON: {direct_error}") from direct_error


def _plan_steps_from_payload(payload: Any) -> list[PlanStep]:
    raw_steps = payload.get("steps") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Planner payload must contain a non-empty steps array")

    plan: list[PlanStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Plan step {index} must be an object")
        if "step_id" not in item or not str(item["step_id"]).strip():
            raise ValueError(f"Plan step {index} requires a non-empty step_id")
        if "goal" not in item or not str(item["goal"]).strip():
            raise ValueError(f"Plan step {index} requires a non-empty goal")
        tool_name = item.get("tool_name")
        if tool_name is not None and not isinstance(tool_name, str):
            raise ValueError(f"Plan step {index} tool_name must be a string or null")
        args = item.get("args") or {}
        if not isinstance(args, Mapping):
            raise ValueError(f"Plan step {index} args must be an object")
        depends_on = item.get("depends_on") or []
        if not isinstance(depends_on, list) or not all(
            isinstance(dependency, str) for dependency in depends_on
        ):
            raise ValueError(f"Plan step {index} depends_on must be an array of step ids")
        status = str(item.get("status") or "pending")
        if status != "pending":
            raise ValueError(f"Plan step {index} status must be pending")
        plan.append(
            PlanStep(
                step_id=str(item["step_id"]).strip(),
                goal=str(item["goal"]).strip(),
                tool_name=tool_name,
                args=dict(args),
                depends_on=list(depends_on),
                status=status,
            )
        )
    return plan


def _validate_plan_structure(
    plan: list[PlanStep],
    registry: ToolRegistry,
    *,
    max_steps: int,
    allowed_external_dependencies: set[str] | None = None,
    require_ordered_dependencies: bool = True,
) -> None:
    if not plan:
        raise ValueError("Plan must contain at least one step")
    if len(plan) > max_steps:
        raise ValueError(f"Plan contains {len(plan)} steps; maximum allowed is {max_steps}")

    seen: set[str] = set(allowed_external_dependencies or set())
    current_step_ids: set[str] = set()
    for step in plan:
        if not step.step_id.strip():
            raise ValueError("Plan step_id must not be empty")
        if step.step_id in current_step_ids:
            raise ValueError(f"Duplicate plan step_id: {step.step_id}")
        if not step.goal.strip():
            raise ValueError(f"Plan step {step.step_id} goal must not be empty")
        if step.status != "pending":
            raise ValueError(f"Plan step {step.step_id} must start with pending status")
        if step.tool_name is not None and step.tool_name not in registry:
            raise ValueError(f"Unknown tool in plan step {step.step_id}: {step.tool_name}")
        for dependency in step.depends_on:
            if dependency == step.step_id:
                raise ValueError(f"Plan step {step.step_id} cannot depend on itself")
            if require_ordered_dependencies and dependency not in seen:
                raise ValueError(
                    f"Plan step {step.step_id} depends on {dependency}, which must appear earlier"
                )
        seen.add(step.step_id)
        current_step_ids.add(step.step_id)


def _existing_step_ids_from_replan_request(user_request: str) -> set[str]:
    if "existing_steps" not in user_request:
        return set()
    return {
        match.group(1)
        for match in re.finditer(r'"step_id"\s*:\s*"([^"]+)"', user_request)
    }


def _planner_retry_prompt(base_prompt: str, error: str, attempt: int) -> str:
    return (
        base_prompt
        + "\nThe previous planner response failed schema validation. "
        + f"Correction attempt {attempt}. Validation error: {_compact(error, max_length=800)}\n"
        + "Return one corrected JSON object only. Do not explain the correction."
    )


def _planner_prompt(user_request: str, registry: ToolRegistry) -> str:
    tool_lines = []
    for name, spec in registry.items():
        tool_lines.append(
            "\n".join(
                [
                    f"- tool_name: {name}",
                    f"  category: {spec.category}",
                    f"  purpose: {spec.purpose}",
                    f"  input: {spec.input_contract}",
                    f"  output: {spec.output_contract}",
                    f"  when_to_use: {spec.when_to_use}",
                    f"  do_not_use_when: {spec.do_not_use_when}",
                    f"  zh_note: {spec.zh_note}",
                    f"  args_schema: {spec.args_schema}",
                    f"  examples: {spec.examples}",
                ]
            )
        )
    return (
        "You are the Planner for an autonomous-driving scenario mining agent.\n"
        "Return JSON only. Do not output Markdown.\n"
        "Plan schema: {\"steps\":[{\"step_id\":\"step_1\",\"goal\":\"...\","
        "\"tool_name\":\"EvidenceStore.query_index or null\",\"args\":{},"
        "\"depends_on\":[],\"status\":\"pending\"}]}.\n"
        "Prefer generic parameterized tools. Use filters.event_type for hard_braking "
        "or close_following instead of inventing specialized tool names.\n"
        "Strict argument rules: use EvidenceStore operation='summary' for counts/statistics "
        "and operation='search' for event rows. Scenario filters are exactly city, object_type, "
        "and has_map; normalize city values to lowercase. For strongest hard braking sort by "
        "min_velocity_acceleration_mps2 ascending=true. For closest following sort by "
        "min_front_distance_m ascending=true. Analysis windows use the exact keys "
        "start_timestep and end_timestep. Preserve user-provided ids and paths verbatim. "
        "Do not set overwrite=true unless the user explicitly requests overwrite or an in-place "
        "table modification.\n"
        "Never use string placeholders such as '$', '${step_1}', or 'previous_result'. "
        "To pass a previous step output into a later tool, use a structured reference object: "
        "{\"$from_step\":\"step_1\",\"$select\":\"output.items[*].result.review_id\"}. "
        "Prefer typed state references for common handoffs: "
        "{\"$from_state\":\"review_ids\"} for event ids and "
        "{\"$from_state\":\"evidence_paths\"} for files to copy, and the latest "
        "export folder from typed state instead of selecting items[0] from create_folder. "
        "Do not use selectors like items[0] directly against a step envelope; use "
        "output.items[0].result only for rare low-level reads, or typed state top_event "
        "when you mean the top ranked event. "
        "For copying event evidence, first search events, then query evidence, then call "
        "ArtifactOps.manage_artifacts(operation='copy') with paths from typed state or omit "
        "paths so the executor can bind evidence_paths. "
        "The executor resolves references and binds typed state before validating and calling tools.\n"
        "All tool outputs are state-transform envelopes with ok, operation, summary, and items. "
        "Each item has input, ok, result, error, and optional error_type.\n"
        f"User request: {user_request}\n"
        "Available tools:\n"
        + "\n".join(tool_lines)
    )


def _compact(value: Any, max_length: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
