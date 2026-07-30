from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from drivescene.agent.digest import ExecutionDigest
from drivescene.agent.evaluator import TaskCompletionEvaluator
from drivescene.agent.plan_execute import PlanStep
from drivescene.agent.tool_registry import ToolRegistry, preflight_tool_call


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, bool]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "checks": self.checks,
            "details": self.details,
        }


def load_jsonl_cases(path: Path | str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict) or not payload.get("id"):
                raise ValueError(f"Eval case at {path}:{line_number} requires an id")
            cases.append(payload)
    return cases


def evaluate_planner_case(
    case: dict[str, Any],
    plan: list[PlanStep],
    registry: ToolRegistry,
) -> EvalCaseResult:
    tool_names = [step.tool_name for step in plan if step.tool_name is not None]
    required_tools = set(case.get("required_tools") or [])
    forbidden_tools = set(case.get("forbidden_tools") or [])
    max_steps = int(case.get("max_steps", 12))
    expected_confirmation = case.get("expected_confirmation")

    checks = {
        "step_count": 0 < len(plan) <= max_steps,
        "known_tools": all(tool_name in registry for tool_name in tool_names),
        "dependencies": _dependencies_are_ordered(plan),
        "required_tools": required_tools.issubset(set(tool_names)),
        "forbidden_tools": not forbidden_tools.intersection(tool_names),
        "arguments": _argument_expectations_match(case, plan),
    }
    confirmation_observed = any(
        preflight_tool_call(
            registry,
            {"tool": step.tool_name, "args": step.args},
        )["risk"]["requires_confirmation"]
        for step in plan
        if step.tool_name in registry
    )
    if expected_confirmation is not None:
        checks["confirmation"] = confirmation_observed is bool(expected_confirmation)

    return EvalCaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category") or "uncategorized"),
        passed=all(checks.values()),
        checks=checks,
        details={
            "question": case.get("question"),
            "planned_tools": tool_names,
            "plan": [
                {
                    "step_id": step.step_id,
                    "goal": step.goal,
                    "tool_name": step.tool_name,
                    "args": step.args,
                    "depends_on": step.depends_on,
                }
                for step in plan
            ],
            "confirmation_observed": confirmation_observed,
        },
    )


def evaluate_evaluator_case(
    case: dict[str, Any],
    evaluator: TaskCompletionEvaluator | None = None,
) -> EvalCaseResult:
    evaluator = evaluator or TaskCompletionEvaluator()
    request = str(case.get("request") or "")
    digest_fields = {
        key: value
        for key, value in dict(case.get("digest") or {}).items()
        if key in ExecutionDigest.__dataclass_fields__ and key != "task"
    }
    digest = ExecutionDigest(task=request, **digest_fields)
    state_payload = dict(case.get("state") or {})
    pending_confirmation = (
        SimpleNamespace(step_id="step_1")
        if state_payload.get("pending_confirmation")
        else None
    )
    plan = [
        SimpleNamespace(status=status)
        for status in state_payload.get("plan_statuses", [])
    ]
    state = SimpleNamespace(
        user_request=request,
        pending_confirmation=pending_confirmation,
        denied_confirmations=list(state_payload.get("denied_confirmations") or []),
        blackboard=state_payload.get("blackboard") or {},
        plan=plan,
    )
    result = evaluator.evaluate(state, digest)
    expected_status = str(case.get("expected_status") or "completed")
    missing_fragments = [str(value) for value in case.get("expected_missing_contains") or []]
    checks = {
        "status": result.status == expected_status,
        "missing_requirements": all(
            any(fragment in requirement for requirement in result.missing_requirements)
            for fragment in missing_fragments
        ),
    }
    return EvalCaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category") or "evaluator"),
        passed=all(checks.values()),
        checks=checks,
        details={
            "request": request,
            "observed_status": result.status,
            "missing_requirements": result.missing_requirements,
        },
    )


def summarize_eval_results(results: list[EvalCaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result.passed for result in results)
    check_names = sorted({name for result in results for name in result.checks})
    check_accuracy = {
        name: (
            sum(result.checks.get(name, False) for result in results if name in result.checks)
            / sum(name in result.checks for result in results)
        )
        for name in check_names
    }
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({result.category for result in results}):
        category_results = [result for result in results if result.category == category]
        category_passed = sum(result.passed for result in category_results)
        categories[category] = {
            "total": len(category_results),
            "passed": category_passed,
            "pass_rate": category_passed / len(category_results),
        }
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "check_accuracy": check_accuracy,
        "categories": categories,
        "failed_case_ids": [result.case_id for result in results if not result.passed],
    }


def _dependencies_are_ordered(plan: list[PlanStep]) -> bool:
    seen: set[str] = set()
    for step in plan:
        if any(dependency not in seen for dependency in step.depends_on):
            return False
        seen.add(step.step_id)
    return True


def _argument_expectations_match(case: dict[str, Any], plan: list[PlanStep]) -> bool:
    for expectation in case.get("arg_expectations") or []:
        tool_name = expectation.get("tool_name")
        args_subset = expectation.get("args_subset") or {}
        candidates = [step.args for step in plan if step.tool_name == tool_name]
        if not any(_is_subset(args_subset, candidate) for candidate in candidates):
            return False
    return True


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _is_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(_is_subset(expected_item, actual_item) for actual_item in actual)
            for expected_item in expected
        )
    return expected == actual
