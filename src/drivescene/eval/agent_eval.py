from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    split: str = "unspecified"
    difficulty: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "checks": self.checks,
            "details": self.details,
            "split": self.split,
            "difficulty": self.difficulty,
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


def validate_planner_case_dataset(
    cases: list[dict[str, Any]],
    *,
    expected_total: int | None = None,
) -> None:
    """Validate dataset identity, split hygiene, and planner expectations."""
    if expected_total is not None and len(cases) != expected_total:
        raise ValueError(f"Expected {expected_total} planner cases, found {len(cases)}")
    if not cases:
        raise ValueError("Planner case dataset must not be empty")

    ids = [str(case.get("id") or "") for case in cases]
    questions = [str(case.get("question") or "").strip() for case in cases]
    if any(not case_id for case_id in ids):
        raise ValueError("Every planner case requires a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("Planner case ids must be unique")
    if any(not question for question in questions):
        raise ValueError("Every planner case requires a non-empty question")
    if len(questions) != len(set(questions)):
        raise ValueError("Planner case questions must be unique")

    allowed_splits = {"development", "regression", "heldout", "adversarial", "safety"}
    allowed_difficulties = {"easy", "medium", "hard"}
    families_by_split: dict[str, set[str]] = {}
    for case in cases:
        split = str(case.get("split") or "")
        if split not in allowed_splits:
            raise ValueError(f"Unknown planner split for {case['id']}: {split}")
        difficulty = str(case.get("difficulty") or "")
        if difficulty not in allowed_difficulties:
            raise ValueError(f"Unknown difficulty for {case['id']}: {difficulty}")
        family = str(case.get("template_family") or "")
        if not family:
            raise ValueError(f"Planner case {case['id']} requires template_family")
        families_by_split.setdefault(split, set()).add(family)
        if not case.get("required_tools"):
            raise ValueError(f"Planner case {case['id']} requires at least one tool")
        if not case.get("arg_expectations"):
            raise ValueError(f"Planner case {case['id']} requires argument expectations")
        if not isinstance(case.get("expected_confirmation"), bool):
            raise TypeError(f"Planner case {case['id']} requires boolean confirmation label")

    split_names = sorted(families_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = families_by_split[left].intersection(families_by_split[right])
            if overlap:
                raise ValueError(
                    f"Template-family leakage between {left} and {right}: {sorted(overlap)}"
                )


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
        split=str(case.get("split") or "unspecified"),
        difficulty=str(case.get("difficulty") or "unspecified"),
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
        split=str(case.get("split") or "unspecified"),
        difficulty=str(case.get("difficulty") or "unspecified"),
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
    categories = _group_summary(results, "category")
    splits = _group_summary(results, "split")
    difficulties = _group_summary(results, "difficulty")
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "pass_rate_ci95": _wilson_interval(passed, total),
        "check_accuracy": check_accuracy,
        "categories": categories,
        "splits": splits,
        "difficulties": difficulties,
        "failed_case_ids": [result.case_id for result in results if not result.passed],
    }


def _group_summary(
    results: list[EvalCaseResult],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    values = sorted({str(getattr(result, attribute)) for result in results})
    for value in values:
        selected = [result for result in results if str(getattr(result, attribute)) == value]
        passed = sum(result.passed for result in selected)
        grouped[value] = {
            "total": len(selected),
            "passed": passed,
            "pass_rate": passed / len(selected),
            "pass_rate_ci95": _wilson_interval(passed, len(selected)),
        }
    return grouped


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


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
