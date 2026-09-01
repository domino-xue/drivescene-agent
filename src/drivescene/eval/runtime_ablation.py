from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drivescene.agent.evaluator import TaskCompletionEvaluator
from drivescene.agent.plan_execute import (
    PlanExecuteState,
    bind_tool_arguments,
    validate_tool_arguments,
)
from drivescene.agent.tool_registry import ToolSpec, preflight_tool_call
from drivescene.eval.agent_eval import evaluate_evaluator_case


@dataclass(frozen=True)
class BinaryMetrics:
    total: int
    correct: int
    accuracy: float
    true_positive_rate: float | None = None
    true_negative_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "true_positive_rate": self.true_positive_rate,
            "true_negative_rate": self.true_negative_rate,
        }


def run_runtime_ablations(evaluator_cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "typed_blackboard_binding": _typed_blackboard_ablation(),
        "completion_evaluator": _completion_evaluator_ablation(evaluator_cases),
        "risk_gate": _risk_gate_ablation(),
    }


def _typed_blackboard_ablation() -> dict[str, Any]:
    cases = _typed_binding_cases()
    full_results: list[bool] = []
    ablated_results: list[bool] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        state = PlanExecuteState(
            user_request="runtime ablation",
            plan=[],
            blackboard=dict(case["blackboard"]),
        )
        full_args = bind_tool_arguments(case["tool_name"], dict(case["args"]), state)
        full_ok = _arguments_are_valid(case["tool_name"], full_args)
        ablated_ok = _arguments_are_valid(case["tool_name"], dict(case["args"]))
        full_results.append(full_ok)
        ablated_results.append(ablated_ok)
        case_results.append(
            {
                "id": case["id"],
                "tool_name": case["tool_name"],
                "full_passed": full_ok,
                "without_typed_binding_passed": ablated_ok,
            }
        )
    return {
        "metric": "valid downstream tool arguments after a missing planner handoff",
        "full": _rate(full_results),
        "without_module": _rate(ablated_results),
        "delta_pp": 100 * (_rate(full_results)["rate"] - _rate(ablated_results)["rate"]),
        "cases": case_results,
    }


def _completion_evaluator_ablation(evaluator_cases: list[dict[str, Any]]) -> dict[str, Any]:
    full_results: list[bool] = []
    ablated_results: list[bool] = []
    case_results: list[dict[str, Any]] = []
    evaluator = TaskCompletionEvaluator()
    for case in evaluator_cases:
        full = evaluate_evaluator_case(case, evaluator=evaluator)
        naive_status = _naive_completion_status(case)
        expected_status = str(case.get("expected_status") or "completed")
        naive_ok = naive_status == expected_status
        full_results.append(full.passed)
        ablated_results.append(naive_ok)
        case_results.append(
            {
                "id": str(case["id"]),
                "expected_status": expected_status,
                "full_status": full.details["observed_status"],
                "without_evaluator_status": naive_status,
                "full_passed": full.passed,
                "without_evaluator_passed": naive_ok,
            }
        )
    full_rate = _rate(full_results)
    ablated_rate = _rate(ablated_results)
    return {
        "metric": "correct task-completion status",
        "full": full_rate,
        "without_module": ablated_rate,
        "delta_pp": 100 * (full_rate["rate"] - ablated_rate["rate"]),
        "cases": case_results,
    }


def _risk_gate_ablation() -> dict[str, Any]:
    registry = _risk_registry()
    cases = _risk_cases()
    expected = [bool(case["requires_confirmation"]) for case in cases]
    full_predictions = [
        bool(
            preflight_tool_call(
                registry,
                {"tool": case["tool_name"], "args": case["args"]},
            )["risk"]["requires_confirmation"]
        )
        for case in cases
    ]
    without_gate_predictions = [False for _ in cases]
    return {
        "metric": "confirmation classification for destructive and safe operations",
        "full": _classification_metrics(expected, full_predictions).to_dict(),
        "without_module": _classification_metrics(expected, without_gate_predictions).to_dict(),
        "delta_accuracy_pp": 100
        * (
            _classification_metrics(expected, full_predictions).accuracy
            - _classification_metrics(expected, without_gate_predictions).accuracy
        ),
        "cases": [
            {
                "id": case["id"],
                "expected_confirmation": label,
                "full_prediction": full,
                "without_gate_prediction": ablated,
            }
            for case, label, full, ablated in zip(
                cases,
                expected,
                full_predictions,
                without_gate_predictions,
            )
        ],
    }


def _typed_binding_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(1, 6):
        cases.append(
            {
                "id": f"evidence-detail-{index}",
                "tool_name": "EvidenceStore.query_index",
                "args": {"target": "events", "operation": "detail"},
                "blackboard": {"review_ids": [f"{index:06d}"]},
            }
        )
        cases.append(
            {
                "id": f"evidence-bundle-{index}",
                "tool_name": "EvidenceStore.query_index",
                "args": {"target": "events", "operation": "evidence"},
                "blackboard": {"review_ids": [f"{index:06d}"]},
            }
        )
        cases.append(
            {
                "id": f"copy-paths-{index}",
                "tool_name": "ArtifactOps.manage_artifacts",
                "args": {
                    "operation": "copy",
                    "output_dir": f"outputs/exports/binding-{index}",
                },
                "blackboard": {
                    "evidence_paths": [f"outputs/review_assets/{index:06d}/animation.gif"]
                },
            }
        )
        cases.append(
            {
                "id": f"copy-output-{index}",
                "tool_name": "ArtifactOps.manage_artifacts",
                "args": {
                    "operation": "copy",
                    "paths": [f"outputs/review_assets/{index:06d}/metrics.png"],
                },
                "blackboard": {"export_dirs": [f"outputs/exports/binding-{index}"]},
            }
        )
    return cases


def _arguments_are_valid(tool_name: str, args: dict[str, Any]) -> bool:
    try:
        validate_tool_arguments(tool_name, args)
    except ValueError:
        return False
    return True


def _naive_completion_status(case: dict[str, Any]) -> str:
    state = dict(case.get("state") or {})
    if state.get("pending_confirmation"):
        return "needs_confirmation"
    if state.get("denied_confirmations"):
        return "cancelled"
    return "completed"


def _risk_registry() -> dict[str, ToolSpec]:
    def no_op(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    names = [
        ("EvidenceStore.query_index", "retrieval"),
        ("AnalysisTools.run_analysis", "analysis"),
        ("ArtifactOps.manage_artifacts", "artifact"),
        ("TableOps.transform_table", "table"),
    ]
    return {
        name: ToolSpec(
            name=name,
            category=category,
            func=no_op,
            args_schema={},
            purpose="runtime ablation",
            input_contract="runtime ablation",
            output_contract="runtime ablation",
            when_to_use="runtime ablation",
            do_not_use_when="runtime ablation",
            zh_note="消融实验",
            examples=[],
        )
        for name, category in names
    }


def _risk_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(1, 5):
        cases.extend(
            [
                {
                    "id": f"delete-{index}",
                    "tool_name": "ArtifactOps.manage_artifacts",
                    "args": {"operation": "delete", "paths": [f"outputs/risk-{index}"]},
                    "requires_confirmation": True,
                },
                {
                    "id": f"copy-overwrite-{index}",
                    "tool_name": "ArtifactOps.manage_artifacts",
                    "args": {
                        "operation": "copy",
                        "paths": [f"source-{index}"],
                        "output_dir": f"outputs/risk-{index}",
                        "overwrite": True,
                    },
                    "requires_confirmation": True,
                },
                {
                    "id": f"table-in-place-{index}",
                    "tool_name": "TableOps.transform_table",
                    "args": {
                        "operation": "drop_columns",
                        "input_path": f"outputs/table-{index}.parquet",
                        "output_path": f"outputs/table-{index}.parquet",
                    },
                    "requires_confirmation": True,
                },
                {
                    "id": f"query-{index}",
                    "tool_name": "EvidenceStore.query_index",
                    "args": {"target": "events", "operation": "summary"},
                    "requires_confirmation": False,
                },
                {
                    "id": f"create-folder-{index}",
                    "tool_name": "ArtifactOps.manage_artifacts",
                    "args": {"operation": "create_folder", "output_dir": f"outputs/new-{index}"},
                    "requires_confirmation": False,
                },
                {
                    "id": f"derived-table-{index}",
                    "tool_name": "TableOps.transform_table",
                    "args": {
                        "operation": "convert_format",
                        "input_path": f"outputs/source-{index}.parquet",
                        "output_path": f"outputs/derived-{index}.csv",
                    },
                    "requires_confirmation": False,
                },
            ]
        )
    return cases


def _classification_metrics(
    expected: list[bool],
    predicted: list[bool],
) -> BinaryMetrics:
    correct = sum(left == right for left, right in zip(expected, predicted))
    positives = sum(expected)
    negatives = len(expected) - positives
    true_positives = sum(left and right for left, right in zip(expected, predicted))
    true_negatives = sum(not left and not right for left, right in zip(expected, predicted))
    return BinaryMetrics(
        total=len(expected),
        correct=correct,
        accuracy=correct / len(expected) if expected else 0.0,
        true_positive_rate=true_positives / positives if positives else None,
        true_negative_rate=true_negatives / negatives if negatives else None,
    )


def _rate(values: list[bool]) -> dict[str, Any]:
    passed = sum(values)
    return {
        "total": len(values),
        "passed": passed,
        "rate": passed / len(values) if values else 0.0,
    }
