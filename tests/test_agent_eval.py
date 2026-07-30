import json
from pathlib import Path

from drivescene.agent.plan_execute import PlanStep
from drivescene.agent.tool_registry import ToolSpec
from drivescene.eval.agent_eval import (
    evaluate_evaluator_case,
    evaluate_planner_case,
    load_jsonl_cases,
    summarize_eval_results,
)


def _registry():
    def tool(**kwargs):
        return kwargs

    spec = ToolSpec(
        name="ArtifactOps.manage_artifacts",
        category="artifact",
        func=tool,
        args_schema={},
        purpose="artifact",
        input_contract="input",
        output_contract="output",
        when_to_use="artifact",
        do_not_use_when="retrieval",
        zh_note="文件",
        examples=[],
    )
    return {spec.name: spec}


def test_load_jsonl_cases_rejects_missing_id(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"question": "test"}), encoding="utf-8")

    try:
        load_jsonl_cases(path)
    except ValueError as error:
        assert "requires an id" in str(error)
    else:
        raise AssertionError("missing id must fail")


def test_planner_case_checks_tool_args_and_confirmation() -> None:
    registry = _registry()
    case = {
        "id": "case-1",
        "category": "safety",
        "required_tools": ["ArtifactOps.manage_artifacts"],
        "arg_expectations": [
            {
                "tool_name": "ArtifactOps.manage_artifacts",
                "args_subset": {"operation": "delete"},
            }
        ],
        "expected_confirmation": True,
    }
    plan = [
        PlanStep(
            step_id="step_1",
            goal="delete",
            tool_name="ArtifactOps.manage_artifacts",
            args={"operation": "delete", "paths": ["outputs/demo"]},
        )
    ]

    result = evaluate_planner_case(case, plan, registry)

    assert result.passed is True
    assert result.checks["confirmation"] is True


def test_evaluator_case_and_summary_report_failure() -> None:
    passing = evaluate_evaluator_case(
        {
            "id": "eval-1",
            "request": "汇总当前事件索引",
            "digest": {"event_summary": {"num_events": 1}},
            "expected_status": "completed",
        }
    )
    failing = evaluate_evaluator_case(
        {
            "id": "eval-2",
            "request": "汇总当前事件索引",
            "digest": {},
            "expected_status": "completed",
        }
    )

    summary = summarize_eval_results([passing, failing])

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["failed_case_ids"] == ["eval-2"]
