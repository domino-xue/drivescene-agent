import hashlib
import json
from pathlib import Path

from drivescene.agent.plan_execute import PlanStep
from drivescene.agent.tool_registry import ToolSpec
from drivescene.eval.agent_eval import (
    evaluate_evaluator_case,
    evaluate_planner_case,
    load_jsonl_cases,
    summarize_eval_results,
    validate_planner_case_dataset,
)
from scripts.build_planner_eval_dataset import SPLIT_CATEGORY_COUNTS, build_cases


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
    assert summary["pass_rate_ci95"][0] < 0.5 < summary["pass_rate_ci95"][1]
    assert summary["failed_case_ids"] == ["eval-2"]


def test_stratified_planner_dataset_contains_200_valid_cases() -> None:
    cases = build_cases()

    validate_planner_case_dataset(cases, expected_total=200)

    assert len(cases) == 200
    assert {
        split: sum(case["split"] == split for case in cases)
        for split in SPLIT_CATEGORY_COUNTS
    } == {
        split: sum(categories.values())
        for split, categories in SPLIT_CATEGORY_COUNTS.items()
    }
    assert len({case["question"] for case in cases}) == 200


def test_committed_planner_dataset_matches_manifest_hash() -> None:
    dataset = Path("evals/agent_planner_cases_200.jsonl")
    manifest = json.loads(
        Path("evals/agent_planner_cases_200.manifest.json").read_text(encoding="utf-8")
    )

    assert len(load_jsonl_cases(dataset)) == 200
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["sha256"]


def test_dataset_validation_rejects_template_family_leakage() -> None:
    cases = build_cases()[:2]
    cases[1] = {**cases[1], "split": "heldout", "template_family": cases[0]["template_family"]}

    try:
        validate_planner_case_dataset(cases)
    except ValueError as error:
        assert "Template-family leakage" in str(error)
    else:
        raise AssertionError("cross-split template leakage must fail")
