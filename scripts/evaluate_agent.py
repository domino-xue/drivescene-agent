from __future__ import annotations

import argparse
import json
from pathlib import Path

from drivescene.eval.agent_eval import (
    EvalCaseResult,
    evaluate_evaluator_case,
    evaluate_planner_case,
    load_jsonl_cases,
    summarize_eval_results,
    validate_planner_case_dataset,
)

try:
    from scripts.run_agent_cli import build_planner, build_registry_from_paths
except ModuleNotFoundError:
    from run_agent_cli import build_planner, build_registry_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveScene Agent offline evaluations.")
    parser.add_argument(
        "--suite",
        choices=["planner", "evaluator", "all"],
        default="all",
    )
    parser.add_argument("--planner-cases", default="evals/agent_planner_cases_200.jsonl")
    parser.add_argument("--evaluator-cases", default="evals/agent_evaluator_cases.jsonl")
    parser.add_argument("--event-index", default="outputs/labeled_event_index.parquet")
    parser.add_argument("--scenario-index", default="outputs/scenario_index.parquet")
    parser.add_argument("--scenario-root", default="data/val")
    parser.add_argument("--model-config", default="config/model.yml")
    parser.add_argument("--use-heuristic-planner", action="store_true")
    parser.add_argument(
        "--planner-prompt-profile",
        choices=["full", "schema_only", "names_only"],
        default="full",
    )
    parser.add_argument("--planner-max-attempts", type=int, default=2)
    parser.add_argument(
        "--splits",
        help="Comma-separated planner splits: development,regression,heldout,adversarial,safety.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--case-ids",
        help="Comma-separated case ids to run, for focused regression.",
    )
    parser.add_argument("--output", default="outputs/agent_eval_results.json")
    args = parser.parse_args()

    all_results = []
    if args.suite in {"planner", "all"}:
        registry = build_registry_from_paths(
            args.event_index,
            args.scenario_index,
            args.scenario_root,
        )
        planner = build_planner(
            registry=registry,
            config_path=args.model_config,
            use_heuristic_planner=args.use_heuristic_planner,
            planner_prompt_profile=args.planner_prompt_profile,
            planner_max_attempts=args.planner_max_attempts,
        )
        planner_cases = load_jsonl_cases(args.planner_cases)
        validate_planner_case_dataset(planner_cases)
        cases = _selected_cases(
            planner_cases,
            args.case_ids,
            args.max_cases,
            args.splits,
        )
        for case in cases:
            try:
                plan = planner.create_plan(str(case["question"]))
                result = evaluate_planner_case(case, plan, registry)
            except Exception as error:  # noqa: BLE001
                result = EvalCaseResult(
                    case_id=str(case["id"]),
                    category=str(case.get("category") or "planner"),
                    passed=False,
                    checks={"planner_execution": False},
                    details={"question": case.get("question"), "error": str(error)},
                    split=str(case.get("split") or "unspecified"),
                    difficulty=str(case.get("difficulty") or "unspecified"),
                )
            all_results.append(result)

    if args.suite in {"evaluator", "all"}:
        cases = _selected_cases(
            load_jsonl_cases(args.evaluator_cases),
            args.case_ids,
            args.max_cases,
            None,
        )
        all_results.extend(evaluate_evaluator_case(case) for case in cases)

    payload = {
        "summary": summarize_eval_results(all_results),
        "results": [result.to_dict() for result in all_results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"results: {output_path}")


def _selected_cases(
    cases: list[dict],
    case_ids: str | None,
    max_cases: int | None,
    splits: str | None = None,
) -> list[dict]:
    if splits:
        selected_splits = {value.strip() for value in splits.split(",") if value.strip()}
        cases = [case for case in cases if str(case.get("split")) in selected_splits]
    if case_ids:
        selected = {value.strip() for value in case_ids.split(",") if value.strip()}
        cases = [case for case in cases if str(case.get("id")) in selected]
    if max_cases is None:
        return cases
    return cases[: max(0, max_cases)]


if __name__ == "__main__":
    main()
