from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from drivescene.agent.model_config import load_model_config
from drivescene.agent.plan_execute import HeuristicPlanner, LLMJsonPlanner
from drivescene.eval.agent_eval import (
    EvalCaseResult,
    evaluate_planner_case,
    load_jsonl_cases,
    summarize_eval_results,
    validate_planner_case_dataset,
)

try:
    from scripts.run_agent_cli import build_registry_from_paths
except ModuleNotFoundError:
    from run_agent_cli import build_registry_from_paths


VARIANTS: dict[str, dict[str, Any]] = {
    "heuristic": {"planner": "heuristic"},
    "full": {"prompt_profile": "full", "max_attempts": 2},
    "no_retry": {"prompt_profile": "full", "max_attempts": 1},
    "schema_only": {"prompt_profile": "schema_only", "max_attempts": 2},
    "names_only": {"prompt_profile": "names_only", "max_attempts": 2},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stratified planner ablations.")
    parser.add_argument("--cases", default="evals/agent_planner_cases_200.jsonl")
    parser.add_argument(
        "--variants",
        default="full,no_retry,schema_only,names_only,heuristic",
        help=f"Comma-separated variants: {','.join(VARIANTS)}",
    )
    parser.add_argument(
        "--splits",
        default="heldout,adversarial,safety",
        help="Primary evaluation splits; development and regression are excluded by default.",
    )
    parser.add_argument("--event-index", default="outputs/labeled_event_index.parquet")
    parser.add_argument("--scenario-index", default="outputs/scenario_index.parquet")
    parser.add_argument("--scenario-root", default="data/val")
    parser.add_argument("--model-config", default="config/model.yml")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent planner requests per variant (default: 1).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180,
        help="Per-request model timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--transport-retries",
        type=int,
        default=1,
        help="HTTP transport retries per model request (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed variants from an existing output file.",
    )
    parser.add_argument("--output", default="outputs/planner_ablation_results.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.transport_retries < 0:
        parser.error("--transport-retries must be non-negative")

    case_path = Path(args.cases)
    cases = load_jsonl_cases(case_path)
    validate_planner_case_dataset(cases, expected_total=200)
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    cases = [case for case in cases if str(case.get("split")) in selected_splits]
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]

    variant_names = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        parser.error(f"Unknown variants: {', '.join(unknown)}")

    registry = build_registry_from_paths(
        args.event_index,
        args.scenario_index,
        args.scenario_root,
    )
    model_config = load_model_config(args.model_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    variant_payloads = (
        _load_resume_variants(output, case_path, model_config)
        if args.resume and output.exists()
        else {}
    )
    experiment_started = time.perf_counter()
    for variant_name in variant_names:
        if variant_name in variant_payloads:
            print(f"{variant_name}: already complete; skipping", flush=True)
            continue
        planner = _build_variant(
            variant_name,
            registry,
            model_config,
            request_timeout=args.request_timeout,
            transport_retries=args.transport_retries,
        )
        variant_started = time.perf_counter()
        results = _evaluate_cases(
            cases,
            planner,
            registry,
            workers=args.workers,
            variant_name=variant_name,
        )
        elapsed_seconds = time.perf_counter() - variant_started
        variant_payloads[variant_name] = {
            "configuration": VARIANTS[variant_name],
            "summary": summarize_eval_results(results),
            "results": [result.to_dict() for result in results],
            "elapsed_seconds": elapsed_seconds,
            "throughput_cases_per_minute": (
                60 * len(results) / elapsed_seconds if elapsed_seconds else None
            ),
        }
        summary = variant_payloads[variant_name]["summary"]
        print(
            f"{variant_name}: {summary['passed']}/{summary['total']} "
            f"({summary['pass_rate']:.1%}) in {elapsed_seconds:.1f}s",
            flush=True,
        )
        _write_payload(
            output,
            case_path,
            cases,
            selected_splits,
            model_config,
            variant_names,
            variant_payloads,
            workers=args.workers,
            request_timeout=args.request_timeout,
            transport_retries=args.transport_retries,
            elapsed_seconds=time.perf_counter() - experiment_started,
        )

    print(f"results: {output}")


def _write_payload(
    output: Path,
    case_path: Path,
    cases: list[dict[str, Any]],
    selected_splits: set[str],
    model_config: Any,
    variant_names: list[str],
    variant_payloads: dict[str, Any],
    *,
    workers: int,
    request_timeout: float,
    transport_retries: int,
    elapsed_seconds: float,
) -> None:
    payload = {
        "experiment": {
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "dataset_path": str(case_path),
            "dataset_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "dataset_total": 200,
            "evaluated_cases": len(cases),
            "splits": sorted(selected_splits),
            "model": model_config.model,
            "temperature": model_config.temperature,
            "variants": variant_names,
            "completed_variants": list(variant_payloads),
            "workers": workers,
            "request_timeout": request_timeout,
            "transport_retries": transport_retries,
            "elapsed_seconds": elapsed_seconds,
            "metric_definition": (
                "Whole-case pass requires step count, known tools, ordered dependencies, "
                "required/forbidden tools, argument subsets, and confirmation behavior to pass."
            ),
        },
        "variants": variant_payloads,
        "comparisons_to_full": _comparisons_to_full(variant_payloads),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_variant(
    variant_name: str,
    registry: Any,
    model_config: Any,
    *,
    request_timeout: float,
    transport_retries: int,
) -> Any:
    configuration = VARIANTS[variant_name]
    if configuration.get("planner") == "heuristic":
        return HeuristicPlanner()
    model = ChatOpenAI(
        **model_config.to_chat_openai_kwargs(),
        request_timeout=request_timeout,
        max_retries=transport_retries,
    )
    return LLMJsonPlanner(
        model,
        registry,
        prompt_profile=str(configuration["prompt_profile"]),
        max_attempts=int(configuration["max_attempts"]),
    )


def _load_resume_variants(
    output: Path,
    case_path: Path,
    model_config: Any,
) -> dict[str, Any]:
    payload = json.loads(output.read_text(encoding="utf-8"))
    experiment = payload.get("experiment", {})
    expected_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    if experiment.get("dataset_sha256") != expected_sha:
        raise ValueError("Cannot resume: dataset hash does not match existing output")
    if experiment.get("model") != model_config.model:
        raise ValueError("Cannot resume: model does not match existing output")
    variants = payload.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Cannot resume: existing output has no variants mapping")
    return variants


def _evaluate_case(case: dict[str, Any], planner: Any, registry: Any) -> EvalCaseResult:
    try:
        plan = planner.create_plan(str(case["question"]))
        return evaluate_planner_case(case, plan, registry)
    except Exception as error:  # noqa: BLE001
        return EvalCaseResult(
            case_id=str(case["id"]),
            category=str(case.get("category") or "planner"),
            passed=False,
            checks={"planner_execution": False},
            details={"question": case.get("question"), "error": str(error)},
            split=str(case.get("split") or "unspecified"),
            difficulty=str(case.get("difficulty") or "unspecified"),
        )


def _evaluate_cases(
    cases: list[dict[str, Any]],
    planner: Any,
    registry: Any,
    *,
    workers: int,
    variant_name: str,
) -> list[EvalCaseResult]:
    if workers == 1 or len(cases) < 2:
        return [_evaluate_case(case, planner, registry) for case in cases]

    ordered_results: list[EvalCaseResult | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_case, case, planner, registry): index
            for index, case in enumerate(cases)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ordered_results[futures[future]] = future.result()
            if completed % 10 == 0 or completed == len(cases):
                print(
                    f"{variant_name}: {completed}/{len(cases)} cases complete",
                    flush=True,
                )
    return [result for result in ordered_results if result is not None]


def _comparisons_to_full(variants: dict[str, Any]) -> dict[str, Any]:
    if "full" not in variants:
        return {}
    full = variants["full"]["summary"]
    comparisons: dict[str, Any] = {}
    for name, payload in variants.items():
        if name == "full":
            continue
        summary = payload["summary"]
        check_names = sorted(
            set(full.get("check_accuracy", {})).union(summary.get("check_accuracy", {}))
        )
        comparisons[name] = {
            "whole_case_delta_pp": 100 * (full["pass_rate"] - summary["pass_rate"]),
            "check_delta_pp": {
                check: 100
                * (
                    full.get("check_accuracy", {}).get(check, 0.0)
                    - summary.get("check_accuracy", {}).get(check, 0.0)
                )
                for check in check_names
            },
        }
    return comparisons


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
