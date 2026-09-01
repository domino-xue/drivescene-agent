from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from drivescene.eval.agent_eval import load_jsonl_cases
from drivescene.eval.runtime_ablation import run_runtime_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic runtime module ablations.")
    parser.add_argument("--evaluator-cases", default="evals/agent_evaluator_cases.jsonl")
    parser.add_argument("--output", default="outputs/runtime_ablation_results.json")
    args = parser.parse_args()

    evaluator_cases = load_jsonl_cases(args.evaluator_cases)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": (
            "Targeted deterministic module ablations. Results measure the constructed failure "
            "modes and must not be interpreted as open-domain model accuracy."
        ),
        "ablations": run_runtime_ablations(evaluator_cases),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["ablations"], ensure_ascii=False, indent=2))
    print(f"results: {output}")


if __name__ == "__main__":
    main()
