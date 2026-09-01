from drivescene.eval.agent_eval import load_jsonl_cases
from drivescene.eval.runtime_ablation import run_runtime_ablations


def test_runtime_ablation_quantifies_targeted_module_effects() -> None:
    payload = run_runtime_ablations(load_jsonl_cases("evals/agent_evaluator_cases.jsonl"))

    binding = payload["typed_blackboard_binding"]
    assert binding["full"] == {"total": 20, "passed": 20, "rate": 1.0}
    assert binding["without_module"] == {"total": 20, "passed": 0, "rate": 0.0}

    evaluator = payload["completion_evaluator"]
    assert evaluator["full"]["passed"] == 13
    assert evaluator["without_module"]["passed"] == 8

    risk = payload["risk_gate"]
    assert risk["full"]["accuracy"] == 1.0
    assert risk["without_module"]["accuracy"] == 0.5
    assert risk["without_module"]["true_positive_rate"] == 0.0
