# Agent Evaluation

DriveScene Agent uses two complementary offline suites.

## Planner suite

`evals/agent_planner_cases.jsonl` contains 30 Chinese planning tasks across:

- indexed retrieval;
- event evidence lookup;
- raw scenario analysis;
- artifact creation, copy, manifest, and deletion;
- table transformations;
- high-risk confirmation.

Each case declares required and forbidden tools, recursive argument subsets,
maximum plan length, and whether the plan should contain a confirmation-gated
operation. The evaluator reports whole-case pass rate plus check-level and
category-level accuracy.

Run the configured LLM planner:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py --suite planner
```

Run the deterministic fallback baseline:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py `
  --suite planner `
  --use-heuristic-planner `
  --output outputs/agent_planner_heuristic_results.json
```

The current heuristic baseline passes 6/30 cases (20%). This is expected: the
fallback only covers a small diagnostic subset and is not the production UI
planner.

## Evaluator suite

`evals/agent_evaluator_cases.jsonl` contains deterministic task-completion
cases for summaries, event counts, evidence, export, raw analysis, table
outputs, and confirmation pauses.

```powershell
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py `
  --suite evaluator `
  --output outputs/agent_evaluator_eval_results.json
```

The current task-completion evaluator passes 13/13 cases. The configured LLM
planner passes 30/30 cases at temperature 0; the heuristic baseline passes
6/30 cases.

## Result interpretation

Planner pass rate measures whether the proposed plan has the required tools,
arguments, dependency order, bounded length, and risk behavior. It does not by
itself prove end-to-end task completion.

Evaluator cases verify deterministic completion judgments. End-to-end replan
behavior is additionally covered by tests where an incomplete first plan is
executed, evaluated, supplemented, and executed again.

When comparing model versions, preserve the case files and report:

- planner whole-case pass rate;
- argument accuracy;
- required-tool accuracy;
- confirmation accuracy;
- category-level pass rates;
- model name and temperature.
