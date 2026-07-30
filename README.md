# DriveScene Agent

DriveScene Agent is a Plan-and-Execute agent for mining autonomous-driving
trajectory scenarios from Argoverse 2 Motion Forecasting data. It combines
deterministic Python tools with an agent layer that plans retrieval, analysis,
evidence export, and risk-gated file operations.

## What It Does

- Loads Argoverse 2 scenario parquet files and map archives.
- Detects hard braking and close following events with deterministic metrics.
- Builds review queues and visual evidence assets for human labeling.
- Stores reviewed events in a unified parquet index.
- Exposes retrieval, analysis, artifact, and table tools to an agent.
- Uses a Plan-and-Execute flow: plan first, execute one tool step at a time,
  pause for human confirmation before high-risk operations, then report results.

## Agent Architecture

The formal agent is Plan-and-Execute rather than ReAct:

```text
user request
-> planner
-> executor
-> risk gate
-> optional human confirmation
-> tool execution
-> reporter
```

The planner produces structured steps with:

```python
step_id, goal, tool_name, args, depends_on, status
```

Planner output is schema-validated before execution. Invalid JSON, unknown
tools, duplicate step ids, forward dependencies, non-pending status, and
oversized plans are rejected. The LLM receives the validation error and gets
one bounded correction attempt.

The executor only calls tools registered in `ToolRegistry`. File deletion,
overwrite, and in-place table modification are blocked until a user confirms
the pending high-risk operation.

After execution, a task-completion evaluator checks requested deliverables such
as event count, event type, evidence paths, exported files, raw analysis, table
outputs, and deletion. The CLI and Streamlit Plan-and-Execute runtime allow one
bounded replan when observations do not satisfy the request.

`demo_langgraph_react_once.py` is kept as a small learning demo for LangGraph
ReAct-style tool calls. The project entrypoints use the Plan-and-Execute core.

## Run

Use the `agent` environment:

```powershell
& 'D:\app\envs\agent\python.exe' -m pytest -q
```

Build and review scenario evidence:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\summarize_dataset.py
& 'D:\app\envs\agent\python.exe' scripts\run_hard_braking.py
& 'D:\app\envs\agent\python.exe' scripts\run_close_following.py
& 'D:\app\envs\agent\python.exe' scripts\run_scene_labels.py --limit 500
& 'D:\app\envs\agent\python.exe' scripts\build_review_queue.py
& 'D:\app\envs\agent\python.exe' scripts\build_review_assets.py --limit 20 --force
& 'D:\app\envs\agent\python.exe' -m streamlit run scripts\review_app.py
& 'D:\app\envs\agent\python.exe' scripts\compute_review_metrics.py
```

Run the offline Agent evaluations:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py --suite evaluator
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py --suite planner
& 'D:\app\envs\agent\python.exe' scripts\evaluate_agent.py --suite planner --use-heuristic-planner
```

The planner suite contains 30 Chinese tasks covering retrieval, evidence,
analysis, artifact, table, and safety planning. The deterministic evaluator
suite contains 13 completion, cancellation, and replan cases. Results are written under
`outputs/`.

Run the Plan-and-Execute CLI:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\run_agent_cli.py --question "汇总当前事件索引"
& 'D:\app\envs\agent\python.exe' scripts\run_agent_cli.py --question "找 5 个有效急刹案例并给出动画路径"
& 'D:\app\envs\agent\python.exe' scripts\run_agent_cli.py --question "删除 outputs/exports/demo_cases"
```

The CLI uses the model configured in `config/model.yml` by default. For local
debugging without model calls, add `--use-heuristic-planner`. CLI output is
streamed as structured events such as `plan_created`, `step_started`,
`step_completed`, `confirmation_required`, and `final_answer`.

Run the Streamlit agent UI:

```powershell
& 'D:\app\envs\agent\python.exe' -m streamlit run scripts\agent_app.py
```

The Streamlit UI consumes the same event stream and updates the plan/tool-result
panel while steps are running.

Model config:

```yaml
model:
  model: deepseek-v3.2
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  temperature: 0
```

Set the real key in the shell instead of committing it to config:

```powershell
$env:OPENAI_API_KEY = "<your-api-key>"
```

The current loader also accepts the assignment-style `model="...",
base_url="..."` format already used by older local config files. If an
`api_key` value is omitted, the loader reads `OPENAI_API_KEY`.

## Tools

- Retrieval tools query the unified event and scenario indexes.
- Analysis tools dynamically reload raw scenarios and recompute motion metrics.
- Artifact tools create folders, copy evidence files, delete paths, and write manifests.
- Table tools filter rows, drop columns, and convert table formats.

The tool descriptions are Chinese-facing so the model can map Chinese user
requests to deterministic tool calls more reliably.

## Current Review Metrics

Latest reviewed index summary:

```text
num_reviewed: 124
num_valid: 113
precision: 0.9113
close_following valid rate: 0.8977
hard_braking valid rate: 0.9565
cut_in valid rate: 1.0
lane_change valid rate: 0.8333
stopped_vehicle_ahead valid rate: 1.0
```

These metrics show that weak labels plus human review produce a usable evidence
index, while the agent adds value by planning multi-step queries, dynamically
rerunning tools, organizing evidence files, and enforcing risk gates.

## Interview Pitch

This project is not an LLM guessing labels from raw files. The LLM is used for
task understanding and tool orchestration. Event detection, kinematic metrics,
evidence rendering, indexing, and file operations are deterministic Python
tools with tests. The agent layer makes the workflow practical: users can ask
Chinese natural-language questions, the planner decomposes them into tool steps,
the executor retrieves or recomputes evidence, and high-risk operations require
human confirmation before execution.
