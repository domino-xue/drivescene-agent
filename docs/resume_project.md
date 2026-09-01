# DriveScene Agent - resume-ready project description

## Project entry

**DriveScene Agent | Autonomous-driving trajectory analysis agent**
Python, LangGraph, LangChain, DeepSeek V4, Pandas, PyArrow, Pydantic, Pytest

- Built a Plan-and-Execute agent for Argoverse 2 trajectory-event mining, separating LLM
  planning from deterministic retrieval, evidence generation, trajectory analysis, table
  transformation, and artifact operations through a typed tool registry and blackboard.
- Designed a reproducible 200-case planner benchmark with development, regression,
  heldout, adversarial, and safety splits. On the 120 primary heldout/adversarial/safety
  cases, the complete planner achieved **104/120 whole-case passes (86.7%, Wilson 95% CI
  79.4%-91.6%)**, compared with **9/120 (7.5%)** for the deterministic heuristic baseline.
- Ran controlled ablations on the same 120 cases: retaining only tool schemas reduced
  whole-case pass rate from **86.7% to 21.7% (-65.0 pp)**, while tool names alone scored
  0%. Added bounded replanning, typed argument binding, completion evaluation, and a
  human-confirmation risk gate; targeted module tests showed drops of 100.0 pp, 38.5 pp,
  and 50.0 pp respectively when those runtime modules were removed.

## Shorter version

Built a LangGraph-based autonomous-driving data agent with typed tool contracts,
blackboard argument binding, completion evaluation, and destructive-operation confirmation.
Designed a 200-case stratified benchmark; achieved 86.7% whole-case pass rate on 120
heldout/adversarial/safety tasks (95% CI 79.4%-91.6%), versus 7.5% for a heuristic baseline,
and demonstrated a 65.0 pp drop when rich tool contracts were ablated to schemas only.

## Interview-safe interpretation

- Say **"whole-case pass rate on a finite contract benchmark"**, not generic model
  accuracy. A case passes only if tool choice, arguments, dependency order, step count, and
  confirmation behavior all pass.
- The 86.7% headline excludes the 80 development/regression cases. The evaluated population
  is explicitly 60 heldout + 40 adversarial + 20 safety cases.
- The rich-contract ablation is strong paired evidence: the complete prompt passed 78 cases
  that `schema_only` failed, with no reverse wins (exact McNemar p<1e-23).
- Do not claim the retry mechanism produced a significant improvement. It changed 85.0% to
  86.7%, but the paired exact test gives p=0.6875 in this single run.
- Volunteer the main weakness if asked: table tasks passed only 6/13 (46.2%), and argument
  mismatches caused 15 of the 16 complete-planner failures. This is the next optimization
  target.
- The 100% runtime-module numbers are targeted constructed cases, not open-domain accuracy;
  they demonstrate that the modules enforce their intended mechanisms.

## Reproduction anchors

- Dataset: `evals/agent_planner_cases_200.jsonl`
- Dataset SHA-256:
  `9629d1c02fd189452fcff0dd3690a887653fd4aa729d3a480cdcf41d875ac1d5`
- Compact result: `evals/planner_ablation_deepseek_v4_flash_summary.json`
- Full local result: `outputs/planner_ablation_results_deepseek_v4_flash.json`
- Code revision recorded at run time: `7e57787943cbd5db6a2400d2bd4673480491f5f7`
