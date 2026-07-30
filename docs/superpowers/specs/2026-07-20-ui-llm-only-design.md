# UI LLM-Only Planner Design

## Goal

Prevent end users from routing natural-language tasks through the rule-based fallback planner. The Streamlit UI must always construct the Plan-and-Execute agent with the configured LLM planner.

## Scope

- Remove the rule fallback checkbox from `scripts/agent_app.py`.
- Pass `use_heuristic_planner=False` from the UI agent factory path.
- Keep `--use-heuristic-planner` in `scripts/run_agent_cli.py` for explicit local development and diagnosis.
- Add app tests proving the UI no longer exposes or selects heuristic mode.
- Rerun the previously blocked real-model evaluation cases after the change.

## Non-goals

- Do not delete or rewrite `HeuristicPlanner`.
- Do not change the tool registry, risk policy, state binding, or memory behavior.
- Do not add automatic fallback after model-service failure; UI reports the LLM service error instead of silently changing planners.

## Behavior

The UI creates its cache key and agent with `use_heuristic_planner=False`. If an old session contains an agent built in heuristic mode, the changed cache key forces an agent rebuild on the next UI execution.

CLI users may still opt into the deterministic planner only through the explicit `--use-heuristic-planner` argument.

## Verification

- App tests assert the fallback control is absent and the UI agent mode is fixed to LLM.
- Existing CLI tests confirm the explicit diagnostic fallback remains available.
- Full regression test suite passes.
- Real LLM evaluation verifies index summary, hard-braking evidence lookup, batch evidence copy, and delete confirmation.
