# Agent LLM-Native Execution Design

## Goal

DriveScene Agent should use the LLM for intent understanding, planning, review, and result narration, while the execution runtime owns state transformation, tool contracts, argument binding, and safety.

## Problem

The current runtime lets planner output fragile selectors such as `output.items[*].result.*_path`. When the selector points at the wrong step, or at a step that does not produce evidence paths, execution fails. The failure handler also appends a fake completed `replan_step_*` with `tool_name=None`, so diagnostics become plan steps and can be referenced by later LLM replans.

## Architecture

The runtime will maintain a typed blackboard derived from successful tool outputs. The blackboard contains durable semantic slots such as selected events, review ids, evidence assets, evidence paths, export directories, manifests, copied files, and the top event.

The planner may still choose tools and high-level operations, but tool arguments are resolved through contract-aware binding before execution. Replan prompts expose the typed blackboard and allowed references, not raw diagnostic text as reusable data.

## Boundaries

The LLM is responsible for:
- interpreting user intent and deliverables;
- deciding which tool class and operation are appropriate;
- reviewing whether the observations satisfy the user request;
- generating user-facing conclusions from the digest.

The deterministic runtime is responsible for:
- maintaining typed state;
- resolving references;
- validating tool arguments;
- binding missing arguments from typed state when safe;
- preserving batch item-level errors;
- recording failures as diagnostics, not fake completed steps.

## First Phase

This phase does not replace every planner prompt or introduce a new external dependency. It adds:
- `PlanExecuteState.blackboard`;
- `PlanExecuteState.diagnostics`;
- `$from_state` structured references;
- blackboard updates after successful tool calls;
- contract-aware defaults for evidence lookup and artifact copy;
- clean failure diagnostics instead of `replan_step_*`;
- replan prompt context based on typed state.

## Success Criteria

- A failed tool call does not create a completed no-tool plan step.
- Replan prompts cannot treat diagnostic notes as data-producing steps.
- `ArtifactOps.manage_artifacts(operation="copy")` can use typed `evidence_paths` instead of fragile selectors.
- Missing evidence paths are reported as structured argument errors.
- Batch wildcard selection skips items without matching fields but fails if no values exist.
- Existing tests continue to pass.
