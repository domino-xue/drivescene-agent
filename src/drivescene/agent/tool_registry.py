from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.risk import assess_tool_risk
from drivescene.ops.tables import TableOps


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    func: Callable[..., Any]
    args_schema: dict[str, str]
    purpose: str
    input_contract: str
    output_contract: str
    when_to_use: str
    do_not_use_when: str
    zh_note: str
    examples: list[dict[str, Any]]
    requires_risk_preflight: bool = True

    @property
    def description(self) -> str:
        return render_tool_description(self)


ToolRegistry = dict[str, ToolSpec]


def build_tool_registry(
    evidence_store: EvidenceStore,
    analysis_tools: AnalysisTools,
    artifact_ops: ArtifactOps,
    table_ops: TableOps,
) -> ToolRegistry:
    specs = [
        ToolSpec(
            name="EvidenceStore.query_index",
            category="retrieval",
            func=evidence_store.query_index,
            args_schema={
                "target": "str",
                "operation": "str",
                "filters": "dict[str, Any] | None",
                "review_ids": "list[str] | None",
                "sort_by": "str | None",
                "ascending": "bool",
                "limit": "int | None",
            },
            purpose="Query prebuilt event or scenario indexes.",
            input_contract=(
                "target must be 'events' or 'scenarios'. For events, operation is "
                "'search', 'summary', 'detail', or 'evidence'. Use summary, not search, when the "
                "user asks for a count or statistics. Event filters use real index columns such "
                "as event_type, is_valid_event, scenario_id, city, severity, or review_status. "
                "For scenarios, operation is 'search' and the only filter keys are city, "
                "object_type, and has_map; city values are lowercase. sort_by must be a real "
                "event-index column, for example min_velocity_acceleration_mps2 for strongest "
                "hard braking or min_front_distance_m for closest following."
            ),
            output_contract=(
                "Returns a state-transform envelope: ok, operation, summary.total/succeeded/failed, "
                "and items. Each item has input, ok, result, error, and optional error_type."
            ),
            when_to_use=(
                "Use for indexed retrieval: event search, event summaries, event details, "
                "evidence asset paths, or scenario metadata search."
            ),
            do_not_use_when=(
                "Do not use to recompute detectors from raw scenario parquet files; use "
                "AnalysisTools.run_analysis for raw data analysis."
            ),
            zh_note="中文备注：查询已经构建好的事件/场景索引；急刹和近距离跟车都通过 filters.event_type 参数化。",
            examples=[
                {
                    "target": "events",
                    "operation": "search",
                    "filters": {"event_type": "hard_braking", "is_valid_event": True},
                    "sort_by": "min_velocity_acceleration_mps2",
                    "ascending": True,
                    "limit": 5,
                },
                {
                    "target": "events",
                    "operation": "summary",
                    "filters": {"event_type": "close_following", "is_valid_event": True},
                },
                {
                    "target": "scenarios",
                    "operation": "search",
                    "filters": {"city": "austin", "object_type": "vehicle"},
                },
                {"target": "events", "operation": "evidence", "review_ids": ["000001"]},
                {
                    "target": "events",
                    "operation": "evidence",
                    "review_ids": {"$from_state": "review_ids"},
                },
            ],
        ),
        ToolSpec(
            name="AnalysisTools.run_analysis",
            category="analysis",
            func=analysis_tools.run_analysis,
            args_schema={
                "operation": "str",
                "event_type": "str | None",
                "scenario_ids": "list[str] | None",
                "scenario_paths": "list[str | Path] | None",
                "scenario_glob": "str | None",
                "track_id": "str | int | None",
                "actor_id": "str | int | None",
                "window": "dict[str, int] | None",
                "parameters": "dict[str, Any] | None",
            },
            purpose="Run deterministic analysis on one or more raw scenario parquet files.",
            input_contract=(
                "Provide exactly one input source: scenario_ids, scenario_paths, or scenario_glob. "
                "operation is 'detect_events', 'compute_track_kinematics', 'compute_relative_motion', "
                "or 'compare_velocity_position_evidence'. event_type is required for detect_events "
                "and can be hard_braking, close_following, cut_in, stopped_vehicle_ahead, or "
                "lane_change. compare_velocity_position_evidence requires track_id and window "
                "with the exact keys start_timestep and end_timestep."
            ),
            output_contract=(
                "Returns ok, operation, summary.total/succeeded/failed, and items. Each item has "
                "input, ok, and either result or error_type/error."
            ),
            when_to_use=(
                "Use when the user asks to rerun detectors, inspect raw kinematics, compute relative "
                "motion, or batch-process scenario files."
            ),
            do_not_use_when=(
                "Do not use for already-indexed event search or evidence path lookup; use "
                "EvidenceStore.query_index."
            ),
            zh_note="中文备注：批量原始场景分析入口；通过 event_type 参数选择 hard_braking 或 close_following。",
            examples=[
                {
                    "operation": "detect_events",
                    "event_type": "cut_in",
                    "scenario_ids": ["scene-1", "scene-2"],
                },
                {
                    "operation": "detect_events",
                    "event_type": "stopped_vehicle_ahead",
                    "scenario_ids": ["scene-1"],
                },
                {
                    "operation": "detect_events",
                    "event_type": "lane_change",
                    "scenario_paths": ["data/val/scene-1/scenario_scene-1.parquet"],
                },
                {
                    "operation": "compute_track_kinematics",
                    "scenario_ids": ["scene-1"],
                    "track_id": "focal",
                },
                {
                    "operation": "compare_velocity_position_evidence",
                    "scenario_ids": ["scene-1"],
                    "track_id": "focal",
                    "window": {"start_timestep": 10, "end_timestep": 20},
                },
            ],
        ),
        ToolSpec(
            name="ArtifactOps.manage_artifacts",
            category="artifact",
            func=artifact_ops.manage_artifacts,
            args_schema={
                "operation": "str",
                "paths": "list[str | Path] | None",
                "output_dir": "str | Path | None",
                "output_path": "str | Path | None",
                "manifest": "dict[str, Any] | None",
                "preserve_structure": "bool",
                "base_dir": "str | Path | None",
                "overwrite": "bool",
                "missing_ok": "bool",
            },
            purpose="Create folders, copy evidence files, delete files, or write manifests.",
            input_contract=(
                "operation is 'create_folder', 'copy', 'delete', or 'write_manifest'. delete and "
                "overwrite operations are risk-gated before execution."
            ),
            output_contract=(
                "Returns a state-transform envelope: ok, operation, summary.total/succeeded/failed, "
                "and items. Copy/delete operations report each path as an item."
            ),
            when_to_use="Use for organizing exported evidence assets and manifests.",
            do_not_use_when="Do not use for table row/column transformations; use TableOps.transform_table.",
            zh_note="中文备注：文件操作统一入口；delete 和 overwrite 会触发高风险确认。",
            examples=[
                {"operation": "create_folder", "output_dir": "outputs/exports/demo_cases"},
                {
                    "operation": "copy",
                    "paths": {"$from_state": "evidence_paths"},
                    "output_dir": "outputs/exports/demo_cases",
                },
                {"operation": "delete", "paths": ["outputs/exports/demo_cases"]},
            ],
        ),
        ToolSpec(
            name="TableOps.transform_table",
            category="table",
            func=table_ops.transform_table,
            args_schema={
                "operation": "str",
                "input_path": "str | Path",
                "output_path": "str | Path",
                "columns": "list[str] | None",
                "filters": "dict[str, Any] | None",
                "overwrite": "bool",
            },
            purpose="Filter, redact, or convert tabular parquet/csv files.",
            input_contract=(
                "operation is 'drop_columns', 'filter_rows', or 'convert_format'. input_path and "
                "output_path are required. Omit overwrite or set it to false for a new derived "
                "output path. Set overwrite=true only when the user explicitly asks to replace "
                "an existing file or modify the input path in place."
            ),
            output_contract=(
                "Returns a state-transform envelope: ok, operation, summary.total/succeeded/failed, "
                "and items, plus operation-specific table metadata."
            ),
            when_to_use="Use for derived table exports, public redaction, row filtering, or csv/parquet conversion.",
            do_not_use_when="Do not use for evidence/event retrieval; use EvidenceStore.query_index.",
            zh_note="中文备注：表格转换统一入口；覆盖写或原地修改会触发高风险确认。",
            examples=[
                {
                    "operation": "filter_rows",
                    "input_path": "outputs/labeled_event_index.parquet",
                    "output_path": "outputs/exports/valid_hard.parquet",
                    "filters": {"event_type": "hard_braking", "is_valid_event": True},
                }
            ],
        ),
    ]
    return {spec.name: spec for spec in specs}


def render_tool_description(spec: ToolSpec) -> str:
    return (
        f"{spec.purpose}\n"
        f"Input: {spec.input_contract}\n"
        f"Output: {spec.output_contract}\n"
        f"When to use: {spec.when_to_use}\n"
        f"Do not use when: {spec.do_not_use_when}\n"
        f"中文备注: {spec.zh_note}\n"
        f"Examples: {spec.examples}"
    )


def preflight_tool_call(registry: ToolRegistry, tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(tool_call.get("tool", ""))
    args = dict(tool_call.get("args", {}))
    if tool_name not in registry:
        return {
            "allowed_tool": False,
            "tool": tool_name,
            "risk": {
                "risk_level": "high",
                "requires_confirmation": True,
                "reason": "unknown tool is not in registry whitelist",
            },
        }
    return {
        "allowed_tool": True,
        "tool": tool_name,
        "risk": assess_tool_risk(tool_name, args),
    }


def execute_registered_tool(
    registry: ToolRegistry,
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    tool_name = str(tool_call.get("tool", ""))
    args = dict(tool_call.get("args", {}))
    if tool_name not in registry:
        raise KeyError(f"Unknown tool: {tool_name}")
    spec = registry[tool_name]
    return {
        "tool": tool_name,
        "args": args,
        "result": spec.func(**args),
    }


def tool_descriptions(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "category": spec.category,
            "description": spec.description,
            "args_schema": spec.args_schema,
            "purpose": spec.purpose,
            "input_contract": spec.input_contract,
            "output_contract": spec.output_contract,
            "when_to_use": spec.when_to_use,
            "do_not_use_when": spec.do_not_use_when,
            "zh_note": spec.zh_note,
            "examples": spec.examples,
        }
        for spec in registry.values()
    ]
