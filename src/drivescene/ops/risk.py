from pathlib import Path
from typing import Any


def assess_tool_risk(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = tool_name.lower()
    operation = str(args.get("operation", "")).lower()

    if normalized.endswith("manage_artifacts"):
        return _assess_artifact_operation(operation, args)

    if normalized.endswith("transform_table"):
        return _assess_table_operation(args)

    if normalized.endswith(("query_index", "run_analysis")):
        return _assessment("low", "read-only retrieval or analysis operation")

    return _assessment("medium", "unknown tool risk defaults to medium")


def _assess_artifact_operation(operation: str, args: dict[str, Any]) -> dict[str, Any]:
    if operation == "delete":
        return _assessment("high", "delete operation can remove files or directories")
    if operation == "copy":
        if bool(args.get("overwrite", False)):
            return _assessment("high", "copy operation may overwrite existing files")
        return _assessment("low", "copy operation writes derived files without deleting sources")
    if operation == "create_folder":
        return _assessment("low", "create folder is non-destructive")
    if operation == "write_manifest":
        if bool(args.get("overwrite", False)):
            return _assessment("medium", "manifest write may overwrite an existing file")
        return _assessment("low", "manifest write creates a traceability file")
    return _assessment("medium", "unknown artifact operation risk defaults to medium")


def _assess_table_operation(args: dict[str, Any]) -> dict[str, Any]:
    input_path = args.get("input_path")
    output_path = args.get("output_path")
    if bool(args.get("overwrite", False)):
        return _assessment("high", "table operation requested overwrite")
    if input_path is not None and output_path is not None and _same_path(input_path, output_path):
        return _assessment("high", "table operation modifies the input path in place")
    return _assessment("medium", "table operation writes a derived table")


def _assessment(risk_level: str, reason: str) -> dict[str, Any]:
    return {
        "risk_level": risk_level,
        "requires_confirmation": risk_level == "high",
        "reason": reason,
    }


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)
