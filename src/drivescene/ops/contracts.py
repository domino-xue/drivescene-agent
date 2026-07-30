from __future__ import annotations

from pathlib import Path
from typing import Any


def tool_result(operation: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = sum(1 for item in items if item.get("ok") is True)
    failed = len(items) - succeeded
    return {
        "ok": failed == 0,
        "operation": operation,
        "summary": {"total": len(items), "succeeded": succeeded, "failed": failed},
        "items": items,
    }


def success_item(input_value: Any, result: Any) -> dict[str, Any]:
    return {
        "input": _jsonable(input_value),
        "ok": True,
        "result": _jsonable(result),
        "error": None,
    }


def error_item(input_value: Any, error: Exception) -> dict[str, Any]:
    return {
        "input": _jsonable(input_value),
        "ok": False,
        "result": None,
        "error_type": error.__class__.__name__,
        "error": str(error),
    }


def validation_error_result(
    operation: str,
    error: Exception,
    input_value: Any = None,
) -> dict[str, Any]:
    result = tool_result(operation, [error_item(input_value, error)])
    result["error_type"] = error.__class__.__name__
    result["error"] = str(error)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
