from pathlib import Path
from typing import Any

import pandas as pd

from drivescene.ops.contracts import success_item, tool_result, validation_error_result


class TableOps:
    def transform_table(
        self,
        operation: str,
        input_path: Path | str,
        output_path: Path | str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        try:
            if operation == "drop_columns":
                result = _drop_columns(input_path, output_path, columns or [], overwrite)
                return _with_legacy_fields(operation, result)
            if operation == "filter_rows":
                result = _filter_rows(input_path, output_path, filters or {}, overwrite)
                return _with_legacy_fields(operation, result)
            if operation == "convert_format":
                result = _convert_format(input_path, output_path, overwrite)
                return _with_legacy_fields(operation, result)
            raise ValueError(f"Unknown table operation: {operation}")
        except (FileExistsError, ValueError) as error:
            return validation_error_result(operation, error, {"input_path": input_path, "output_path": output_path})


def _drop_columns(
    input_path: Path | str,
    output_path: Path | str,
    columns: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    _ensure_can_write(input_file, output_file, overwrite)
    df = _read_table(input_file)
    existing_columns = [column for column in columns if column in df.columns]
    result = df.drop(columns=existing_columns)
    _write_table(result, output_file)
    return {
        "operation": "drop_columns",
        "input_path": str(input_file),
        "output_path": str(output_file),
        "dropped_columns": existing_columns,
        "missing_columns": [column for column in columns if column not in df.columns],
        "remaining_columns": list(result.columns),
        "num_rows": len(result),
    }


def _filter_rows(
    input_path: Path | str,
    output_path: Path | str,
    filters: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    _ensure_can_write(input_file, output_file, overwrite)
    df = _read_table(input_file)
    result = df.copy()
    ignored_filters: dict[str, Any] = {}
    for column, value in filters.items():
        if column not in result.columns:
            ignored_filters[column] = value
            continue
        if isinstance(value, list):
            result = result[result[column].isin(value)]
        else:
            result = result[result[column] == value]
    _write_table(result, output_file)
    return {
        "operation": "filter_rows",
        "input_path": str(input_file),
        "output_path": str(output_file),
        "filters": filters,
        "ignored_filters": ignored_filters,
        "num_rows": len(result),
    }


def _convert_format(
    input_path: Path | str,
    output_path: Path | str,
    overwrite: bool,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    _ensure_can_write(input_file, output_file, overwrite)
    df = _read_table(input_file)
    _write_table(df, output_file)
    return {
        "operation": "convert_format",
        "input_path": str(input_file),
        "output_path": str(output_file),
        "num_rows": len(df),
        "columns": list(df.columns),
    }


def _ensure_can_write(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite and input_path.resolve() != output_path.resolve():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _with_legacy_fields(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    envelope = tool_result(
        operation,
        [success_item({"input_path": result.get("input_path")}, result)],
    )
    return {**envelope, **result}
