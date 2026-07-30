from pathlib import Path
from typing import Any

import pandas as pd

from drivescene.ops.contracts import error_item, success_item, tool_result, validation_error_result


class EvidenceStore:
    def __init__(self, index_path: Path | str, scenario_index_path: Path | str | None = None):
        self.index_path = Path(index_path)
        self.scenario_index_path = Path(scenario_index_path) if scenario_index_path is not None else None
        self._index: pd.DataFrame | None = None
        self._scenario_index: pd.DataFrame | None = None

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None:
            self._index = pd.read_parquet(self.index_path)
        return self._index

    @property
    def scenario_index(self) -> pd.DataFrame:
        if self.scenario_index_path is None:
            raise ValueError("scenario_index_path was not provided")
        if self._scenario_index is None:
            self._scenario_index = pd.read_parquet(self.scenario_index_path)
        return self._scenario_index

    def query_index(
        self,
        target: str,
        operation: str,
        filters: dict[str, Any] | None = None,
        review_ids: list[str] | None = None,
        sort_by: str | None = None,
        ascending: bool = True,
        limit: int | None = None,
    ) -> Any:
        try:
            if target == "events":
                return self._query_events(operation, filters or {}, review_ids, sort_by, ascending, limit)
            if target == "scenarios":
                return self._query_scenarios(operation, filters or {}, limit)
            raise ValueError(f"Unknown evidence target: {target}")
        except ValueError as error:
            return validation_error_result(operation, error, {"target": target})

    def _query_events(
        self,
        operation: str,
        filters: dict[str, Any],
        review_ids: list[str] | None,
        sort_by: str | None,
        ascending: bool,
        limit: int | None,
    ) -> Any:
        if operation == "search":
            records = _records(_filter_events(self.index.copy(), filters, sort_by, ascending, limit))
            return tool_result(
                operation,
                [success_item({"filters": filters}, record) for record in records],
            )
        if operation == "summary":
            summarized = _filter_events(self.index.copy(), filters, None, True, None)
            return tool_result(
                operation,
                [success_item({"filters": filters}, _summarize_events(summarized, filters))],
            )
        if operation == "detail":
            return self._query_review_id_batch(operation, review_ids, self._event_detail)
        if operation == "evidence":
            return self._query_review_id_batch(operation, review_ids, self._event_evidence)
        raise ValueError(f"Unknown evidence operation for events: {operation}")

    def _query_review_id_batch(
        self,
        operation: str,
        review_ids: list[str] | None,
        handler,
    ) -> dict[str, Any]:
        try:
            ids = _required_review_ids(review_ids)
        except ValueError as error:
            return validation_error_result(operation, error)

        items: list[dict[str, Any]] = []
        for review_id in ids:
            input_value = {"review_id": review_id}
            try:
                items.append(success_item(input_value, handler(review_id)))
            except (KeyError, ValueError) as error:
                items.append(error_item(input_value, error))
        return tool_result(operation, items)

    def _query_scenarios(
        self,
        operation: str,
        filters: dict[str, Any],
        limit: int | None,
    ) -> Any:
        if operation != "search":
            raise ValueError(f"Unknown evidence operation for scenarios: {operation}")
        result = self.scenario_index.copy()
        city = filters.get("city")
        object_type = filters.get("object_type")
        has_map = filters.get("has_map")
        if city is not None and "city" in result.columns:
            result = result[result["city"] == city]
        if object_type is not None and "object_types" in result.columns:
            result = result[
                result["object_types"].fillna("").str.split(",").apply(lambda values: object_type in values)
            ]
        if has_map is not None and "has_map" in result.columns:
            result = result[result["has_map"] == has_map]
        if limit is not None:
            result = result.head(limit)
        records = _records(result)
        return tool_result(
            operation,
            [success_item({"filters": filters}, record) for record in records],
        )

    def _event_detail(self, review_id: str) -> dict[str, Any]:
        match = self.index[self.index["review_id"].astype(str) == str(review_id)]
        if match.empty:
            raise KeyError(f"Unknown review_id: {review_id}")
        return _records(match.head(1))[0]

    def _event_evidence(self, review_id: str) -> dict[str, Any]:
        detail = self._event_detail(review_id)
        return {
            "review_id": detail["review_id"],
            "animation_path": detail.get("animation_path"),
            "metrics_path": detail.get("metrics_path"),
            "trajectory_path": detail.get("trajectory_path"),
            "has_animation": Path(str(detail.get("animation_path", ""))).exists(),
            "has_metrics": Path(str(detail.get("metrics_path", ""))).exists(),
            "has_trajectory": Path(str(detail.get("trajectory_path", ""))).exists(),
        }


def _filter_events(
    result: pd.DataFrame,
    filters: dict[str, Any],
    sort_by: str | None,
    ascending: bool,
    limit: int | None,
) -> pd.DataFrame:
    for column, value in filters.items():
        if column not in result.columns or value is None:
            continue
        if isinstance(value, list):
            result = result[result[column].isin(value)]
        else:
            result = result[result[column] == value]
    if sort_by is not None and sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending, na_position="last")
    if limit is not None:
        result = result.head(limit)
    return result


def _summarize_events(index: pd.DataFrame, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    num_events = len(index)
    valid_mask = index.get("is_valid_event", pd.Series([], dtype=object)) == True  # noqa: E712
    num_valid = int(valid_mask.sum()) if len(index) else 0
    return {
        "num_events": num_events,
        "num_valid": num_valid,
        "valid_rate": num_valid / num_events if num_events else 0.0,
        "event_type_counts": index["event_type"].value_counts().to_dict()
        if "event_type" in index.columns
        else {},
        "filters": filters or {},
    }


def _required_review_ids(review_ids: list[str] | None) -> list[str]:
    if not review_ids:
        raise ValueError("review_ids is required for this evidence operation")
    return [str(review_id) for review_id in review_ids]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.where(pd.notnull(df), None).to_dict(orient="records")
