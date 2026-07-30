from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class ExecutionDigest:
    task: str
    status: str = "completed"
    events_found: int = 0
    event_types: list[str] = field(default_factory=list)
    scenarios_processed: int = 0
    files_copied: int = 0
    files_deleted: int = 0
    output_paths: list[str] = field(default_factory=list)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    event_summary: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    answer_hints: list[str] = field(default_factory=list)
    execution_summary: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "events_found": self.events_found,
            "event_types": self.event_types,
            "scenarios_processed": self.scenarios_processed,
            "files_copied": self.files_copied,
            "files_deleted": self.files_deleted,
            "output_paths": self.output_paths,
            "failed_items": self.failed_items,
            "event_summary": self.event_summary,
            "events": self.events,
            "answer_hints": self.answer_hints,
            "execution_summary": self.execution_summary,
            "observations": self.observations,
        }


class ExecutionDigestBuilder:
    def build(self, state: Any) -> ExecutionDigest:
        digest = ExecutionDigest(task=str(getattr(state, "user_request", "")))
        observed_step_ids = {
            str(getattr(step_result, "step_id", ""))
            for step_result in getattr(state, "step_results", [])
        }
        digest.execution_summary = [
            str(step.goal)
            for step in getattr(state, "plan", [])
            if getattr(step, "status", "") in {"completed", "failed", "skipped"}
            or str(getattr(step, "step_id", "")) in observed_step_ids
        ]

        for step_result in getattr(state, "step_results", []):
            self._merge_step_result(digest, step_result)

        if getattr(state, "pending_confirmation", None) is not None:
            digest.status = "needs_confirmation"
        elif digest.failed_items:
            digest.status = "partial" if _has_any_success(digest) else "failed"
        else:
            digest.status = "completed"
        return digest

    def _merge_step_result(self, digest: ExecutionDigest, step_result: Any) -> None:
        result = getattr(step_result, "result", None)
        if not isinstance(result, dict):
            return

        tool_name = getattr(step_result, "tool_name", None)
        args = getattr(step_result, "args", {}) or {}
        observation = {
            "step_id": getattr(step_result, "step_id", ""),
            "tool_name": tool_name,
            "operation": result.get("operation") or args.get("operation"),
            "ok": result.get("ok"),
            "summary": result.get("summary", {}),
        }
        digest.observations.append(observation)
        self._collect_failed_items(digest, step_result, result)

        if tool_name == "EvidenceStore.query_index":
            self._merge_evidence_result(digest, args, result)
        elif tool_name == "ArtifactOps.manage_artifacts":
            self._merge_artifact_result(digest, result)
        elif tool_name == "AnalysisTools.run_analysis":
            self._merge_analysis_result(digest, result)
        elif tool_name == "TableOps.transform_table":
            self._merge_table_result(digest, result)
        elif tool_name is None:
            self._merge_message_result(digest, result)

    def _merge_evidence_result(
        self,
        digest: ExecutionDigest,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        operation = str(result.get("operation") or args.get("operation") or "")
        if operation == "summary":
            for item in result.get("items", []):
                item_result = item.get("result") if isinstance(item, dict) else None
                if not isinstance(item_result, dict):
                    continue
                digest.event_summary = dict(item_result)
                digest.events_found += int(item_result.get("num_events", 0) or 0)
                for event_type in (item_result.get("event_type_counts") or {}).keys():
                    _append_unique(digest.event_types, str(event_type))
                event_type = (item_result.get("filters") or args.get("filters") or {}).get(
                    "event_type"
                )
                if event_type:
                    _append_unique(digest.event_types, str(event_type))
        if operation == "search":
            items = [item for item in result.get("items", []) if item.get("ok") is True]
            digest.events_found += len(items)
            event_type = (args.get("filters") or {}).get("event_type")
            if event_type:
                _append_unique(digest.event_types, str(event_type))
            for item in items:
                item_result = item.get("result") if isinstance(item, dict) else None
                self._collect_event(digest, item_result)
        if operation == "evidence":
            for item in result.get("items", []):
                item_result = item.get("result") if isinstance(item, dict) else None
                if isinstance(item_result, dict):
                    self._collect_event(digest, item_result)
                    for key in ["animation_path", "metrics_path", "trajectory_path", "asset_dir"]:
                        if item_result.get(key):
                            _append_unique(digest.output_paths, str(item_result[key]))

    def _collect_event(self, digest: ExecutionDigest, item_result: Any) -> None:
        if not isinstance(item_result, dict) or not item_result.get("review_id"):
            return
        event = {
            key: item_result[key]
            for key in [
                "review_id",
                "event_type",
                "scenario_id",
                "animation_path",
                "metrics_path",
                "trajectory_path",
                "asset_dir",
                "min_distance_m",
                "min_front_distance_m",
                "min_ttc_s",
                "max_closing_speed_mps",
                "min_velocity_acceleration_mps2",
                "min_position_acceleration_mps2",
                "min_acceleration_mps2",
                "speed_before_mps",
                "speed_after_mps",
                "severity",
            ]
            if _has_value(item_result.get(key))
        }
        if event not in digest.events:
            digest.events.append(event)
        if event.get("event_type"):
            _append_unique(digest.event_types, str(event["event_type"]))
        for key in ["animation_path", "metrics_path", "trajectory_path", "asset_dir"]:
            if event.get(key):
                _append_unique(digest.output_paths, str(event[key]))

    def _merge_artifact_result(self, digest: ExecutionDigest, result: dict[str, Any]) -> None:
        operation = str(result.get("operation") or "")
        if result.get("output_dir"):
            _append_unique(digest.output_paths, str(result["output_dir"]))
        if result.get("manifest_path"):
            _append_unique(digest.output_paths, str(result["manifest_path"]))
        if operation == "copy":
            digest.files_copied += len(result.get("copied_files", []))
        if operation == "delete":
            digest.files_deleted += len(result.get("deleted_files", []))

    def _merge_analysis_result(self, digest: ExecutionDigest, result: dict[str, Any]) -> None:
        summary = result.get("summary")
        if isinstance(summary, dict):
            digest.scenarios_processed += int(summary.get("total", 0) or 0)
        for item in result.get("items", []):
            item_result = item.get("result") if isinstance(item, dict) else None
            if isinstance(item_result, list):
                digest.events_found += len(item_result)

    def _merge_table_result(self, digest: ExecutionDigest, result: dict[str, Any]) -> None:
        for key in ["output_path", "path"]:
            if result.get(key):
                _append_unique(digest.output_paths, str(result[key]))

    def _merge_message_result(self, digest: ExecutionDigest, result: dict[str, Any]) -> None:
        if result.get("operation") != "message":
            return
        for item in result.get("items", []):
            item_result = item.get("result") if isinstance(item, dict) else None
            if isinstance(item_result, str) and item_result:
                _append_unique(digest.answer_hints, item_result)

    def _collect_failed_items(
        self,
        digest: ExecutionDigest,
        step_result: Any,
        result: dict[str, Any],
    ) -> None:
        if getattr(step_result, "error", None):
            digest.failed_items.append(
                {
                    "step_id": getattr(step_result, "step_id", ""),
                    "tool_name": getattr(step_result, "tool_name", None),
                    "input": getattr(step_result, "args", {}),
                    "error_type": "ExecutionError",
                    "error": getattr(step_result, "error"),
                }
            )
        for item in result.get("items", []):
            if isinstance(item, dict) and item.get("ok") is False:
                digest.failed_items.append(
                    {
                        "step_id": getattr(step_result, "step_id", ""),
                        "tool_name": getattr(step_result, "tool_name", None),
                        "input": item.get("input"),
                        "error_type": item.get("error_type"),
                        "error": item.get("error"),
                    }
                )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _has_any_success(digest: ExecutionDigest) -> bool:
    return bool(
        digest.events_found
        or digest.scenarios_processed
        or digest.files_copied
        or digest.files_deleted
        or digest.output_paths
        or digest.answer_hints
    )
