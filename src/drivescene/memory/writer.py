from __future__ import annotations

from typing import Any

from drivescene.memory.context import semantic_tags_from_text
from drivescene.memory.store import MemoryStore


class AgentMemoryWriter:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def record_tool_result(
        self,
        user_id: int,
        thread_id: int,
        user_request: str,
        tool_name: str | None,
        result: Any,
    ) -> None:
        if not isinstance(result, dict):
            return
        if tool_name == "EvidenceStore.query_index":
            self._record_event_result(
                user_id=user_id,
                thread_id=thread_id,
                user_request=user_request,
                tool_name=tool_name,
                result=result,
            )
            return
        if tool_name != "ArtifactOps.manage_artifacts":
            return
        operation = str(result.get("operation", ""))
        if operation == "delete":
            deleted_paths = _deleted_paths_from_result(result)
            if deleted_paths:
                self.store.mark_path_deleted(user_id, deleted_paths)
            return

        tags = _artifact_tags(user_request, operation)
        if operation == "create_folder":
            output_dir = result.get("output_dir") or _first_item_result_value(result, "output_dir")
            if output_dir:
                self._remember_path(
                    user_id=user_id,
                    thread_id=thread_id,
                    value=str(output_dir),
                    object_kind="folder",
                    semantic_tags=tags,
                    description=f"Folder created by {tool_name} for: {user_request}",
                    source_tool=tool_name,
                    operation=operation,
                )
            return

        if operation == "copy":
            output_dir = result.get("output_dir")
            if output_dir:
                self._remember_path(
                    user_id=user_id,
                    thread_id=thread_id,
                    value=str(output_dir),
                    object_kind="folder",
                    semantic_tags=tags,
                    description=f"Folder containing copied artifacts for: {user_request}",
                    source_tool=tool_name,
                    operation=operation,
                )
            return

        if operation == "write_manifest":
            manifest_path = result.get("manifest_path") or _first_item_result_value(
                result,
                "manifest_path",
            )
            if manifest_path:
                self._remember_path(
                    user_id=user_id,
                    thread_id=thread_id,
                    value=str(manifest_path),
                    object_kind="file",
                    semantic_tags=tags + ["manifest"],
                    description=f"Manifest file written for: {user_request}",
                    source_tool=tool_name,
                    operation=operation,
                )

    def _remember_path(
        self,
        user_id: int,
        thread_id: int,
        value: str,
        object_kind: str,
        semantic_tags: list[str],
        description: str,
        source_tool: str,
        operation: str,
    ) -> None:
        self.store.upsert_memory_item(
            user_id=user_id,
            thread_id=thread_id,
            key="artifact_folder" if object_kind == "folder" else "artifact_file",
            memory_type="path",
            value=value,
            object_kind=object_kind,
            semantic_tags=semantic_tags,
            description=description,
            source_tool=source_tool,
            operation=operation,
        )

    def _record_event_result(
        self,
        user_id: int,
        thread_id: int,
        user_request: str,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        operation = str(result.get("operation", ""))
        if operation not in {"search", "detail", "evidence"}:
            return
        for event in _event_values_from_result(result):
            self.store.upsert_memory_item(
                user_id=user_id,
                thread_id=thread_id,
                key="last_event",
                memory_type="entity",
                value=event,
                object_kind="event",
                semantic_tags=_event_tags(user_request, event),
                description=f"Event returned by {tool_name} for: {user_request}",
                source_tool=tool_name,
                operation=operation,
            )


def _artifact_tags(user_request: str, operation: str) -> list[str]:
    tags = set(semantic_tags_from_text(user_request))
    if operation in {"create_folder", "copy"}:
        tags.add("export")
        tags.add("evidence_folder")
    return sorted(tags)


def _event_values_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in result.get("items", []):
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        item_result = item.get("result")
        if not isinstance(item_result, dict) or not item_result.get("review_id"):
            continue
        event = {
            key: str(item_result[key])
            for key in [
                "review_id",
                "event_type",
                "scenario_id",
                "animation_path",
                "metrics_path",
                "trajectory_path",
                "asset_dir",
            ]
            if item_result.get(key) is not None
        }
        events.append(event)
    return events


def _event_tags(user_request: str, event: dict[str, Any]) -> list[str]:
    tags = set(semantic_tags_from_text(user_request))
    event_type = event.get("event_type")
    if event_type:
        tags.add(str(event_type))
    tags.add("event")
    return sorted(tags)


def _deleted_paths_from_result(result: dict[str, Any]) -> list[str]:
    paths = [str(path) for path in result.get("deleted_files", [])]
    for item in result.get("items", []):
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        item_result = item.get("result")
        if isinstance(item_result, dict) and item_result.get("deleted_path"):
            paths.append(str(item_result["deleted_path"]))
    return sorted(set(paths))


def _first_item_result_value(result: dict[str, Any], key: str) -> Any:
    for item in result.get("items", []):
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        item_result = item.get("result")
        if isinstance(item_result, dict) and item_result.get(key):
            return item_result[key]
    return None
