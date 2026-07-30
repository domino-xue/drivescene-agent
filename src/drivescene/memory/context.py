from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

from drivescene.memory.store import MemoryItem, MemoryStore


@dataclass(frozen=True)
class ContextPack:
    current_request: str
    thread_summary: str
    recent_messages: list[dict[str, Any]]
    structured_memory: list[dict[str, Any]]
    resolved_context: dict[str, Any] = field(default_factory=dict)
    unresolved_references: list[dict[str, Any]] = field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        store: MemoryStore,
        recent_message_limit: int = 6,
        memory_limit: int = 10,
        summary_char_limit: int = 800,
    ) -> None:
        self.store = store
        self.recent_message_limit = recent_message_limit
        self.memory_limit = memory_limit
        self.summary_char_limit = summary_char_limit

    def build_context_pack(
        self,
        user_id: int,
        thread_id: int,
        current_request: str,
    ) -> ContextPack:
        thread = self.store.get_thread(thread_id)
        thread_summary = (thread.summary if thread is not None else "")[: self.summary_char_limit]
        recent_messages = [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
            }
            for message in self.store.list_messages(thread_id, limit=self.recent_message_limit)
        ]

        tags = semantic_tags_from_text(current_request)
        mentions_folder = _mentions_folder(current_request)
        relevant_memory = self._retrieve_relevant_memory(
            user_id=user_id,
            thread_id=thread_id,
            tags=tags,
            mentions_folder=mentions_folder,
        )
        resolved_context, unresolved = self._resolve_context(current_request, tags, relevant_memory)

        return ContextPack(
            current_request=current_request,
            thread_summary=thread_summary,
            recent_messages=recent_messages,
            structured_memory=[_memory_to_context_dict(item) for item in relevant_memory],
            resolved_context=resolved_context,
            unresolved_references=unresolved,
        )

    def _retrieve_relevant_memory(
        self,
        user_id: int,
        thread_id: int,
        tags: list[str],
        mentions_folder: bool,
    ) -> list[MemoryItem]:
        if mentions_folder:
            if tags:
                return self.store.query_memory(
                    user_id=user_id,
                    thread_id=thread_id,
                    memory_type="path",
                    object_kind="folder",
                    semantic_tags=tags,
                    include_user_scope=True,
                    limit=self.memory_limit,
                )
            return self.store.query_memory(
                user_id=user_id,
                thread_id=thread_id,
                memory_type="path",
                object_kind="folder",
                include_user_scope=True,
                limit=self.memory_limit,
            )

        if tags:
            return self.store.query_memory(
                user_id=user_id,
                thread_id=thread_id,
                semantic_tags=tags,
                include_user_scope=True,
                limit=self.memory_limit,
            )
        return self.store.query_memory(
            user_id=user_id,
            thread_id=thread_id,
            include_user_scope=True,
            limit=self.memory_limit,
        )

    def _resolve_context(
        self,
        current_request: str,
        tags: list[str],
        memory: list[MemoryItem],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        resolved: dict[str, Any] = {}
        unresolved: list[dict[str, Any]] = []
        event_memory = [
            item
            for item in memory
            if item.memory_type == "entity" and item.object_kind == "event"
        ]

        if _asks_about_event_focus(current_request):
            focus = _dominant_event_focus(event_memory)
            if focus is not None:
                resolved["event_focus"] = focus
                resolved["direct_answer"] = (
                    f"你最近主要咨询的是 {focus['event_type']}（{_event_type_label(focus['event_type'])}）事件。"
                )
            else:
                resolved["direct_answer"] = "当前上下文中还没有足够的事件咨询记录，无法判断主要事件类别。"

        if _looks_like_event_reference(current_request):
            selected_event = _select_event_memory(event_memory, tags)
            if selected_event is None:
                unresolved.append(
                    {"phrase": current_request, "reason": "no_matching_event", "candidates": []}
                )
            else:
                event_value = selected_event.value if isinstance(selected_event.value, dict) else {}
                review_id = event_value.get("review_id")
                if review_id:
                    resolved["review_ids"] = [str(review_id)]
                    resolved["event"] = event_value
                    resolved["source"] = "structured_memory"
                    resolved["source_memory_id"] = selected_event.id

        if not _looks_like_path_reference(current_request):
            return resolved, unresolved

        folder_memory = [
            item
            for item in memory
            if item.memory_type == "path" and item.object_kind in {None, "folder"}
        ]
        if not folder_memory:
            unresolved.append(
                {"phrase": current_request, "reason": "no_matching_memory", "candidates": []}
            )
            return resolved, unresolved

        if tags and len(folder_memory) > 1:
            unresolved.append(
                {
                    "phrase": current_request,
                    "reason": "ambiguous_memory",
                    "candidates": [str(item.value) for item in folder_memory],
                }
            )
            return resolved, unresolved

        selected = folder_memory[0]
        resolved.update(
            {
                "paths": [str(selected.value)],
                "source": "structured_memory",
                "source_memory_id": selected.id,
                "matched_tags": tags,
            }
        )
        return resolved, unresolved


def semantic_tags_from_text(text: str) -> list[str]:
    normalized = text.lower()
    tags: list[str] = []
    if "hard_braking" in normalized or "急刹" in text or "急煞" in text:
        tags.append("hard_braking")
    if (
        "close_following" in normalized
        or "近距离跟车" in text
        or "跟车" in text
        or "近距" in text
    ):
        tags.append("close_following")
    if "cut_in" in normalized or "cut-in" in normalized or "切入" in text or "插入" in text:
        tags.append("cut_in")
    if "stopped_vehicle_ahead" in normalized or "前方静止" in text or "静止车辆" in text:
        tags.append("stopped_vehicle_ahead")
    if "lane_change" in normalized or "换道" in text or "变道" in text:
        tags.append("lane_change")
    if "导出" in text or "export" in normalized or "复制" in text:
        tags.append("export")
    return sorted(set(tags))


def _mentions_folder(text: str) -> bool:
    normalized = text.lower()
    return "文件夹" in text or "目录" in text or "folder" in normalized or "dir" in normalized


def _looks_like_path_reference(text: str) -> bool:
    if not _mentions_folder(text):
        return False
    return any(token in text for token in ["这个", "该", "之前", "上次", "刚刚", "建的", "包含"])


def _looks_like_event_reference(text: str) -> bool:
    return any(token in text for token in ["它", "这个事件", "该事件", "上一个事件", "刚才的事件"]) and any(
        token in text for token in ["文件", "地址", "路径", "证据", "动画", "详情", "信息"]
    )


def _asks_about_event_focus(text: str) -> bool:
    return ("哪类事件" in text or "什么事件" in text) and any(
        token in text for token in ["主要", "咨询", "问", "聊"]
    )


def _select_event_memory(memory: list[MemoryItem], tags: list[str]) -> MemoryItem | None:
    if not memory:
        return None
    event_tags = set(tags) - {"export", "event"}
    if event_tags:
        tagged = [item for item in memory if event_tags.issubset(set(item.semantic_tags))]
        if tagged:
            return tagged[0]
    return memory[0]


def _dominant_event_focus(memory: list[MemoryItem]) -> dict[str, Any] | None:
    counter: Counter[str] = Counter()
    for item in memory:
        value = item.value if isinstance(item.value, dict) else {}
        event_type = value.get("event_type")
        if event_type:
            counter[str(event_type)] += 1
            continue
        for tag in item.semantic_tags:
            if tag in _EVENT_TYPE_LABELS:
                counter[tag] += 1
    if not counter:
        return None
    event_type, count = counter.most_common(1)[0]
    return {"event_type": event_type, "count": count}


_EVENT_TYPE_LABELS = {
    "hard_braking": "急刹",
    "close_following": "近距离跟车",
    "cut_in": "切入",
    "stopped_vehicle_ahead": "前方静止车辆",
    "lane_change": "换道",
}


def _event_type_label(event_type: str) -> str:
    return _EVENT_TYPE_LABELS.get(event_type, event_type)


def _memory_to_context_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "scope": item.scope,
        "key": item.key,
        "type": item.memory_type,
        "value": item.value,
        "object_kind": item.object_kind,
        "semantic_tags": item.semantic_tags,
        "description": item.description,
        "source_tool": item.source_tool,
        "operation": item.operation,
        "status": item.status,
        "updated_at": item.updated_at,
    }
