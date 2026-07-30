from __future__ import annotations

import json
import math
from typing import Any

from drivescene.agent.digest import ExecutionDigest


class LLMReporter:
    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def generate(self, digest: ExecutionDigest) -> str:
        if self.model is not None:
            try:
                response = self.model.invoke(_reporter_prompt(digest))
                content = getattr(response, "content", response)
                text = str(content).strip()
                if text:
                    return text
            except Exception:  # noqa: BLE001
                pass
        return deterministic_report(digest)


def deterministic_report(digest: ExecutionDigest) -> str:
    if digest.status == "needs_confirmation":
        return "此操作需要人工确认后才能继续执行。"

    heading = "已完成" if digest.status == "completed" else "已部分完成"
    if digest.answer_hints:
        return "\n".join(digest.answer_hints)

    lines = [f"{heading}：{digest.task}", "", "结果："]
    if digest.events_found:
        event_suffix = f"（{', '.join(digest.event_types)}）" if digest.event_types else ""
        lines.append(f"- 共找到 {digest.events_found} 条事件{event_suffix}")
    if digest.event_summary:
        filters = digest.event_summary.get("filters") or {}
        if filters:
            rendered_filters = ", ".join(f"{key}={value}" for key, value in filters.items())
            lines.append(f"- 统计条件：{rendered_filters}")
        if "num_valid" in digest.event_summary:
            lines.append(f"- 其中有效事件：{digest.event_summary['num_valid']} 条")
    if digest.scenarios_processed:
        lines.append(f"- 共处理 {digest.scenarios_processed} 个场景")
    if digest.files_copied:
        lines.append(f"- 已复制 {digest.files_copied} 个文件")
    if digest.files_deleted:
        lines.append(f"- 已删除 {digest.files_deleted} 个路径")
    if digest.failed_items:
        lines.append(f"- 有 {len(digest.failed_items)} 个项目失败")
    if not any(
        [
            digest.events_found,
            digest.scenarios_processed,
            digest.files_copied,
            digest.files_deleted,
            digest.failed_items,
        ]
    ):
        lines.append("- 任务已执行完成")

    if digest.events:
        top_event = digest.events[0]
        if top_event.get("event_type") == "close_following" and _has_metric(
            top_event,
            "min_front_distance_m",
        ):
            lines.extend(["", "跟车距离最近的事件："])
            lines.append("- " + _render_event_line(top_event))
        elif _has_metric(top_event, "min_velocity_acceleration_mps2"):
            lines.extend(["", "刹停加速度最大的事件："])
            lines.append("- " + _render_event_line(top_event))
        lines.extend(["", "事件样例："])
        for event in digest.events[:5]:
            lines.append("- " + _render_event_line(event))

    display_paths = _display_output_paths(digest)
    if display_paths:
        lines.extend(["", "文件位置："])
        lines.extend(display_paths)

    if digest.execution_summary:
        lines.extend(["", "执行摘要："])
        for index, summary in enumerate(digest.execution_summary, start=1):
            lines.append(f"{index}. {summary}")

    if digest.failed_items:
        lines.extend(["", "失败项："])
        for item in digest.failed_items[:5]:
            lines.append(f"- {item.get('error_type')}: {item.get('error')}")

    return "\n".join(lines)


def _render_event_line(event: dict[str, Any]) -> str:
    parts = [f"review_id={event.get('review_id')}"]
    if event.get("event_type"):
        parts.append(f"event_type={event.get('event_type')}")
    if event.get("scenario_id"):
        parts.append(f"scenario_id={event.get('scenario_id')}")
    if _has_metric(event, "min_front_distance_m"):
        parts.append(f"min_front_distance_m={event.get('min_front_distance_m')}")
    if _has_metric(event, "min_distance_m"):
        parts.append(f"min_distance_m={event.get('min_distance_m')}")
    if _has_metric(event, "min_ttc_s"):
        parts.append(f"min_ttc_s={event.get('min_ttc_s')}")
    if _has_metric(event, "max_closing_speed_mps"):
        parts.append(f"max_closing_speed_mps={event.get('max_closing_speed_mps')}")
    if _has_metric(event, "min_velocity_acceleration_mps2"):
        parts.append(
            f"min_velocity_acceleration_mps2={event.get('min_velocity_acceleration_mps2')}"
        )
    if _has_metric(event, "min_acceleration_mps2"):
        parts.append(f"min_acceleration_mps2={event.get('min_acceleration_mps2')}")
    return "，".join(parts)


def _has_metric(event: dict[str, Any], key: str) -> bool:
    value = event.get(key)
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _display_output_paths(digest: ExecutionDigest) -> list[str]:
    if not digest.files_copied:
        return digest.output_paths
    copied_outputs = [
        path
        for path in digest.output_paths
        if "outputs\\review_assets" not in path and "outputs/review_assets" not in path
    ]
    return copied_outputs or digest.output_paths


def _reporter_prompt(digest: ExecutionDigest) -> str:
    return (
        "你是 DriveScene Agent 的 Reporter。请只根据 ExecutionDigest 生成中文最终回答。\n"
        "要求：用户友好、简洁、结果优先；保留准确的数量和路径；不要编造 digest 中没有的事实；"
        "不要输出原始 JSON；如果存在失败项，需要简要说明。\n\n"
        "ExecutionDigest:\n"
        + json.dumps(digest.to_dict(), ensure_ascii=False, indent=2)
    )
