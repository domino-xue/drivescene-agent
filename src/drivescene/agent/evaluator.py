from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class EvaluationResult:
    status: str
    completed: bool
    needs_replan: bool = False
    missing_requirements: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completed": self.completed,
            "needs_replan": self.needs_replan,
            "missing_requirements": self.missing_requirements,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TaskRequirements:
    event_type: str | None = None
    min_events: int = 0
    require_summary: bool = False
    require_animation: bool = False
    require_metrics: bool = False
    require_trajectory: bool = False
    require_evidence: bool = False
    require_export: bool = False
    require_delete: bool = False
    require_raw_analysis: bool = False
    require_table_output: bool = False


class TaskCompletionEvaluator:
    """Evaluate whether deterministic observations satisfy the user's deliverables."""

    def evaluate(self, state: Any, digest: Any) -> EvaluationResult:
        if getattr(state, "pending_confirmation", None) is not None:
            return EvaluationResult(
                status="needs_confirmation",
                completed=False,
                reason="A high-risk operation is waiting for human confirmation.",
            )
        if getattr(state, "denied_confirmations", []):
            return EvaluationResult(
                status="cancelled",
                completed=True,
                reason="The user rejected a high-risk operation; do not replan it.",
            )

        request = str(getattr(state, "user_request", "") or "")
        requirements = infer_task_requirements(request)
        missing = _missing_requirements(requirements, state, digest)
        failed_steps = [
            step
            for step in getattr(state, "plan", [])
            if getattr(step, "status", "") in {"failed", "blocked"}
        ]
        if failed_steps and not missing:
            missing.append("至少一个计划步骤执行失败，需要生成修正步骤")

        if missing:
            return EvaluationResult(
                status="needs_replan",
                completed=False,
                needs_replan=True,
                missing_requirements=missing,
                reason="The execution observations do not yet satisfy all requested deliverables.",
            )

        if getattr(digest, "failed_items", []):
            return EvaluationResult(
                status="partial",
                completed=True,
                reason="The requested deliverables were produced with item-level failures.",
            )
        return EvaluationResult(
            status="completed",
            completed=True,
            reason="All inferred task requirements are satisfied by deterministic observations.",
        )


class SimpleEvaluator(TaskCompletionEvaluator):
    """Backward-compatible name for the task-completion evaluator."""


def infer_task_requirements(user_request: str) -> TaskRequirements:
    normalized = user_request.lower()
    event_type = _event_type_from_text(user_request, normalized)
    asks_for_events = event_type is not None or any(
        token in normalized for token in ["event", "case", "案例", "事件"]
    )
    min_events = _requested_count(user_request) if asks_for_events else 0
    if asks_for_events and min_events == 0 and any(
        token in normalized for token in ["找", "查", "search", "find", "案例"]
    ):
        min_events = 1

    require_animation = "动画" in user_request or "gif" in normalized
    require_metrics = any(
        token in normalized for token in ["metrics", "metric_path", "指标图", "指标路径"]
    )
    require_trajectory = any(
        token in normalized for token in ["trajectory", "轨迹图", "轨迹路径"]
    )
    require_evidence = any(token in user_request for token in ["证据", "素材"]) or (
        asks_for_events
        and "路径" in user_request
        and not any(token in user_request for token in ["删除", "文件夹", "目录"])
    )

    return TaskRequirements(
        event_type=event_type,
        min_events=min_events,
        require_summary=any(
            token in user_request for token in ["汇总", "统计", "概览", "总结"]
        ),
        require_animation=require_animation,
        require_metrics=require_metrics,
        require_trajectory=require_trajectory,
        require_evidence=require_evidence,
        require_export=any(
            token in normalized for token in ["导出", "复制", "export", "copy"]
        ),
        require_delete="删除" in user_request or "delete" in normalized,
        require_raw_analysis=any(
            token in normalized
            for token in ["重跑", "重新检测", "原始场景", "raw scenario", "run_analysis"]
        ),
        require_table_output=any(
            token in normalized
            for token in ["过滤表格", "删列", "转换格式", "filter_rows", "drop_columns"]
        ),
    )


def _missing_requirements(
    requirements: TaskRequirements,
    state: Any,
    digest: Any,
) -> list[str]:
    missing: list[str] = []
    blackboard = getattr(state, "blackboard", {}) or {}
    events = _merged_events(getattr(digest, "events", []))

    if requirements.require_summary and not getattr(digest, "event_summary", {}):
        missing.append("需要返回事件索引汇总")

    if requirements.min_events > 0 and len(events) < requirements.min_events:
        missing.append(
            f"至少需要 {requirements.min_events} 个事件，当前只有 {len(events)} 个"
        )

    event_types = {str(value) for value in getattr(digest, "event_types", [])}
    if (
        requirements.event_type is not None
        and requirements.min_events > 0
        and requirements.event_type not in event_types
    ):
        missing.append(f"需要返回 {requirements.event_type} 类型事件")

    evidence_count = _evidence_event_count(events, blackboard)
    required_evidence_count = max(1, requirements.min_events)
    if requirements.require_animation:
        animation_count = sum(bool(event.get("animation_path")) for event in events.values())
        if animation_count < required_evidence_count:
            missing.append(
                f"至少需要 {required_evidence_count} 个事件的动画路径，当前只有 {animation_count} 个"
            )
    if requirements.require_metrics:
        metrics_count = sum(bool(event.get("metrics_path")) for event in events.values())
        if metrics_count < required_evidence_count:
            missing.append(
                f"至少需要 {required_evidence_count} 个事件的指标图路径，当前只有 {metrics_count} 个"
            )
    if requirements.require_trajectory:
        trajectory_count = sum(bool(event.get("trajectory_path")) for event in events.values())
        if trajectory_count < required_evidence_count:
            missing.append(
                f"至少需要 {required_evidence_count} 个事件的轨迹图路径，当前只有 {trajectory_count} 个"
            )
    if (
        requirements.require_evidence
        and not any(
            [
                requirements.require_animation,
                requirements.require_metrics,
                requirements.require_trajectory,
            ]
        )
        and evidence_count < required_evidence_count
    ):
        missing.append(
            f"至少需要 {required_evidence_count} 个事件的证据路径，当前只有 {evidence_count} 个"
        )

    if requirements.require_export:
        copied_files = int(getattr(digest, "files_copied", 0) or 0)
        table_output = requirements.require_table_output and bool(
            getattr(digest, "output_paths", [])
        )
        if copied_files == 0 and not table_output:
            missing.append("需要实际导出文件，而不只是创建目录")

    if requirements.require_delete and int(getattr(digest, "files_deleted", 0) or 0) == 0:
        missing.append("需要完成用户确认后的删除操作")

    if (
        requirements.require_raw_analysis
        and int(getattr(digest, "scenarios_processed", 0) or 0) == 0
    ):
        missing.append("需要对至少一个原始场景执行确定性分析")

    if requirements.require_table_output and not getattr(digest, "output_paths", []):
        missing.append("需要生成派生表格文件")

    return missing


def _merged_events(raw_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            continue
        key = str(event.get("review_id") or f"event-{index}")
        merged.setdefault(key, {}).update(event)
    return merged


def _evidence_event_count(
    events: dict[str, dict[str, Any]],
    blackboard: dict[str, Any],
) -> int:
    evidence_ids = {
        review_id
        for review_id, event in events.items()
        if any(
            event.get(key)
            for key in ["animation_path", "metrics_path", "trajectory_path", "asset_dir"]
        )
    }
    for asset in blackboard.get("evidence_assets", []) or []:
        if isinstance(asset, dict) and asset.get("paths"):
            evidence_ids.add(str(asset.get("review_id") or f"asset-{len(evidence_ids)}"))
    return len(evidence_ids)


def _requested_count(text: str) -> int:
    arabic = re.search(r"(\d+)\s*(?:个|条|例|cases?|events?)", text, flags=re.IGNORECASE)
    if arabic is not None:
        return max(0, int(arabic.group(1)))
    chinese = re.search(r"([一二两三四五六七八九十])\s*(?:个|条|例)", text)
    if chinese is None:
        return 0
    values = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return values[chinese.group(1)]


def _event_type_from_text(text: str, normalized: str) -> str | None:
    mappings = [
        ("hard_braking", ["hard_braking", "急刹", "急煞"]),
        ("close_following", ["close_following", "近距离跟车", "跟车"]),
        ("cut_in", ["cut_in", "cut-in", "切入"]),
        ("stopped_vehicle_ahead", ["stopped_vehicle_ahead", "前方静止", "静止车辆"]),
        ("lane_change", ["lane_change", "换道", "变道"]),
    ]
    for event_type, tokens in mappings:
        if any(token in normalized or token in text for token in tokens):
            return event_type
    return None
