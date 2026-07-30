from collections import Counter, defaultdict
from typing import Any


def compute_review_metrics_for_queue(
    queue: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    k: int = 20,
) -> dict[str, Any]:
    reviewed_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reviewed:
        reviewed_by_id[str(item.get("review_id"))].append(item)

    filtered: list[dict[str, Any]] = []
    stale_reviewed = 0

    for queue_item in queue:
        review_id = str(queue_item.get("review_id"))
        reviewed_items = reviewed_by_id.get(review_id, [])
        if not reviewed_items:
            continue
        matching = [
            reviewed_item
            for reviewed_item in reviewed_items
            if _event_identity(queue_item) == _event_identity(reviewed_item)
        ]
        stale_reviewed += len(reviewed_items) - len(matching)
        if matching:
            filtered.append(matching[-1])

    queue_ids = {str(item.get("review_id")) for item in queue}
    stale_reviewed += sum(1 for item in reviewed if str(item.get("review_id")) not in queue_ids)

    metrics = compute_review_metrics(filtered, k=k)
    metrics["num_stale_reviewed"] = stale_reviewed
    metrics["num_labeled_in_queue"] = len(filtered)
    metrics["num_pending_or_unclear_in_queue"] = len(queue) - metrics["num_reviewed"]
    metrics["num_queue_items"] = len(queue)
    return metrics


def compute_review_metrics(reviewed: list[dict[str, Any]], k: int = 20) -> dict[str, Any]:
    completed = [item for item in reviewed if item.get("review", {}).get("is_valid_event") is not None]
    valid = [item for item in completed if item["review"]["is_valid_event"] is True]
    precision = len(valid) / len(completed) if completed else 0.0
    top_k = completed[:k]
    top_k_valid = [item for item in top_k if item["review"]["is_valid_event"] is True]

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in completed:
        by_type[str(item["event_type"])].append(item)

    valid_rate_by_event_type = {}
    for event_type, items in by_type.items():
        count_valid = sum(1 for item in items if item["review"]["is_valid_event"] is True)
        valid_rate_by_event_type[event_type] = count_valid / len(items)

    fp_reasons = Counter(
        item["review"].get("false_positive_reason")
        for item in completed
        if item["review"]["is_valid_event"] is False and item["review"].get("false_positive_reason")
    )

    return {
        "num_reviewed": len(completed),
        "num_valid": len(valid),
        "precision": precision,
        f"precision_at_{k}": len(top_k_valid) / len(top_k) if top_k else 0.0,
        "valid_rate_by_event_type": valid_rate_by_event_type,
        "false_positive_breakdown": dict(fp_reasons),
    }


def _event_identity(item: dict[str, Any]) -> tuple[object, ...]:
    return (
        item.get("scenario_id"),
        item.get("event_type"),
        str(item.get("track_id")),
        str(item.get("actor_id", "")),
        item.get("start_timestep"),
        item.get("end_timestep"),
    )
