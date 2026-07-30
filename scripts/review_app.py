from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from drivescene.review.queue import event_identity, load_jsonl, write_jsonl


FALSE_POSITIVE_REASONS = [
    "",
    "not_in_front",
    "too_far",
    "low_speed_noise",
    "track_fragment",
    "wrong_actor",
    "brief_contact",
    "normal_following",
    "visual_unclear",
]


VALID_OPTIONS = ["valid", "invalid", "unclear"]
SEVERITY_OPTIONS = ["", "low", "medium", "high"]
EVENT_TYPE_OPTIONS = [
    "hard_braking",
    "close_following",
    "cut_in",
    "stopped_vehicle_ahead",
    "lane_change",
    "other",
]


def main() -> None:
    st.set_page_config(page_title="DriveScene Review", layout="wide")
    queue_path = Path("outputs/review_queue.jsonl")
    reviewed_path = Path("outputs/reviewed_events.jsonl")

    queue = load_jsonl(queue_path)
    reviewed = load_jsonl(reviewed_path)
    reviewed_by_id = build_reviewed_lookup(queue, reviewed)

    st.title("DriveScene Human Review")
    st.caption(f"{len(reviewed)} reviewed / {len(queue)} total")

    if not queue:
        st.warning("No review queue found. Run scripts/build_review_queue.py first.")
        return

    current_index = clamp_review_index(int(st.session_state.get("review_index", 0)), len(queue))
    st.session_state["review_index"] = current_index

    selected_number = st.number_input(
        "Review item",
        min_value=1,
        max_value=len(queue),
        value=current_index + 1,
        step=1,
    )
    selected_index = clamp_review_index(int(selected_number) - 1, len(queue))
    if selected_index != current_index:
        st.session_state["review_index"] = selected_index
        st.rerun()

    item = queue[current_index]
    reviewed_item = reviewed_by_id.get(item["review_id"])
    display_item = reviewed_item or item
    existing_review = display_item.get("review", {})

    left, right = st.columns([1.2, 1])

    with left:
        status_text = "reviewed" if reviewed_item else "unreviewed"
        st.subheader(
            f"Review {item['review_id']} - {item['event_type']} "
            f"({current_index + 1}/{len(queue)}, {status_text})"
        )
        asset_dir = Path("outputs/review_assets") / item["review_id"]
        trajectory_window = asset_dir / "trajectory_window.png"
        metrics = asset_dir / "metrics.png"
        animation = asset_dir / "animation.gif"

        if animation.exists():
            st.image(str(animation), caption="Event window animation")
        else:
            st.info(f"No animation found at {animation}. Run scripts/build_review_assets.py.")

        if trajectory_window.exists():
            st.image(str(trajectory_window), caption="Event-local trajectory window")
        else:
            plot_path = _plot_path(item)
            if plot_path.exists():
                st.image(str(plot_path), caption="Fallback full trajectory plot")

        if metrics.exists():
            st.image(str(metrics), caption="Evidence metrics")
        st.json({key: value for key, value in display_item.items() if key not in {"review"}})

    with right:
        st.subheader("Label")
        valid_index = _valid_option_index(existing_review.get("is_valid_event"))
        is_valid = st.radio(
            "Is this a valid event?",
            options=VALID_OPTIONS,
            index=valid_index,
            horizontal=True,
        )
        severity = st.selectbox(
            "Severity",
            SEVERITY_OPTIONS,
            index=_option_index(SEVERITY_OPTIONS, existing_review.get("severity") or ""),
        )
        current_event_type = existing_review.get("correct_event_type", item["event_type"])
        correct_event_type = st.selectbox(
            "Correct event type",
            EVENT_TYPE_OPTIONS,
            index=_option_index(EVENT_TYPE_OPTIONS, current_event_type, default=len(EVENT_TYPE_OPTIONS) - 1),
        )
        start_timestep = st.number_input(
            "Correct start timestep",
            value=int(existing_review.get("correct_start_timestep", item.get("start_timestep", 0))),
            step=1,
        )
        end_timestep = st.number_input(
            "Correct end timestep",
            value=int(existing_review.get("correct_end_timestep", item.get("end_timestep", 0))),
            step=1,
        )
        false_positive_reason = st.selectbox(
            "False positive reason",
            FALSE_POSITIVE_REASONS,
            index=_option_index(
                FALSE_POSITIVE_REASONS,
                existing_review.get("false_positive_reason") or "",
            ),
        )
        review_note = st.text_area("Review note", existing_review.get("review_note", ""))

        nav_prev, nav_save, nav_next = st.columns([1, 1.2, 1])
        with nav_prev:
            if st.button("Previous", disabled=current_index == 0, use_container_width=True):
                st.session_state["review_index"] = clamp_review_index(current_index - 1, len(queue))
                st.rerun()
        with nav_save:
            save_clicked = st.button("Save label", type="primary", use_container_width=True)
        with nav_next:
            if st.button(
                "Next",
                disabled=current_index >= len(queue) - 1,
                use_container_width=True,
            ):
                st.session_state["review_index"] = clamp_review_index(current_index + 1, len(queue))
                st.rerun()

        if save_clicked:
            labeled = dict(item)
            labeled["review_status"] = "reviewed"
            labeled["review"] = {
                "is_valid_event": True
                if is_valid == "valid"
                else False
                if is_valid == "invalid"
                else None,
                "correct_event_type": correct_event_type,
                "severity": severity or None,
                "correct_start_timestep": int(start_timestep),
                "correct_end_timestep": int(end_timestep),
                "false_positive_reason": false_positive_reason or None,
                "review_note": review_note,
            }
            reviewed = upsert_reviewed_item(reviewed, labeled)
            write_jsonl(reviewed_path, reviewed)
            st.session_state["review_index"] = next_review_index_after_save(current_index, len(queue))
            st.rerun()


def clamp_review_index(index: int, queue_size: int) -> int:
    if queue_size <= 0:
        return 0
    return max(0, min(index, queue_size - 1))


def next_review_index_after_save(current_index: int, queue_size: int) -> int:
    return clamp_review_index(current_index + 1, queue_size)


def upsert_reviewed_item(
    reviewed: list[dict],
    labeled_item: dict,
) -> list[dict]:
    result = list(reviewed)
    labeled_identity = event_identity(labeled_item)
    for index, item in enumerate(result):
        if event_identity(item) == labeled_identity:
            result[index] = labeled_item
            return result
    result.append(labeled_item)
    return result


def build_reviewed_lookup(
    queue: list[dict],
    reviewed: list[dict],
) -> dict[str, dict]:
    queue_by_identity = {event_identity(item): item for item in queue}
    result: dict[str, dict] = {}
    for item in reviewed:
        queue_item = queue_by_identity.get(event_identity(item))
        if queue_item is not None:
            result[str(queue_item["review_id"])] = item
    return result


def _valid_option_index(is_valid_event: object) -> int:
    if is_valid_event is True:
        return VALID_OPTIONS.index("valid")
    if is_valid_event is False:
        return VALID_OPTIONS.index("invalid")
    return VALID_OPTIONS.index("unclear")


def _option_index(options: list[str], value: object, default: int = 0) -> int:
    try:
        return options.index(str(value))
    except ValueError:
        return default


def _plot_path(item: dict) -> Path:
    if item["event_type"] == "hard_braking":
        return Path("outputs/plots/hard_braking") / f"{item['scenario_id']}.png"
    if item["event_type"] == "close_following":
        return Path("outputs/plots/close_following") / f"{item['scenario_id']}_{item['actor_id']}.png"
    return Path("outputs/plots") / f"{item['scenario_id']}.png"


if __name__ == "__main__":
    main()
