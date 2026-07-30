from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.review.queue import build_review_queue, load_jsonl, write_jsonl


def main() -> None:
    reviewed = load_jsonl(Path("outputs/reviewed_events.jsonl"))
    queue = build_review_queue(
        [
            Path("outputs/hard_braking_events.jsonl"),
            Path("outputs/close_following_events.jsonl"),
            Path("outputs/cut_in_events.jsonl"),
            Path("outputs/stopped_vehicle_ahead_events.jsonl"),
            Path("outputs/lane_change_events.jsonl"),
        ],
        per_type_limit=100,
        reviewed_items=reviewed,
    )
    output = Path("outputs/review_queue.jsonl")
    write_jsonl(output, queue)
    print(f"Wrote {len(queue)} review items to {output}")


if __name__ == "__main__":
    main()
