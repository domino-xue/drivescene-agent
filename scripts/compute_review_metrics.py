import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.eval.review_metrics import compute_review_metrics_for_queue
from drivescene.review.queue import load_jsonl


def main() -> None:
    queue = load_jsonl(Path("outputs/review_queue.jsonl"))
    reviewed = load_jsonl(Path("outputs/reviewed_events.jsonl"))
    metrics = compute_review_metrics_for_queue(queue, reviewed, k=20)
    output = Path("outputs/eval_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
