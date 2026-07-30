from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivescene.analysis.summary import summarize_dataset


def main() -> None:
    root = Path("data/val")
    output = Path("outputs/dataset_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset(root, limit=100)
    summary.to_csv(output, index=False)
    print(f"Wrote {len(summary)} scenario summaries to {output}")


if __name__ == "__main__":
    main()
