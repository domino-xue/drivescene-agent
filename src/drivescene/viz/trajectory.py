from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render_trajectory_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    highlight_track_id: str | int | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    highlight = str(highlight_track_id) if highlight_track_id is not None else None

    plt.figure(figsize=(8, 8))
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("timestep")
        is_highlight = highlight is not None and str(track_id) == highlight
        plt.plot(
            group["position_x"],
            group["position_y"],
            linewidth=3 if is_highlight else 1,
            alpha=1.0 if is_highlight else 0.3,
            label="highlight" if is_highlight else None,
        )

    plt.axis("equal")
    plt.xlabel("x / meters")
    plt.ylabel("y / meters")
    if title:
        plt.title(title)
    if highlight is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
