from pathlib import Path

import pandas as pd

from drivescene.data.loader import find_scenario_files, load_scenario


def summarize_dataset(root: Path, limit: int = 100) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in find_scenario_files(root, limit=limit):
        df = load_scenario(path)
        actor_lengths = df.groupby("track_id")["timestep"].nunique()
        object_counts = df[["track_id", "object_type"]].drop_duplicates()["object_type"].value_counts()
        observed_range = df[df["observed"]]["timestep"]
        future_range = df[~df["observed"]]["timestep"]
        focal_track_id = str(df["focal_track_id"].iloc[0])

        row: dict[str, object] = {
            "scenario_id": str(df["scenario_id"].iloc[0]),
            "city": str(df["city"].iloc[0]),
            "num_rows": int(len(df)),
            "num_actors": int(df["track_id"].nunique()),
            "avg_track_length": float(actor_lengths.mean()),
            "focal_track_id": focal_track_id,
            "focal_track_present": bool((df["track_id"].astype(str) == focal_track_id).any()),
            "observed_min_timestep": int(observed_range.min()) if not observed_range.empty else None,
            "observed_max_timestep": int(observed_range.max()) if not observed_range.empty else None,
            "future_min_timestep": int(future_range.min()) if not future_range.empty else None,
            "future_max_timestep": int(future_range.max()) if not future_range.empty else None,
        }
        for object_type, count in object_counts.items():
            row[f"num_{object_type}"] = int(count)
        rows.append(row)

    return pd.DataFrame(rows)
