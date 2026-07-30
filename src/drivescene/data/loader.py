from pathlib import Path

import pandas as pd


def find_scenario_files(root: Path, limit: int | None = None) -> list[Path]:
    # 发现parquet文件，并根据limit返回多少条
    files = sorted(root.rglob("scenario_*.parquet"))
    if limit is None:
        return files
    return files[:limit]


def load_scenario(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(["track_id", "timestep"]).reset_index(drop=True)


def get_actor_track(df: pd.DataFrame, track_id: str | int) -> pd.DataFrame:
    track = df[df["track_id"].astype(str) == str(track_id)]
    return track.sort_values("timestep").reset_index(drop=True)


def get_focal_track(df: pd.DataFrame) -> pd.DataFrame:
    focal_track_id = df["focal_track_id"].iloc[0]
    return get_actor_track(df, focal_track_id)
