from pathlib import Path

import pandas as pd

from drivescene.data.loader import (
    find_scenario_files,
    get_actor_track,
    get_focal_track,
    load_scenario,
)


def test_find_scenario_files_sorts_and_limits(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    b_file = tmp_path / "b" / "scenario_b.parquet"
    a_file = tmp_path / "a" / "scenario_a.parquet"
    b_file.write_bytes(b"placeholder")
    a_file.write_bytes(b"placeholder")

    assert find_scenario_files(tmp_path, limit=1) == [a_file]


def test_track_helpers_return_sorted_tracks() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["focal", "other", "focal"],
            "focal_track_id": ["focal", "focal", "focal"],
            "timestep": [2, 1, 1],
            "position_x": [2.0, 0.0, 1.0],
            "position_y": [0.0, 0.0, 0.0],
        }
    )

    focal = get_focal_track(df)
    other = get_actor_track(df, "other")

    assert focal["timestep"].tolist() == [1, 2]
    assert other["track_id"].tolist() == ["other"]


def test_load_real_scenario_smoke() -> None:
    files = find_scenario_files(Path("data/val"), limit=1)

    df = load_scenario(files[0])

    assert not df.empty
    assert {"track_id", "timestep", "position_x", "position_y", "focal_track_id"}.issubset(
        df.columns
    )
