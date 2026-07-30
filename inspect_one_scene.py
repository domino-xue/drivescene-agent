# from pathlib import Path
# import pandas as pd
#
# import os
# from pathlib import Path
#
# print("cwd =", os.getcwd())
# print("exists =", Path("data/val").exists())
# print("resolve =", Path("data/val").resolve())
#
# scene_file = next(Path(r"E:\drivescene-agent\data\val").rglob("*.parquet"))
# print(scene_file)
#
# df = pd.read_parquet(scene_file)
# print(df.shape)
# print(df.columns)
# print(df.head())

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from drivescene.viz.trajectory import render_trajectory_plot

scene_file = Path(
    "data/val/00010486-9a07-48ae-b493-cf4545855937/"
    "scenario_00010486-9a07-48ae-b493-cf4545855937.parquet"
)

df = pd.read_parquet(scene_file)
Path("outputs").mkdir(exist_ok=True)

focal_track_id = str(df["focal_track_id"].iloc[0])

output = render_trajectory_plot(
    df,
    Path("outputs/one_scene_trajectory.png"),
    title=f"Scenario {df['scenario_id'].iloc[0]}",
    highlight_track_id=focal_track_id,
)
print(f"Wrote {output}")
