from pathlib import Path
from typing import Any
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

from drivescene.features.kinematics import add_acceleration, add_speed
from drivescene.features.relative_motion import compute_relative_motion
from drivescene.maps.overlay import MapOverlay, draw_map_overlay
from drivescene.review.queue import event_identity


ASSET_RENDERER_VERSION = "event_subject_follow_v1"


def render_review_assets(
    df: pd.DataFrame,
    event: dict[str, Any],
    output_root: Path,
    context_steps: int = 20,
    focus_extent_m: float = 50.0,
    scale_bar_m: float = 10.0,
    map_overlay: MapOverlay | None = None,
) -> dict[str, Path]:
    review_id = str(event["review_id"])
    output_dir = output_root / review_id
    output_dir.mkdir(parents=True, exist_ok=True)

    start = int(event["start_timestep"])
    end = int(event["end_timestep"])
    window = _window_df(df, start, end, context_steps)

    paths = {
        "trajectory_window": output_dir / "trajectory_window.png",
        "metrics": output_dir / "metrics.png",
        "animation": output_dir / "animation.gif",
        "manifest": output_dir / "manifest.json",
    }
    _render_trajectory_window(
        window, event, paths["trajectory_window"], focus_extent_m, scale_bar_m, map_overlay
    )
    _render_metrics(window, event, paths["metrics"])
    _render_animation(window, event, paths["animation"], focus_extent_m, scale_bar_m, map_overlay)
    _write_asset_manifest(paths["manifest"], event)
    return paths


def assets_match_event(asset_dir: Path, event: dict[str, Any]) -> bool:
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("event_identity") == event_identity(event)
        and manifest.get("renderer_version") == ASSET_RENDERER_VERSION
        and manifest.get("camera_mode") == "event_subject_follow"
        and manifest.get("event_subject_track_id") == event_subject_track_id(event)
    )


def actor_shape_spec(object_type: str) -> dict[str, float | str]:
    specs: dict[str, dict[str, float | str]] = {
        "vehicle": {"shape": "rectangle", "length": 4.5, "width": 2.0},
        "bus": {"shape": "rectangle", "length": 12.0, "width": 2.6},
        "motorcyclist": {"shape": "thin_rectangle", "length": 2.4, "width": 0.8},
        "cyclist": {"shape": "thin_rectangle", "length": 1.8, "width": 0.6},
        "pedestrian": {"shape": "circle", "radius": 0.35},
        "riderless_bicycle": {"shape": "thin_rectangle", "length": 1.8, "width": 0.5},
        "construction": {"shape": "rectangle", "length": 2.0, "width": 2.0},
        "static": {"shape": "rectangle", "length": 1.0, "width": 1.0},
        "background": {"shape": "circle", "radius": 0.2},
        "unknown": {"shape": "circle", "radius": 0.25},
    }
    return specs.get(object_type, {"shape": "rectangle", "length": 4.5, "width": 2.0})


def compute_focus_bounds(
    df: pd.DataFrame,
    event: dict[str, Any],
    extent_m: float = 50.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    focus_ids = {str(event["track_id"])}
    if event.get("actor_id"):
        focus_ids.add(str(event["actor_id"]))
    focus = df[df["track_id"].astype(str).isin(focus_ids)]
    if focus.empty:
        focus = df
    start_focus = _focus_at_start_timestep(focus, event)
    center_source = start_focus if not start_focus.empty else focus
    center_x = float((center_source["position_x"].min() + center_source["position_x"].max()) / 2)
    center_y = float((center_source["position_y"].min() + center_source["position_y"].max()) / 2)
    half = extent_m / 2
    return (center_x - half, center_x + half), (center_y - half, center_y + half)


def compute_frame_focus_bounds(
    df: pd.DataFrame,
    event: dict[str, Any],
    timestep: int,
    extent_m: float = 50.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    row = _track_row_at_timestep(df, event_subject_track_id(event), timestep)
    if row is None:
        row = _track_row_at_timestep(df, str(event["track_id"]), timestep)
    if row is None and event.get("actor_id"):
        row = _track_row_at_timestep(df, str(event["actor_id"]), timestep)
    if row is None:
        return compute_focus_bounds(df, event, extent_m=extent_m)

    half = extent_m / 2
    center_x = float(row["position_x"])
    center_y = float(row["position_y"])
    return (center_x - half, center_x + half), (center_y - half, center_y + half)


def event_subject_track_id(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", ""))
    if event_type in {"cut_in", "stopped_vehicle_ahead"} and event.get("actor_id"):
        return str(event["actor_id"])
    return str(event.get("track_id", ""))


def _track_row_at_timestep(
    df: pd.DataFrame,
    track_id: str,
    timestep: int,
) -> pd.Series | None:
    rows = df[
        (df["track_id"].astype(str) == track_id)
        & (df["timestep"].astype(int) == int(timestep))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _focus_at_start_timestep(
    focus: pd.DataFrame,
    event: dict[str, Any],
) -> pd.DataFrame:
    if "start_timestep" not in event or focus.empty or "timestep" not in focus.columns:
        return pd.DataFrame()
    return focus[focus["timestep"].astype(int) == int(event["start_timestep"])]


def _write_asset_manifest(path: Path, event: dict[str, Any]) -> None:
    manifest = {
        "renderer_version": ASSET_RENDERER_VERSION,
        "camera_mode": "event_subject_follow",
        "review_id": str(event.get("review_id", "")),
        "event_identity": event_identity(event),
        "event_subject_track_id": event_subject_track_id(event),
        "scenario_id": str(event.get("scenario_id", "")),
        "event_type": str(event.get("event_type", "")),
        "track_id": str(event.get("track_id", "")),
        "actor_id": str(event.get("actor_id", "")),
        "start_timestep": event.get("start_timestep"),
        "end_timestep": event.get("end_timestep"),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def frame_metric_text(metrics: pd.DataFrame, event: dict[str, Any], timestep: int) -> str:
    event_type = str(event["event_type"])
    if metrics.empty:
        return "metrics unavailable"
    nearest_index = (metrics["timestep"] - timestep).abs().idxmin()
    row = metrics.loc[nearest_index]
    if event_type == "hard_braking":
        category = str(row.get("object_type", row.get("focal_object_type", "unknown")))
        return f"focal={category}"
    front_distance = row.get("front_projection_m", row.get("longitudinal_offset_m", row.get("distance_m")))
    ttc = row["ttc_s"]
    ttc_text = "inf" if not np.isfinite(ttc) else f"{ttc:.2f}s"
    base = (
        f"front={float(front_distance):.2f}m\n"
        f"TTC={ttc_text}\n"
        f"lat={row['lateral_offset_m']:.2f}m"
    )
    categories = role_category_text(metrics, timestep)
    if categories:
        base += f"\n{categories}"
    return base


def role_category_text(metrics: pd.DataFrame, timestep: int) -> str:
    if metrics.empty:
        return ""
    nearest_index = (metrics["timestep"] - timestep).abs().idxmin()
    row = metrics.loc[nearest_index]
    focal_type = row.get("focal_object_type", row.get("object_type", None))
    actor_type = row.get("actor_object_type", None)
    lines = []
    if focal_type is not None:
        lines.append(f"focal={focal_type}")
    if actor_type is not None:
        lines.append(f"primary={actor_type}")
    return "\n".join(lines)


def role_label_text(row: pd.Series) -> str:
    speed = float(row.get("speed_mps", np.nan))
    accel = float(row.get("accel_mps2", np.nan))
    if np.isfinite(speed) and np.isfinite(accel):
        return f"v={speed:.2f}m/s\na={accel:.2f}m/s^2"
    return ""


def _window_df(df: pd.DataFrame, start: int, end: int, context_steps: int) -> pd.DataFrame:
    lower = max(int(df["timestep"].min()), start - context_steps)
    upper = min(int(df["timestep"].max()), end + context_steps)
    return df[(df["timestep"] >= lower) & (df["timestep"] <= upper)].copy()


def _render_trajectory_window(
    df: pd.DataFrame,
    event: dict[str, Any],
    output_path: Path,
    focus_extent_m: float,
    scale_bar_m: float,
    map_overlay: MapOverlay | None,
) -> None:
    focal_id = str(event["track_id"])
    actor_id = str(event.get("actor_id", ""))
    subject_id = event_subject_track_id(event)
    start = int(event["start_timestep"])
    end = int(event["end_timestep"])

    fig, ax = plt.subplots(figsize=(8, 8))
    xlim, ylim = compute_focus_bounds(df, event, extent_m=focus_extent_m)
    draw_map_overlay(ax, map_overlay)
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("timestep")
        track_id_str = str(track_id)
        if track_id_str == subject_id:
            color, width, alpha, label = "tab:red", 3, 1.0, "event subject"
        elif track_id_str == focal_id:
            color, width, alpha, label = "tab:green", 2.5, 0.9, "focal"
        elif actor_id and track_id_str == actor_id:
            color, width, alpha, label = "tab:blue", 2.5, 0.9, "primary actor"
        else:
            color, width, alpha, label = "0.6", 1, 0.35, None
        ax.plot(group["position_x"], group["position_y"], color=color, linewidth=width, alpha=alpha)
        if label:
            event_points = group[group["timestep"].isin([start, end])]
            ax.scatter(event_points["position_x"], event_points["position_y"], color=color, s=60, label=label)

    enriched = _add_actor_kinematics(df)
    start_frame = enriched[enriched["timestep"] == start]
    for _, row in start_frame.iterrows():
        _draw_actor_box(ax, row, event)
        _draw_role_label(ax, row, event)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    _draw_scale_bar(ax, scale_bar_m)
    ax.set_xlabel("x / meters")
    ax.set_ylabel("y / meters")
    ax.set_title(f"{event['review_id']} {event['event_type']} window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_metrics(df: pd.DataFrame, event: dict[str, Any], output_path: Path) -> None:
    start = int(event["start_timestep"])
    end = int(event["end_timestep"])
    focal_id = str(event["track_id"])
    event_type = str(event["event_type"])

    if event_type == "hard_braking":
        track = df[df["track_id"].astype(str) == focal_id].sort_values("timestep")
        metrics = add_acceleration(add_speed(track))
        fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(metrics["timestep"], metrics["speed_mps"], color="tab:green")
        axes[0].set_ylabel("vel speed")
        axes[1].plot(metrics["timestep"], metrics["position_speed_mps"], color="tab:blue")
        axes[1].set_ylabel("pos speed")
        axes[2].plot(metrics["timestep"], metrics["accel_mps2"], color="tab:red")
        axes[2].axhline(-3.0, color="0.4", linestyle="--", linewidth=1)
        axes[2].set_ylabel("vel accel")
        axes[3].plot(metrics["timestep"], metrics["position_accel_smooth_mps2"], color="tab:purple")
        axes[3].axhline(-3.0, color="0.4", linestyle="--", linewidth=1)
        axes[3].set_ylabel("pos accel")
    else:
        rel = compute_relative_motion(df)
        if "actor_id" in event:
            rel = rel[rel["actor_id"].astype(str) == str(event["actor_id"])]
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        front_distance = rel.get("front_projection_m", rel["longitudinal_offset_m"])
        axes[0].plot(rel["timestep"], front_distance, color="tab:purple")
        axes[0].axhline(10.0, color="0.4", linestyle="--", linewidth=1)
        axes[0].set_ylabel("front m")
        finite_ttc = rel["ttc_s"].replace([np.inf], np.nan)
        axes[1].plot(rel["timestep"], finite_ttc, color="tab:orange")
        axes[1].axhline(2.0, color="0.4", linestyle="--", linewidth=1)
        axes[1].set_ylabel("TTC s")
        axes[2].plot(rel["timestep"], rel["lateral_offset_m"], color="tab:blue")
        axes[2].axhline(2.0, color="0.4", linestyle="--", linewidth=1)
        axes[2].axhline(-2.0, color="0.4", linestyle="--", linewidth=1)
        axes[2].set_ylabel("lat m")

    for axis in axes:
        axis.axvspan(start, end, color="gold", alpha=0.2)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"{event['review_id']} evidence metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_animation(
    df: pd.DataFrame,
    event: dict[str, Any],
    output_path: Path,
    focus_extent_m: float,
    scale_bar_m: float,
    map_overlay: MapOverlay | None,
) -> None:
    focal_id = str(event["track_id"])
    actor_id = str(event.get("actor_id", ""))
    subject_id = event_subject_track_id(event)
    start = int(event["start_timestep"])
    end = int(event["end_timestep"])
    timesteps = sorted(df["timestep"].unique())
    metrics = _frame_metrics(df, event)
    enriched = _add_actor_kinematics(df)

    fig, ax = plt.subplots(figsize=(7, 7))
    initial_xlim, initial_ylim = compute_frame_focus_bounds(
        df,
        event,
        int(timesteps[0]),
        extent_m=focus_extent_m,
    )
    ax.set_xlim(*initial_xlim)
    ax.set_ylim(*initial_ylim)
    ax.set_aspect("equal")

    def update(frame_index: int) -> None:
        timestep = timesteps[frame_index]
        xlim, ylim = compute_frame_focus_bounds(
            df,
            event,
            int(timestep),
            extent_m=focus_extent_m,
        )
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{event['review_id']} t={timestep}")
        draw_map_overlay(ax, map_overlay)

        past = df[df["timestep"] <= timestep]
        current = enriched[enriched["timestep"] == timestep]
        for track_id, group in past.groupby("track_id"):
            track_id_str = str(track_id)
            if track_id_str == subject_id:
                color, width, alpha = "tab:red", 2.8, 1.0
            elif track_id_str == focal_id:
                color, width, alpha = "tab:green", 2.2, 0.9
            elif actor_id and track_id_str == actor_id:
                color, width, alpha = "tab:blue", 2.2, 0.9
            else:
                color, width, alpha = "0.6", 0.8, 0.25
            group = group.sort_values("timestep")
            ax.plot(
                group["position_x"],
                group["position_y"],
                color=color,
                linewidth=width,
                alpha=alpha,
                zorder=3,
            )

        for _, row in current.iterrows():
            _draw_actor_box(ax, row, event)
            _draw_role_label(ax, row, event)

        if start <= timestep <= end:
            ax.text(0.02, 0.95, "EVENT WINDOW", transform=ax.transAxes, color="darkred")
        ax.text(
            0.98,
            0.98,
            frame_metric_text(metrics, event, int(timestep)),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        _draw_scale_bar(ax, scale_bar_m)

    animation = FuncAnimation(fig, update, frames=len(timesteps), interval=120, repeat=True)
    animation.save(output_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def _frame_metrics(df: pd.DataFrame, event: dict[str, Any]) -> pd.DataFrame:
    if event["event_type"] == "hard_braking":
        track = df[df["track_id"].astype(str) == str(event["track_id"])].sort_values("timestep")
        return add_acceleration(add_speed(track))
    rel = compute_relative_motion(df)
    if event.get("actor_id"):
        rel = rel[rel["actor_id"].astype(str) == str(event["actor_id"])]
    return _add_role_kinematics_to_relative_metrics(df, rel, event)


def _add_actor_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    return add_acceleration(add_speed(df))


def _add_role_kinematics_to_relative_metrics(
    df: pd.DataFrame,
    rel: pd.DataFrame,
    event: dict[str, Any],
) -> pd.DataFrame:
    if rel.empty:
        return rel
    enriched = _add_actor_kinematics(df)
    focal_id = str(event["track_id"])
    actor_id = str(event.get("actor_id", ""))
    focal = enriched[enriched["track_id"].astype(str) == focal_id][
        ["timestep", "speed_mps", "accel_mps2", "object_type"]
    ].rename(
        columns={
            "speed_mps": "focal_speed_mps",
            "accel_mps2": "focal_accel_mps2",
            "object_type": "focal_object_type",
        }
    )
    result = rel.merge(focal, on="timestep", how="left")
    if actor_id:
        actor = enriched[enriched["track_id"].astype(str) == actor_id][
            ["timestep", "speed_mps", "accel_mps2", "object_type"]
        ].rename(
            columns={
                "speed_mps": "actor_speed_mps",
                "accel_mps2": "actor_accel_mps2",
                "object_type": "actor_object_type",
            }
        )
        result = result.merge(actor, on="timestep", how="left")
    return result


def _draw_actor_box(ax: plt.Axes, row: pd.Series, event: dict[str, Any]) -> None:
    track_id = str(row["track_id"])
    focal_id = str(event["track_id"])
    actor_id = str(event.get("actor_id", ""))
    subject_id = event_subject_track_id(event)
    if track_id == subject_id:
        facecolor, edgecolor, alpha = "tab:red", "darkred", 0.98
    elif track_id == focal_id:
        facecolor, edgecolor, alpha = "tab:green", "darkgreen", 0.9
    elif actor_id and track_id == actor_id:
        facecolor, edgecolor, alpha = "tab:blue", "navy", 0.95
    else:
        facecolor, edgecolor, alpha = "0.75", "0.4", 0.55

    object_type = str(row.get("object_type", "vehicle"))
    spec = actor_shape_spec(object_type)
    x = float(row["position_x"])
    y = float(row["position_y"])
    heading = float(row.get("heading", 0.0))

    if spec["shape"] == "circle":
        radius = float(spec["radius"])
        ax.scatter(
            x,
            y,
            color=facecolor,
            edgecolor=edgecolor,
            s=max(20, radius * 220),
            alpha=alpha,
            zorder=5,
        )
        return

    length = float(spec["length"])
    width = float(spec["width"])
    rect = Rectangle(
        (-length / 2, -width / 2),
        length,
        width,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        zorder=5,
    )
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)


def _draw_role_label(ax: plt.Axes, row: pd.Series, event: dict[str, Any]) -> None:
    track_id = str(row["track_id"])
    focal_id = str(event["track_id"])
    actor_id = str(event.get("actor_id", ""))
    subject_id = event_subject_track_id(event)
    if not (
        track_id == subject_id
        or track_id == focal_id
        or (actor_id and track_id == actor_id)
    ):
        return
    ax.text(
        float(row["position_x"]) + 1.2,
        float(row["position_y"]) + 1.2,
        role_label_text(row),
        fontsize=8,
        color="black",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def _draw_scale_bar(ax: plt.Axes, scale_bar_m: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    margin_x = (x1 - x0) * 0.06
    margin_y = (y1 - y0) * 0.08
    start_x = x0 + margin_x
    start_y = y0 + margin_y
    ax.plot([start_x, start_x + scale_bar_m], [start_y, start_y], color="black", linewidth=3)
    ax.text(
        start_x + scale_bar_m / 2,
        start_y + margin_y * 0.25,
        f"{scale_bar_m:g} m",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
