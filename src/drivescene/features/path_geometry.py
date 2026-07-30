from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


Point2D = tuple[float, float]


@dataclass(frozen=True)
class PathCorridorMatch:
    lateral_error_m: float
    path_progress_m: float
    on_path: bool


def heading_delta_deg(headings: Iterable[float]) -> float:
    values = np.asarray(list(headings), dtype=float)
    if len(values) < 2:
        return 0.0
    unwrapped = np.unwrap(values)
    return abs(float(unwrapped[-1] - unwrapped[0]) * 180.0 / math.pi)


def path_length_m(position_x: Iterable[float], position_y: Iterable[float]) -> float:
    xs = np.asarray(list(position_x), dtype=float)
    ys = np.asarray(list(position_y), dtype=float)
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum())


def trajectory_curvature_rad_per_m(
    headings: Iterable[float],
    position_x: Iterable[float],
    position_y: Iterable[float],
) -> float:
    length = path_length_m(position_x, position_y)
    if length <= 1e-6:
        return 0.0
    return math.radians(heading_delta_deg(headings)) / length


def future_path_match(
    focal_track: pd.DataFrame,
    timestep: int,
    point: Point2D,
    lateral_threshold_m: float = 1.5,
    lookahead_distance_m: float = 35.0,
    extension_distance_m: float = 35.0,
    min_progress_m: float = 0.1,
    max_extension_heading_delta_deg: float = 15.0,
) -> PathCorridorMatch:
    points = _future_path_points(
        focal_track,
        timestep,
        lookahead_distance_m,
        extension_distance_m,
        max_extension_heading_delta_deg,
    )
    lateral_error, progress = _nearest_polyline_distance_and_progress(point, points)
    return PathCorridorMatch(
        lateral_error_m=lateral_error,
        path_progress_m=progress,
        on_path=lateral_error <= lateral_threshold_m and progress >= min_progress_m,
    )


def _future_path_points(
    focal_track: pd.DataFrame,
    timestep: int,
    lookahead_distance_m: float,
    extension_distance_m: float,
    max_extension_heading_delta_deg: float,
) -> list[Point2D]:
    if focal_track.empty:
        return []

    rows = focal_track.sort_values("timestep").reset_index(drop=True)
    future = rows[rows["timestep"].astype(int) >= int(timestep)].copy()
    if future.empty:
        future = rows.tail(1).copy()

    points: list[Point2D] = []
    used_indices: list[int] = []
    cumulative = 0.0
    previous: Point2D | None = None
    for index, row in future.iterrows():
        point = (float(row["position_x"]), float(row["position_y"]))
        if previous is not None:
            cumulative += math.dist(previous, point)
        points.append(point)
        used_indices.append(int(index))
        previous = point
        if cumulative >= lookahead_distance_m:
            break

    if not points:
        return []

    used = rows.loc[used_indices] if used_indices else future.head(1)
    can_extend = (
        cumulative < lookahead_distance_m
        and heading_delta_deg(used["heading"].astype(float).tolist())
        <= max_extension_heading_delta_deg
    )
    if can_extend:
        last = used.iloc[-1]
        dx, dy = _direction_from_row(last)
        extend_by = max(extension_distance_m, lookahead_distance_m - cumulative)
        points.append((points[-1][0] + dx * extend_by, points[-1][1] + dy * extend_by))

    return points


def _direction_from_row(row: pd.Series) -> Point2D:
    vx = float(row.get("velocity_x", 0.0))
    vy = float(row.get("velocity_y", 0.0))
    speed = math.hypot(vx, vy)
    if speed > 1e-6:
        return vx / speed, vy / speed
    heading = float(row.get("heading", 0.0))
    return math.cos(heading), math.sin(heading)


def _nearest_polyline_distance_and_progress(
    point: Point2D,
    polyline: list[Point2D],
) -> tuple[float, float]:
    if not polyline:
        return math.inf, 0.0
    if len(polyline) == 1:
        return math.dist(point, polyline[0]), 0.0

    best_distance = math.inf
    best_progress = 0.0
    progress_at_segment_start = 0.0
    for start, end in zip(polyline, polyline[1:]):
        distance, progress_on_segment, segment_length = _project_to_segment(point, start, end)
        if distance < best_distance:
            best_distance = distance
            best_progress = progress_at_segment_start + progress_on_segment
        progress_at_segment_start += segment_length
    return best_distance, best_progress


def _project_to_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> tuple[float, float, float]:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    segment_length_sq = dx * dx + dy * dy
    if segment_length_sq <= 1e-12:
        return math.dist(point, start), 0.0, 0.0

    t = ((px - x1) * dx + (py - y1) * dy) / segment_length_sq
    t = max(0.0, min(1.0, t))
    closest = (x1 + t * dx, y1 + t * dy)
    segment_length = math.sqrt(segment_length_sq)
    return math.dist(point, closest), t * segment_length, segment_length
