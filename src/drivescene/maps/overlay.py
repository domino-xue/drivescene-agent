import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


Point2D = tuple[float, float]


@dataclass(frozen=True)
class MapOverlay:
    drivable_areas: list[list[Point2D]]
    lane_centerlines: list[list[Point2D]]
    lane_boundaries: list[list[Point2D]]
    crosswalk_edges: list[list[Point2D]]


@dataclass(frozen=True)
class LaneProjection:
    lane_id: int
    distance_m: float
    signed_lateral_offset_m: float
    progress_m: float
    tangent_heading_rad: float


def load_map_overlay(path: Path) -> MapOverlay:
    data = json.loads(path.read_text(encoding="utf-8"))
    drivable_areas = [
        _points(area["area_boundary"])
        for area in data.get("drivable_areas", {}).values()
        if area.get("area_boundary")
    ]
    lane_centerlines = [
        _points(lane["centerline"])
        for lane in data.get("lane_segments", {}).values()
        if lane.get("centerline")
    ]
    lane_boundaries: list[list[Point2D]] = []
    for lane in data.get("lane_segments", {}).values():
        if lane.get("left_lane_boundary"):
            lane_boundaries.append(_points(lane["left_lane_boundary"]))
        if lane.get("right_lane_boundary"):
            lane_boundaries.append(_points(lane["right_lane_boundary"]))

    crosswalk_edges: list[list[Point2D]] = []
    for crossing in data.get("pedestrian_crossings", {}).values():
        if crossing.get("edge1"):
            crosswalk_edges.append(_points(crossing["edge1"]))
        if crossing.get("edge2"):
            crosswalk_edges.append(_points(crossing["edge2"]))

    return MapOverlay(
        drivable_areas=drivable_areas,
        lane_centerlines=lane_centerlines,
        lane_boundaries=lane_boundaries,
        crosswalk_edges=crosswalk_edges,
    )


def draw_map_overlay(ax: plt.Axes, overlay: MapOverlay | None) -> None:
    if overlay is None:
        return

    for polygon in overlay.drivable_areas:
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#d0d0d0",
                edgecolor="none",
                alpha=0.65,
                zorder=0,
            )
        )
    for boundary in overlay.lane_boundaries:
        _plot_line(ax, boundary, color="white", linewidth=1.0, alpha=0.55, zorder=1)
    for centerline in overlay.lane_centerlines:
        _plot_line(ax, centerline, color="#eeeeee", linewidth=0.7, alpha=0.55, zorder=1)
    for edge in overlay.crosswalk_edges:
        _plot_line(ax, edge, color="#f7f7f7", linewidth=1.5, alpha=0.85, zorder=1)


def nearest_lane_index(
    overlay: MapOverlay | None,
    point: Point2D,
    max_distance_m: float = 1.5,
    heading_rad: float | None = None,
    max_heading_delta_deg: float | None = None,
) -> int | None:
    if heading_rad is None or max_heading_delta_deg is None:
        if overlay is None:
            return None

        best_index: int | None = None
        best_distance = math.inf
        for index, centerline in enumerate(overlay.lane_centerlines):
            distance = _distance_to_polyline(point, centerline)
            if distance < best_distance:
                best_index = index
                best_distance = distance

        if best_distance > max_distance_m:
            return None
        return best_index

    projection = project_to_nearest_lane(
        overlay,
        point,
        max_distance_m=max_distance_m,
        heading_rad=heading_rad,
        max_heading_delta_deg=max_heading_delta_deg,
    )
    if projection is None:
        return None
    return projection.lane_id


def project_to_nearest_lane(
    overlay: MapOverlay | None,
    point: Point2D,
    max_distance_m: float = 1.5,
    heading_rad: float | None = None,
    max_heading_delta_deg: float | None = None,
) -> LaneProjection | None:
    if overlay is None:
        return None

    best_projection: LaneProjection | None = None
    candidates: list[tuple[float, int, list[Point2D]]] = []
    for index, centerline in enumerate(overlay.lane_centerlines):
        distance = _distance_to_polyline(point, centerline)
        if distance <= max_distance_m:
            candidates.append((distance, index, centerline))

    for _, index, centerline in sorted(candidates):
        projection = _project_to_polyline(index, point, centerline)
        if projection is None:
            continue
        if (
            heading_rad is not None
            and max_heading_delta_deg is not None
            and _axis_heading_delta_deg(heading_rad, projection.tangent_heading_rad)
            > max_heading_delta_deg
        ):
            continue
        if best_projection is None or projection.distance_m < best_projection.distance_m:
            best_projection = projection

    if best_projection is None or best_projection.distance_m > max_distance_m:
        return None
    return best_projection


def _points(raw: list[dict[str, Any]]) -> list[Point2D]:
    return [(float(point["x"]), float(point["y"])) for point in raw]


def _plot_line(
    ax: plt.Axes,
    points: list[Point2D],
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    if len(points) < 2:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _distance_to_polyline(point: Point2D, polyline: list[Point2D]) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return math.dist(point, polyline[0])
    return min(_distance_to_segment(point, start, end) for start, end in zip(polyline, polyline[1:]))


def _distance_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.dist(point, start)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (x1 + t * dx, y1 + t * dy)
    return math.dist(point, closest)


def _axis_heading_delta_deg(first_rad: float, second_rad: float) -> float:
    delta = abs(_wrap_to_pi(first_rad - second_rad))
    return math.degrees(min(delta, math.pi - delta))


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def _project_to_polyline(
    lane_id: int,
    point: Point2D,
    polyline: list[Point2D],
) -> LaneProjection | None:
    if not polyline:
        return None
    if len(polyline) == 1:
        distance = math.dist(point, polyline[0])
        return LaneProjection(
            lane_id=lane_id,
            distance_m=distance,
            signed_lateral_offset_m=distance,
            progress_m=0.0,
            tangent_heading_rad=0.0,
        )

    best: LaneProjection | None = None
    progress_at_segment_start = 0.0
    for start, end in zip(polyline, polyline[1:]):
        projection = _project_to_segment_detail(
            lane_id,
            point,
            start,
            end,
            progress_at_segment_start,
        )
        if best is None or projection.distance_m < best.distance_m:
            best = projection
        progress_at_segment_start += math.dist(start, end)
    return best


def _project_to_segment_detail(
    lane_id: int,
    point: Point2D,
    start: Point2D,
    end: Point2D,
    progress_at_segment_start: float,
) -> LaneProjection:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        distance = math.dist(point, start)
        return LaneProjection(
            lane_id=lane_id,
            distance_m=distance,
            signed_lateral_offset_m=distance,
            progress_m=progress_at_segment_start,
            tangent_heading_rad=0.0,
        )

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (x1 + t * dx, y1 + t * dy)
    distance = math.dist(point, closest)
    length = math.sqrt(length_sq)
    cross = dx * (py - closest[1]) - dy * (px - closest[0])
    signed_offset = math.copysign(distance, cross) if distance > 1e-9 else 0.0
    return LaneProjection(
        lane_id=lane_id,
        distance_m=distance,
        signed_lateral_offset_m=signed_offset,
        progress_m=progress_at_segment_start + t * length,
        tangent_heading_rad=math.atan2(dy, dx),
    )
