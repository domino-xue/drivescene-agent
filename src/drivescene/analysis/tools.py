from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from drivescene.data.loader import get_actor_track, load_scenario
from drivescene.detectors.close_following import detect_close_following as _detect_close_following
from drivescene.detectors.cut_in import detect_cut_in as _detect_cut_in
from drivescene.detectors.hard_braking import detect_hard_braking as _detect_hard_braking
from drivescene.detectors.lane_change import detect_lane_change as _detect_lane_change
from drivescene.detectors.stopped_vehicle_ahead import (
    detect_stopped_vehicle_ahead as _detect_stopped_vehicle_ahead,
)
from drivescene.features.kinematics import add_acceleration, add_speed
from drivescene.features.relative_motion import compute_relative_motion as _compute_relative_motion
from drivescene.maps.overlay import load_map_overlay
from drivescene.ops.contracts import error_item, success_item, tool_result


class AnalysisTools:
    def __init__(self, scenario_root: Path | str = Path("data/val")):
        self.scenario_root = Path(scenario_root)

    def run_analysis(
        self,
        operation: str,
        event_type: str | None = None,
        scenario_ids: list[str] | None = None,
        scenario_paths: list[Path | str] | None = None,
        scenario_glob: str | None = None,
        track_id: str | int | None = None,
        actor_id: str | int | None = None,
        window: dict[str, int] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            inputs = self._resolve_inputs(scenario_ids, scenario_paths, scenario_glob)
            if operation not in {
                "detect_events",
                "compute_track_kinematics",
                "compute_relative_motion",
                "compare_velocity_position_evidence",
            }:
                raise ValueError(f"Unknown analysis operation: {operation}")
        except ValueError as error:
            return _validation_error_result(operation, error)

        items: list[dict[str, Any]] = []
        for input_ref in inputs:
            try:
                df, scenario_path = self._load_input(input_ref)
                result = self._run_one(
                    operation=operation,
                    df=df,
                    scenario_path=scenario_path,
                    event_type=event_type,
                    track_id=track_id,
                    actor_id=actor_id,
                    window=window,
                    parameters=parameters or {},
                )
                items.append(success_item(str(input_ref), result))
            except (FileNotFoundError, KeyError, ValueError) as error:
                items.append(_error_item(input_ref, error))
        return _batch_result(operation, items)

    def _resolve_inputs(
        self,
        scenario_ids: list[str] | None,
        scenario_paths: list[Path | str] | None,
        scenario_glob: str | None,
    ) -> list[str | Path]:
        sources = [
            scenario_ids is not None,
            scenario_paths is not None,
            scenario_glob is not None,
        ]
        if sum(sources) != 1:
            raise ValueError("Provide exactly one of scenario_ids, scenario_paths, or scenario_glob")
        if scenario_ids is not None:
            return [str(scenario_id) for scenario_id in scenario_ids]
        if scenario_paths is not None:
            return [Path(path) for path in scenario_paths]
        return [Path(path) for path in sorted(glob(str(scenario_glob)))]

    def _load_input(self, input_ref: str | Path) -> tuple[pd.DataFrame, Path]:
        if isinstance(input_ref, Path):
            if not input_ref.exists():
                raise FileNotFoundError(f"Scenario parquet not found: {input_ref}")
            return load_scenario(input_ref), input_ref
        return self._load_scenario(input_ref)

    def _load_scenario(self, scenario_id: str) -> tuple[pd.DataFrame, Path]:
        path = self.scenario_root / scenario_id / f"scenario_{scenario_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Scenario parquet not found: {path}")
        return load_scenario(path), path

    def _run_one(
        self,
        operation: str,
        df: pd.DataFrame,
        scenario_path: Path | None,
        event_type: str | None,
        track_id: str | int | None,
        actor_id: str | int | None,
        window: dict[str, int] | None,
        parameters: dict[str, Any],
    ) -> Any:
        if operation == "detect_events":
            return _json_records(_detect_events(df, event_type, parameters, scenario_path))
        if operation == "compute_track_kinematics":
            if track_id is None:
                raise ValueError("track_id is required for compute_track_kinematics")
            return _df_records(_compute_track_kinematics(df, track_id))
        if operation == "compute_relative_motion":
            return _df_records(_compute_relative_motion_filtered(df, actor_id))
        if operation == "compare_velocity_position_evidence":
            if track_id is None:
                raise ValueError("track_id is required for compare_velocity_position_evidence")
            if window is None:
                raise ValueError("window is required for compare_velocity_position_evidence")
            return _compare_velocity_position_evidence(
                df,
                track_id,
                int(window["start_timestep"]),
                int(window["end_timestep"]),
                float(parameters.get("accel_threshold_mps2", -3.0)),
            )
        raise ValueError(f"Unknown analysis operation: {operation}")


def _detect_events(
    df: pd.DataFrame,
    event_type: str | None,
    parameters: dict[str, Any],
    scenario_path: Path | None = None,
) -> list[dict[str, Any]]:
    if event_type == "hard_braking":
        return _detect_hard_braking(df, **parameters)
    if event_type == "close_following":
        return _detect_close_following(
            df,
            **_parameters_with_map_overlay(parameters, scenario_path),
        )
    if event_type == "cut_in":
        return _detect_cut_in(
            df,
            **_parameters_with_map_overlay(parameters, scenario_path),
        )
    if event_type == "stopped_vehicle_ahead":
        return _detect_stopped_vehicle_ahead(df, **parameters)
    if event_type == "lane_change":
        return _detect_lane_change(df, **parameters)
    raise ValueError(
        "event_type must be one of: hard_braking, close_following, cut_in, "
        "stopped_vehicle_ahead, lane_change"
    )


def _parameters_with_map_overlay(
    parameters: dict[str, Any],
    scenario_path: Path | None,
) -> dict[str, Any]:
    if "map_overlay" in parameters or scenario_path is None:
        return parameters
    map_path = scenario_path.parent / f"log_map_archive_{_scenario_id_from_path(scenario_path)}.json"
    if not map_path.exists():
        return parameters
    return {**parameters, "map_overlay": load_map_overlay(map_path)}


def _scenario_id_from_path(scenario_path: Path) -> str:
    stem = scenario_path.stem
    return stem.removeprefix("scenario_")


def _compute_track_kinematics(df: pd.DataFrame, track_id: str | int) -> pd.DataFrame:
    track = get_actor_track(df, track_id)
    return add_acceleration(add_speed(track))


def _compute_relative_motion_filtered(
    df: pd.DataFrame,
    actor_id: str | int | None = None,
) -> pd.DataFrame:
    rel = _compute_relative_motion(df)
    if actor_id is not None and not rel.empty:
        rel = rel[rel["actor_id"].astype(str) == str(actor_id)]
    return rel


def _compare_velocity_position_evidence(
    df: pd.DataFrame,
    track_id: str | int,
    start_timestep: int,
    end_timestep: int,
    accel_threshold_mps2: float,
) -> dict[str, Any]:
    metrics = _compute_track_kinematics(df, track_id)
    window = metrics[
        (metrics["timestep"] >= start_timestep)
        & (metrics["timestep"] <= end_timestep)
    ]
    if window.empty:
        raise ValueError("No track rows found in the requested timestep window")

    min_velocity_accel = float(window["accel_mps2"].min())
    min_position_accel = float(window["position_accel_smooth_mps2"].min())
    scenario_id = str(window["scenario_id"].iloc[0]) if "scenario_id" in window.columns else ""
    return {
        "scenario_id": scenario_id,
        "track_id": str(track_id),
        "start_timestep": int(start_timestep),
        "end_timestep": int(end_timestep),
        "min_velocity_acceleration_mps2": min_velocity_accel,
        "min_position_acceleration_mps2": min_position_accel,
        "supports_velocity_braking": bool(min_velocity_accel <= accel_threshold_mps2),
        "supports_position_braking": bool(min_position_accel <= accel_threshold_mps2),
        "supports_hard_braking": bool(
            min_velocity_accel <= accel_threshold_mps2
            and min_position_accel <= accel_threshold_mps2
        ),
    }


def _batch_result(operation: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return tool_result(operation, items)


def _validation_error_result(operation: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "operation": operation,
        "summary": {"total": 0, "succeeded": 0, "failed": 1},
        "items": [_error_item(None, error)],
    }


def _error_item(input_ref: Any, error: Exception) -> dict[str, Any]:
    return error_item(None if input_ref is None else str(input_ref), error)


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_records(df.where(pd.notnull(df), None).to_dict(orient="records"))


def _json_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_json_value(row) for row in rows]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
