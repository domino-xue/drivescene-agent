from pathlib import Path
import json

import pandas as pd

from drivescene.review.assets import (
    actor_shape_spec,
    assets_match_event,
    compute_frame_focus_bounds,
    compute_focus_bounds,
    event_subject_track_id,
    frame_metric_text,
    role_category_text,
    role_label_text,
    render_review_assets,
)


def test_render_review_assets_creates_static_metrics_and_animation(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 10,
            "track_id": ["focal", "front"] * 5,
            "focal_track_id": ["focal"] * 10,
            "timestep": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "position_x": [0.0, 8.0, 1.0, 8.5, 2.0, 9.0, 3.0, 9.5, 4.0, 10.0],
            "position_y": [0.0, 0.5] * 5,
            "heading": [0.0] * 10,
            "velocity_x": [10.0, 8.0] * 5,
            "velocity_y": [0.0] * 10,
            "object_type": ["vehicle"] * 10,
        }
    )
    event = {
        "review_id": "000001",
        "scenario_id": "scene-1",
        "event_type": "close_following",
        "track_id": "focal",
        "actor_id": "front",
        "start_timestep": 1,
        "end_timestep": 3,
    }

    assets = render_review_assets(df, event, tmp_path, context_steps=1)

    assert assets["trajectory_window"].exists()
    assert assets["metrics"].exists()
    assert assets["animation"].exists()
    assert assets["manifest"].exists()
    assert assets_match_event(tmp_path / "000001", event)


def test_assets_match_event_rejects_stale_review_id_directory(tmp_path: Path) -> None:
    asset_dir = tmp_path / "000001"
    asset_dir.mkdir()
    old_event = {
        "review_id": "000001",
        "scenario_id": "old-scene",
        "event_type": "cut_in",
        "track_id": "focal",
        "actor_id": "old",
        "start_timestep": 1,
        "end_timestep": 2,
    }
    new_event = {
        **old_event,
        "scenario_id": "new-scene",
        "actor_id": "new",
    }
    render_review_assets(
        pd.DataFrame(
            {
                "scenario_id": ["old-scene", "old-scene"],
                "track_id": ["focal", "old"],
                "focal_track_id": ["focal", "focal"],
                "timestep": [1, 1],
                "position_x": [0.0, 8.0],
                "position_y": [0.0, 1.0],
                "heading": [0.0, 0.0],
                "velocity_x": [10.0, 8.0],
                "velocity_y": [0.0, 0.0],
                "object_type": ["vehicle", "vehicle"],
            }
        ),
        old_event,
        tmp_path,
        context_steps=0,
    )

    assert assets_match_event(asset_dir, old_event)
    assert not assets_match_event(asset_dir, new_event)


def test_assets_match_event_rejects_old_camera_manifest(tmp_path: Path) -> None:
    asset_dir = tmp_path / "000001"
    asset_dir.mkdir()
    event = {
        "review_id": "000001",
        "scenario_id": "scene-1",
        "event_type": "cut_in",
        "track_id": "focal",
        "actor_id": "actor",
        "start_timestep": 1,
        "end_timestep": 2,
    }
    (asset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "review_id": "000001",
                "event_identity": "scene-1|cut_in|focal|actor|1|2",
            }
        ),
        encoding="utf-8",
    )

    assert not assets_match_event(asset_dir, event)


def test_render_review_assets_creates_hard_braking_dual_evidence_metrics(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["scene-1"] * 6,
            "track_id": ["focal"] * 6,
            "focal_track_id": ["focal"] * 6,
            "timestep": [0, 1, 2, 3, 4, 5],
            "position_x": [0.0, 1.0, 1.96, 2.87, 3.73, 4.55],
            "position_y": [0.0] * 6,
            "heading": [0.0] * 6,
            "velocity_x": [10.0, 9.6, 9.1, 8.6, 8.2, 8.2],
            "velocity_y": [0.0] * 6,
            "object_type": ["vehicle"] * 6,
        }
    )
    event = {
        "review_id": "000001",
        "scenario_id": "scene-1",
        "event_type": "hard_braking",
        "track_id": "focal",
        "start_timestep": 2,
        "end_timestep": 4,
    }

    assets = render_review_assets(df, event, tmp_path, context_steps=1)

    assert assets["metrics"].exists()


def test_compute_focus_bounds_uses_fixed_extent_around_focus_tracks() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["focal", "front", "other"],
            "position_x": [0.0, 10.0, 200.0],
            "position_y": [0.0, 0.0, 200.0],
        }
    )
    event = {"track_id": "focal", "actor_id": "front"}

    xlim, ylim = compute_focus_bounds(df, event, extent_m=50.0)

    assert xlim == (-20.0, 30.0)
    assert ylim == (-25.0, 25.0)


def test_compute_focus_bounds_keeps_start_frame_actor_visible_for_long_events() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["focal", "front", "focal", "front"],
            "timestep": [0, 0, 100, 100],
            "position_x": [0.0, 8.0, 200.0, 208.0],
            "position_y": [0.0, 2.0, 0.0, 0.2],
        }
    )
    event = {
        "track_id": "focal",
        "actor_id": "front",
        "start_timestep": 0,
        "end_timestep": 100,
    }

    xlim, ylim = compute_focus_bounds(df, event, extent_m=50.0)

    assert xlim[0] <= 8.0 <= xlim[1]
    assert ylim[0] <= 2.0 <= ylim[1]


def test_compute_frame_focus_bounds_follows_primary_track() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["focal", "front", "focal", "front"],
            "timestep": [0, 0, 100, 100],
            "position_x": [0.0, 8.0, 200.0, 208.0],
            "position_y": [0.0, 2.0, 10.0, 12.0],
        }
    )
    event = {"track_id": "focal", "actor_id": "front"}

    start_xlim, start_ylim = compute_frame_focus_bounds(df, event, timestep=0, extent_m=50.0)
    end_xlim, end_ylim = compute_frame_focus_bounds(df, event, timestep=100, extent_m=50.0)

    assert start_xlim == (-25.0, 25.0)
    assert start_ylim == (-25.0, 25.0)
    assert end_xlim == (175.0, 225.0)
    assert end_ylim == (-15.0, 35.0)


def test_compute_frame_focus_bounds_follows_cut_in_actor_subject() -> None:
    df = pd.DataFrame(
        {
            "track_id": ["focal", "actor", "focal", "actor"],
            "timestep": [0, 0, 10, 10],
            "position_x": [0.0, 100.0, 10.0, 120.0],
            "position_y": [0.0, 50.0, 0.0, 60.0],
        }
    )
    event = {
        "event_type": "cut_in",
        "track_id": "focal",
        "actor_id": "actor",
    }

    xlim, ylim = compute_frame_focus_bounds(df, event, timestep=10, extent_m=50.0)

    assert event_subject_track_id(event) == "actor"
    assert xlim == (95.0, 145.0)
    assert ylim == (35.0, 85.0)


def test_event_subject_track_id_uses_track_for_hard_braking() -> None:
    event = {
        "event_type": "hard_braking",
        "track_id": "braking-car",
        "actor_id": "front",
    }

    assert event_subject_track_id(event) == "braking-car"


def test_frame_metric_text_reports_close_following_values() -> None:
    rel = pd.DataFrame(
        {
            "timestep": [3],
            "actor_id": ["front"],
            "distance_m": [7.5],
            "longitudinal_offset_m": [6.75],
            "front_projection_m": [6.25],
            "ttc_s": [1.25],
            "lateral_offset_m": [0.6],
        }
    )
    event = {"event_type": "close_following", "actor_id": "front"}

    text = frame_metric_text(rel, event, timestep=3)

    assert "front=6.25m" in text
    assert "distance=7.50m" not in text
    assert "TTC=1.25s" in text
    assert "lat=0.60m" in text


def test_frame_metric_text_reports_role_categories_only() -> None:
    metrics = pd.DataFrame(
        {
            "timestep": [3],
            "actor_id": ["front"],
            "distance_m": [7.5],
            "longitudinal_offset_m": [6.75],
            "front_projection_m": [6.25],
            "ttc_s": [1.25],
            "lateral_offset_m": [0.6],
            "focal_speed_mps": [10.0],
            "focal_accel_mps2": [-1.5],
            "actor_speed_mps": [8.0],
            "actor_accel_mps2": [0.5],
            "focal_object_type": ["vehicle"],
            "actor_object_type": ["motorcyclist"],
        }
    )
    event = {"event_type": "close_following", "actor_id": "front"}

    text = frame_metric_text(metrics, event, timestep=3)

    assert "focal=vehicle" in text
    assert "primary=motorcyclist" in text
    assert "focal v=" not in text
    assert "primary v=" not in text


def test_role_label_text_follows_actor_with_speed_and_acceleration_only() -> None:
    row = pd.Series(
        {
            "track_id": "focal",
            "object_type": "vehicle",
            "speed_mps": 9.5,
            "accel_mps2": -2.0,
        }
    )

    assert role_label_text(row) == "v=9.50m/s\na=-2.00m/s^2"


def test_role_category_text_reports_focal_and_primary_categories() -> None:
    metrics = pd.DataFrame(
        {
            "timestep": [3],
            "focal_object_type": ["vehicle"],
            "actor_object_type": ["pedestrian"],
        }
    )

    assert role_category_text(metrics, timestep=3) == "focal=vehicle\nprimary=pedestrian"


def test_actor_shape_spec_uses_category_specific_realistic_dimensions() -> None:
    assert actor_shape_spec("vehicle") == {"shape": "rectangle", "length": 4.5, "width": 2.0}
    assert actor_shape_spec("bus") == {"shape": "rectangle", "length": 12.0, "width": 2.6}
    assert actor_shape_spec("motorcyclist") == {"shape": "thin_rectangle", "length": 2.4, "width": 0.8}
    assert actor_shape_spec("cyclist") == {"shape": "thin_rectangle", "length": 1.8, "width": 0.6}
    assert actor_shape_spec("pedestrian") == {"shape": "circle", "radius": 0.35}
