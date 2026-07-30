from pathlib import Path

import pandas as pd

from drivescene.evidence.tools import EvidenceStore


def test_evidence_store_queries_events_with_filters(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "scenario_id": "scene-1",
                "event_type": "hard_braking",
                "city": "austin",
                "is_valid_event": True,
                "severity": "high",
                "min_velocity_acceleration_mps2": -8.0,
                "min_ttc_s": None,
            },
            {
                "review_id": "000002",
                "scenario_id": "scene-2",
                "event_type": "close_following",
                "city": "miami",
                "is_valid_event": False,
                "severity": None,
                "min_velocity_acceleration_mps2": None,
                "min_ttc_s": 0.7,
            },
        ]
    ).to_parquet(index_path)

    result = EvidenceStore(index_path).query_index(
        target="events",
        operation="search",
        filters={"event_type": "hard_braking", "is_valid_event": True, "city": "austin"},
        sort_by="min_velocity_acceleration_mps2",
        ascending=True,
    )

    assert result["ok"] is True
    assert [item["result"]["review_id"] for item in result["items"]] == ["000001"]


def test_evidence_store_returns_event_details_and_evidence_bundles(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    animation = tmp_path / "review_assets" / "000001" / "animation.gif"
    metrics = tmp_path / "review_assets" / "000001" / "metrics.png"
    trajectory = tmp_path / "review_assets" / "000001" / "trajectory_window.png"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    metrics.write_bytes(b"png")
    trajectory.write_bytes(b"png")
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "scenario_id": "scene-1",
                "event_type": "hard_braking",
                "track_id": "focal",
                "actor_id": "",
                "start_timestep": 10,
                "end_timestep": 12,
                "animation_path": str(animation),
                "metrics_path": str(metrics),
                "trajectory_path": str(trajectory),
                "is_valid_event": True,
            }
        ]
    ).to_parquet(index_path)

    store = EvidenceStore(index_path)
    detail = store.query_index(target="events", operation="detail", review_ids=["000001"])
    bundle = store.query_index(target="events", operation="evidence", review_ids=["000001"])

    assert detail["items"][0]["result"]["scenario_id"] == "scene-1"
    assert bundle["items"][0]["result"]["animation_path"] == str(animation)
    assert bundle["items"][0]["result"]["has_animation"] is True
    assert bundle["items"][0]["result"]["has_metrics"] is True
    assert bundle["items"][0]["result"]["has_trajectory"] is True


def test_evidence_store_evidence_batches_partial_missing_review_ids(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    animation = tmp_path / "review_assets" / "000001" / "animation.gif"
    animation.parent.mkdir(parents=True)
    animation.write_bytes(b"gif")
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "animation_path": str(animation),
            }
        ]
    ).to_parquet(index_path)

    result = EvidenceStore(index_path).query_index(
        target="events",
        operation="evidence",
        review_ids=["000001", "missing"],
    )

    assert result["ok"] is False
    assert result["operation"] == "evidence"
    assert result["summary"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert result["items"][0]["input"] == {"review_id": "000001"}
    assert result["items"][0]["ok"] is True
    assert result["items"][0]["result"]["animation_path"] == str(animation)
    assert result["items"][1]["input"] == {"review_id": "missing"}
    assert result["items"][1]["ok"] is False
    assert result["items"][1]["error_type"] == "KeyError"


def test_evidence_store_summarizes_events(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    pd.DataFrame(
        [
            {"review_id": "1", "event_type": "hard_braking", "is_valid_event": True},
            {"review_id": "2", "event_type": "hard_braking", "is_valid_event": False},
            {"review_id": "3", "event_type": "close_following", "is_valid_event": True},
        ]
    ).to_parquet(index_path)

    result = EvidenceStore(index_path).query_index(target="events", operation="summary")
    summary = result["items"][0]["result"]

    assert result["ok"] is True
    assert summary["num_events"] == 3
    assert summary["num_valid"] == 2
    assert summary["valid_rate"] == 2 / 3
    assert summary["event_type_counts"] == {"hard_braking": 2, "close_following": 1}


def test_evidence_store_summarizes_events_with_filters(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "1",
                "event_type": "hard_braking",
                "review_status": "reviewed",
                "is_valid_event": True,
            },
            {
                "review_id": "2",
                "event_type": "hard_braking",
                "review_status": "unreviewed",
                "is_valid_event": None,
            },
            {
                "review_id": "3",
                "event_type": "close_following",
                "review_status": "reviewed",
                "is_valid_event": True,
            },
        ]
    ).to_parquet(index_path)

    result = EvidenceStore(index_path).query_index(
        target="events",
        operation="summary",
        filters={"event_type": "hard_braking", "review_status": "reviewed"},
    )
    summary = result["items"][0]["result"]

    assert result["ok"] is True
    assert summary["num_events"] == 1
    assert summary["num_valid"] == 1
    assert summary["filters"] == {"event_type": "hard_braking", "review_status": "reviewed"}
    assert summary["event_type_counts"] == {"hard_braking": 1}


def test_evidence_store_queries_scenarios(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    scenario_index_path = tmp_path / "scenario_index.parquet"
    pd.DataFrame([{"review_id": "1", "event_type": "hard_braking"}]).to_parquet(index_path)
    pd.DataFrame(
        [
            {
                "scenario_id": "scene-1",
                "city": "austin",
                "object_types": "pedestrian,vehicle",
                "has_map": True,
            },
            {
                "scenario_id": "scene-2",
                "city": "miami",
                "object_types": "vehicle",
                "has_map": False,
            },
        ]
    ).to_parquet(scenario_index_path)

    result = EvidenceStore(index_path, scenario_index_path=scenario_index_path).query_index(
        target="scenarios",
        operation="search",
        filters={"city": "austin", "object_type": "pedestrian", "has_map": True},
    )

    assert [item["result"]["scenario_id"] for item in result["items"]] == ["scene-1"]


def test_evidence_store_returns_structured_error_for_unknown_query(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    pd.DataFrame([{"review_id": "1", "event_type": "hard_braking"}]).to_parquet(index_path)

    result = EvidenceStore(index_path).query_index(target="events", operation="unknown")

    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert "Unknown evidence operation" in result["items"][0]["error"]


def test_evidence_store_removed_old_public_wrappers(tmp_path: Path) -> None:
    index_path = tmp_path / "event_index.parquet"
    pd.DataFrame([{"review_id": "1", "event_type": "hard_braking"}]).to_parquet(index_path)

    store = EvidenceStore(index_path)

    assert not hasattr(store, "search_events")
    assert not hasattr(store, "get_event_detail")
    assert not hasattr(store, "get_event_evidence")
    assert not hasattr(store, "search_scenarios")
    assert not hasattr(store, "summarize_batch")
