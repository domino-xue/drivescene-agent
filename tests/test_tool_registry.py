from pathlib import Path

import pandas as pd

from drivescene.agent.tool_registry import (
    build_tool_registry,
    execute_registered_tool,
    preflight_tool_call,
    tool_descriptions,
)
from drivescene.analysis.tools import AnalysisTools
from drivescene.evidence.tools import EvidenceStore
from drivescene.ops.artifacts import ArtifactOps
from drivescene.ops.tables import TableOps


def _store(tmp_path: Path) -> EvidenceStore:
    event_index = tmp_path / "event_index.parquet"
    scenario_index = tmp_path / "scenario_index.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "is_valid_event": True,
                "city": "austin",
            }
        ]
    ).to_parquet(event_index)
    pd.DataFrame(
        [
            {
                "scenario_id": "scene-1",
                "city": "austin",
                "object_types": "vehicle",
                "has_map": True,
            }
        ]
    ).to_parquet(scenario_index)
    return EvidenceStore(event_index, scenario_index_path=scenario_index)


def _registry(tmp_path: Path):
    return build_tool_registry(
        evidence_store=_store(tmp_path),
        analysis_tools=AnalysisTools(tmp_path),
        artifact_ops=ArtifactOps(),
        table_ops=TableOps(),
    )


def test_build_tool_registry_exposes_only_generic_parameterized_tools(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    expected_names = {
        "EvidenceStore.query_index",
        "AnalysisTools.run_analysis",
        "ArtifactOps.manage_artifacts",
        "TableOps.transform_table",
    }
    removed_names = {
        "EvidenceStore.search_events",
        "EvidenceStore.get_event_detail",
        "EvidenceStore.get_event_evidence",
        "EvidenceStore.search_scenarios",
        "EvidenceStore.summarize_batch",
        "AnalysisTools.detect_hard_braking",
        "AnalysisTools.detect_close_following",
        "AnalysisTools.compute_track_kinematics",
        "AnalysisTools.compute_relative_motion",
        "AnalysisTools.compare_velocity_position_evidence",
        "ArtifactOps.create_folder",
        "ArtifactOps.copy_files",
        "ArtifactOps.delete_files",
        "ArtifactOps.write_manifest",
        "TableOps.drop_columns",
        "TableOps.filter_rows",
        "TableOps.convert_table_format",
    }

    assert set(registry) == expected_names
    assert removed_names.isdisjoint(registry)
    assert registry["EvidenceStore.query_index"].category == "retrieval"
    assert registry["EvidenceStore.query_index"].args_schema["target"] == "str"
    assert registry["ArtifactOps.manage_artifacts"].requires_risk_preflight is True


def test_tool_descriptions_are_structured_english_with_chinese_notes(tmp_path: Path) -> None:
    exported = tool_descriptions(_registry(tmp_path))

    assert set(exported[0]) == {
        "name",
        "category",
        "description",
        "args_schema",
        "purpose",
        "input_contract",
        "output_contract",
        "when_to_use",
        "do_not_use_when",
        "zh_note",
        "examples",
    }
    description = exported[0]["description"]
    assert "Input:" in description
    assert "Output:" in description
    assert "When to use:" in description
    assert "Do not use when:" in description
    assert "中文备注" in description
    assert exported[0]["purpose"]
    assert isinstance(exported[0]["examples"], list)


def test_execute_registered_tool_calls_whitelisted_generic_function(tmp_path: Path) -> None:
    result = execute_registered_tool(
        _registry(tmp_path),
        {
            "tool": "EvidenceStore.query_index",
            "args": {
                "target": "events",
                "operation": "search",
                "filters": {"event_type": "hard_braking"},
                "limit": 1,
            },
        },
    )

    assert result["tool"] == "EvidenceStore.query_index"
    assert result["result"]["items"][0]["result"]["review_id"] == "000001"


def test_execute_registered_tool_rejects_unknown_tool(tmp_path: Path) -> None:
    try:
        execute_registered_tool(_registry(tmp_path), {"tool": "os.remove", "args": {"path": "x"}})
    except KeyError as error:
        assert "Unknown tool" in str(error)
    else:
        raise AssertionError("unknown tool should be rejected")


def test_preflight_tool_call_uses_operation_aware_risk_policy(tmp_path: Path) -> None:
    result = preflight_tool_call(
        _registry(tmp_path),
        {
            "tool": "ArtifactOps.manage_artifacts",
            "args": {"operation": "delete", "paths": ["outputs/old"]},
        },
    )

    assert result["allowed_tool"] is True
    assert result["risk"]["risk_level"] == "high"
    assert result["risk"]["requires_confirmation"] is True
