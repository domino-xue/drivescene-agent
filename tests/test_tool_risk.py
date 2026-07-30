from drivescene.ops.risk import assess_tool_risk


def test_assess_tool_risk_marks_safe_artifact_operations_low() -> None:
    assessment = assess_tool_risk(
        "ArtifactOps.manage_artifacts",
        {"operation": "copy", "paths": ["a.txt"], "output_dir": "out", "overwrite": False},
    )

    assert assessment["risk_level"] == "low"
    assert assessment["requires_confirmation"] is False


def test_assess_tool_risk_marks_delete_high() -> None:
    assessment = assess_tool_risk(
        "ArtifactOps.manage_artifacts",
        {"operation": "delete", "paths": ["outputs/old"]},
    )

    assert assessment["risk_level"] == "high"
    assert assessment["requires_confirmation"] is True
    assert "delete" in assessment["reason"]


def test_assess_tool_risk_marks_overwrite_and_in_place_table_ops_high() -> None:
    overwrite = assess_tool_risk(
        "TableOps.transform_table",
        {
            "operation": "drop_columns",
            "input_path": "outputs/event_index.parquet",
            "output_path": "outputs/event_index_public.parquet",
            "columns": ["review_note"],
            "overwrite": True,
        },
    )
    in_place = assess_tool_risk(
        "TableOps.transform_table",
        {
            "operation": "filter_rows",
            "input_path": "outputs/event_index.parquet",
            "output_path": "outputs/event_index.parquet",
            "filters": {"event_type": "hard_braking"},
        },
    )

    assert overwrite["risk_level"] == "high"
    assert overwrite["requires_confirmation"] is True
    assert in_place["risk_level"] == "high"
    assert in_place["requires_confirmation"] is True


def test_assess_tool_risk_marks_derived_table_write_medium() -> None:
    assessment = assess_tool_risk(
        "TableOps.transform_table",
        {
            "operation": "filter_rows",
            "input_path": "outputs/event_index.parquet",
            "output_path": "outputs/exports/hard.parquet",
            "filters": {"event_type": "hard_braking"},
        },
    )

    assert assessment["risk_level"] == "medium"
    assert assessment["requires_confirmation"] is False
