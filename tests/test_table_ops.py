from pathlib import Path

import pandas as pd

from drivescene.ops.tables import TableOps


def test_transform_table_drops_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "events.parquet"
    output_path = tmp_path / "events_public.parquet"
    pd.DataFrame(
        [
            {
                "review_id": "000001",
                "event_type": "hard_braking",
                "review_note": "private",
            }
        ]
    ).to_parquet(input_path)

    result = TableOps().transform_table(
        operation="drop_columns",
        input_path=input_path,
        output_path=output_path,
        columns=["review_note"],
    )

    output = pd.read_parquet(output_path)
    assert "review_note" not in output.columns
    assert result["dropped_columns"] == ["review_note"]
    assert "event_type" in result["remaining_columns"]


def test_transform_table_supports_in_place_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "events.parquet"
    pd.DataFrame([{"review_id": "000001", "secret": "x"}]).to_parquet(input_path)

    result = TableOps().transform_table(
        operation="drop_columns",
        input_path=input_path,
        output_path=input_path,
        columns=["secret"],
        overwrite=True,
    )

    output = pd.read_parquet(input_path)
    assert "secret" not in output.columns
    assert result["output_path"] == str(input_path)


def test_transform_table_filters_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "events.parquet"
    output_path = tmp_path / "valid_hard.parquet"
    pd.DataFrame(
        [
            {"review_id": "1", "event_type": "hard_braking", "is_valid_event": True},
            {"review_id": "2", "event_type": "close_following", "is_valid_event": True},
            {"review_id": "3", "event_type": "hard_braking", "is_valid_event": False},
        ]
    ).to_parquet(input_path)

    result = TableOps().transform_table(
        operation="filter_rows",
        input_path=input_path,
        output_path=output_path,
        filters={"event_type": "hard_braking", "is_valid_event": True},
    )

    output = pd.read_parquet(output_path)
    assert output["review_id"].tolist() == ["1"]
    assert result["num_rows"] == 1


def test_transform_table_converts_format(tmp_path: Path) -> None:
    input_path = tmp_path / "events.parquet"
    output_path = tmp_path / "events.csv"
    pd.DataFrame([{"review_id": "000001", "event_type": "hard_braking"}]).to_parquet(input_path)

    result = TableOps().transform_table(
        operation="convert_format",
        input_path=input_path,
        output_path=output_path,
    )

    output = pd.read_csv(output_path)
    assert output["review_id"].tolist() == [1]
    assert result["output_path"] == str(output_path)


def test_transform_table_unknown_operation_returns_structured_error(tmp_path: Path) -> None:
    input_path = tmp_path / "events.parquet"
    output_path = tmp_path / "out.parquet"
    pd.DataFrame([{"review_id": "1"}]).to_parquet(input_path)

    result = TableOps().transform_table(
        operation="unknown",
        input_path=input_path,
        output_path=output_path,
    )

    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert "Unknown table operation" in result["error"]


def test_table_ops_removed_old_public_wrappers() -> None:
    ops = TableOps()

    assert not hasattr(ops, "drop_columns")
    assert not hasattr(ops, "filter_rows")
    assert not hasattr(ops, "convert_table_format")
