import json
from pathlib import Path

from drivescene.ops.artifacts import ArtifactOps


def test_manage_artifacts_creates_folder(tmp_path: Path) -> None:
    target = tmp_path / "exports" / "demo"

    result = ArtifactOps().manage_artifacts(operation="create_folder", output_dir=target)

    assert target.exists()
    assert result["output_dir"] == str(target)


def test_manage_artifacts_copies_existing_files_and_reports_missing(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    missing = tmp_path / "missing.txt"
    output_dir = tmp_path / "export"
    source.write_text("hello", encoding="utf-8")

    result = ArtifactOps().manage_artifacts(
        operation="copy",
        paths=[source, missing],
        output_dir=output_dir,
    )

    copied_path = output_dir / "source.txt"
    assert copied_path.read_text(encoding="utf-8") == "hello"
    assert result["num_copied"] == 1
    assert result["copied_files"] == [str(copied_path)]
    assert result["missing_files"] == [str(missing)]


def test_manage_artifacts_preserves_relative_structure(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "a" / "b" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("nested", encoding="utf-8")
    output_dir = tmp_path / "out"

    result = ArtifactOps().manage_artifacts(
        operation="copy",
        paths=[nested],
        output_dir=output_dir,
        preserve_structure=True,
        base_dir=root,
    )

    copied_path = output_dir / "a" / "b" / "file.txt"
    assert copied_path.exists()
    assert result["copied_files"] == [str(copied_path)]


def test_manage_artifacts_deletes_paths(tmp_path: Path) -> None:
    target = tmp_path / "delete-me.txt"
    target.write_text("bye", encoding="utf-8")

    result = ArtifactOps().manage_artifacts(operation="delete", paths=[target])

    assert not target.exists()
    assert result["deleted_files"] == [str(target)]


def test_manage_artifacts_writes_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.json"
    manifest = {
        "query": "valid hard braking",
        "review_ids": ["000094"],
        "files": ["animation.gif"],
    }

    result = ArtifactOps().manage_artifacts(
        operation="write_manifest",
        output_path=output_path,
        manifest=manifest,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["query"] == "valid hard braking"
    assert saved["review_ids"] == ["000094"]
    assert "created_at" in saved
    assert result["manifest_path"] == str(output_path)


def test_manage_artifacts_unknown_operation_returns_structured_error() -> None:
    result = ArtifactOps().manage_artifacts(operation="unknown")

    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert "Unknown artifact operation" in result["error"]


def test_artifact_ops_removed_old_public_wrappers() -> None:
    ops = ArtifactOps()

    assert not hasattr(ops, "create_folder")
    assert not hasattr(ops, "copy_files")
    assert not hasattr(ops, "delete_files")
    assert not hasattr(ops, "write_manifest")
