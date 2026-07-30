from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from drivescene.ops.contracts import error_item, success_item, tool_result, validation_error_result


class ArtifactOps:
    def manage_artifacts(
        self,
        operation: str,
        paths: list[Path | str] | None = None,
        output_dir: Path | str | None = None,
        output_path: Path | str | None = None,
        manifest: dict[str, Any] | None = None,
        preserve_structure: bool = False,
        base_dir: Path | str | None = None,
        overwrite: bool = False,
        missing_ok: bool = True,
    ) -> dict[str, Any]:
        try:
            if operation == "create_folder":
                result = _create_folder(_required_path(output_dir, "output_dir"))
                envelope = tool_result(operation, [success_item({"output_dir": output_dir}, result)])
                return {**envelope, **result}
            if operation == "copy":
                return _copy_artifacts(
                    paths or [],
                    _required_path(output_dir, "output_dir"),
                    preserve_structure,
                    base_dir,
                    overwrite,
                )
            if operation == "delete":
                return _delete_artifacts(paths or [], missing_ok)
            if operation == "write_manifest":
                result = _write_manifest(
                    _required_path(output_path, "output_path"),
                    manifest or {},
                )
                envelope = tool_result(operation, [success_item({"output_path": output_path}, result)])
                return {**envelope, **result}
            raise ValueError(f"Unknown artifact operation: {operation}")
        except (FileNotFoundError, ValueError) as error:
            return validation_error_result(operation, error)


def _create_folder(output_dir: Path | str) -> dict[str, Any]:
    output_path = Path(output_dir)
    existed = output_path.exists()
    output_path.mkdir(parents=True, exist_ok=True)
    return {
        "operation": "create_folder",
        "output_dir": str(output_path),
        "created": not existed,
    }


def _copy_artifacts(
    paths: list[Path | str],
    output_dir: Path | str,
    preserve_structure: bool,
    base_dir: Path | str | None,
    overwrite: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_dir) if base_dir is not None else None
    items: list[dict[str, Any]] = []

    for path_like in paths:
        source = Path(path_like)
        if not source.exists() or not source.is_file():
            items.append(error_item({"path": str(source)}, FileNotFoundError(str(source))))
            continue
        destination = _destination_for(source, output_path, preserve_structure, base_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            items.append(
                success_item(
                    {"path": str(source)},
                    {
                        "source_path": str(source),
                        "destination_path": str(destination),
                        "status": "skipped",
                    },
                )
            )
            continue
        shutil.copy2(source, destination)
        items.append(
            success_item(
                {"path": str(source)},
                {
                    "source_path": str(source),
                    "destination_path": str(destination),
                    "status": "copied",
                },
            )
        )

    result = tool_result("copy", items)
    result["output_dir"] = str(output_path)
    result["copied_files"] = [
        item["result"]["destination_path"]
        for item in result["items"]
        if item.get("ok") and item.get("result", {}).get("status") == "copied"
    ]
    result["missing_files"] = [
        item["input"]["path"] for item in result["items"] if item.get("ok") is False
    ]
    result["skipped_files"] = [
        item["result"]["destination_path"]
        for item in result["items"]
        if item.get("ok") and item.get("result", {}).get("status") == "skipped"
    ]
    result["num_copied"] = len(result["copied_files"])
    return result


def _delete_artifacts(paths: list[Path | str], missing_ok: bool) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            if not missing_ok:
                raise FileNotFoundError(path)
            items.append(error_item({"path": str(path)}, FileNotFoundError(str(path))))
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        items.append(success_item({"path": str(path)}, {"deleted_path": str(path)}))

    result = tool_result("delete", items)
    result["deleted_files"] = [
        item["result"]["deleted_path"] for item in result["items"] if item.get("ok")
    ]
    result["missing_files"] = [
        item["input"]["path"] for item in result["items"] if item.get("ok") is False
    ]
    result["num_deleted"] = len(result["deleted_files"])
    return result


def _write_manifest(
    output_path: Path | str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "operation": "write_manifest",
        "manifest_path": str(path),
    }


def _required_path(value: Path | str | None, name: str) -> Path | str:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _destination_for(
    source: Path,
    output_dir: Path,
    preserve_structure: bool,
    base_dir: Path | None,
) -> Path:
    if not preserve_structure:
        return output_dir / source.name
    if base_dir is None:
        return output_dir / source.name
    return output_dir / source.relative_to(base_dir)
