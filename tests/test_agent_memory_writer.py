from pathlib import Path

from drivescene.memory.store import MemoryStore
from drivescene.memory.writer import AgentMemoryWriter


def test_agent_memory_writer_remembers_created_export_folder(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "Exports")
    writer = AgentMemoryWriter(store)
    result = {
        "ok": True,
        "operation": "create_folder",
        "output_dir": "outputs/exports/hard_braking_cases",
        "items": [
            {
                "input": {"output_dir": "outputs/exports/hard_braking_cases"},
                "ok": True,
                "result": {"output_dir": "outputs/exports/hard_braking_cases"},
            }
        ],
    }

    writer.record_tool_result(
        user_id=user.id,
        thread_id=thread.id,
        user_request="帮我导出急刹案例到一个文件夹",
        tool_name="ArtifactOps.manage_artifacts",
        result=result,
    )

    matches = store.query_memory(
        user_id=user.id,
        thread_id=thread.id,
        memory_type="path",
        object_kind="folder",
        semantic_tags=["hard_braking"],
    )
    assert len(matches) == 1
    assert matches[0].value == "outputs/exports/hard_braking_cases"
    assert "export" in matches[0].semantic_tags


def test_agent_memory_writer_marks_deleted_paths(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "Exports")
    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value="outputs/exports/hard_braking_cases",
        object_kind="folder",
        semantic_tags=["hard_braking", "export"],
        description="Hard braking export folder",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )
    writer = AgentMemoryWriter(store)
    result = {
        "ok": True,
        "operation": "delete",
        "deleted_files": ["outputs/exports/hard_braking_cases"],
        "items": [
            {
                "input": {"path": "outputs/exports/hard_braking_cases"},
                "ok": True,
                "result": {"deleted_path": "outputs/exports/hard_braking_cases"},
            }
        ],
    }

    writer.record_tool_result(
        user_id=user.id,
        thread_id=thread.id,
        user_request="删除急刹文件夹",
        tool_name="ArtifactOps.manage_artifacts",
        result=result,
    )

    assert store.query_memory(user_id=user.id, thread_id=thread.id) == []
    deleted = store.query_memory(user_id=user.id, thread_id=thread.id, status="deleted")
    assert deleted[0].value == "outputs/exports/hard_braking_cases"


def test_agent_memory_writer_remembers_last_event_from_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "Events")
    writer = AgentMemoryWriter(store)
    result = {
        "ok": True,
        "operation": "search",
        "summary": {"total": 1, "succeeded": 1, "failed": 0},
        "items": [
            {
                "input": {"filters": {"event_type": "hard_braking"}},
                "ok": True,
                "result": {
                    "review_id": "000123",
                    "event_type": "hard_braking",
                    "scenario_id": "scene-1",
                    "animation_path": "outputs/review_assets/000123/animation.gif",
                },
                "error": None,
            }
        ],
    }

    writer.record_tool_result(
        user_id=user.id,
        thread_id=thread.id,
        user_request="给我一个急刹的事件",
        tool_name="EvidenceStore.query_index",
        result=result,
    )

    matches = store.query_memory(
        user_id=user.id,
        thread_id=thread.id,
        memory_type="entity",
        object_kind="event",
        semantic_tags=["hard_braking"],
    )
    assert len(matches) == 1
    assert matches[0].key == "last_event"
    assert matches[0].value["review_id"] == "000123"
    assert matches[0].value["animation_path"] == "outputs/review_assets/000123/animation.gif"
