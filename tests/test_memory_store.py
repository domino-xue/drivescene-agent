from pathlib import Path

import pytest

from drivescene.memory.store import MemoryStore


def test_memory_store_registers_authenticates_and_lists_user_threads(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")

    user = store.create_user("alice", "correct horse battery staple")
    authenticated = store.authenticate_user("alice", "correct horse battery staple")
    failed = store.authenticate_user("alice", "wrong password")
    first_thread = store.create_thread(user.id, "Hard braking exports")
    second_thread = store.create_thread(user.id, "Close following review")

    assert authenticated is not None
    assert authenticated.id == user.id
    assert failed is None
    assert [thread.title for thread in store.list_threads(user.id)] == [
        "Close following review",
        "Hard braking exports",
    ]
    assert first_thread.user_id == user.id
    assert second_thread.user_id == user.id


def test_memory_store_rejects_duplicate_usernames(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.create_user("alice", "secret")

    with pytest.raises(ValueError, match="username already exists"):
        store.create_user("alice", "different")


def test_memory_store_persists_messages_and_structured_memory(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    store = MemoryStore(db_path)
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "Exports")
    store.append_message(thread.id, "user", "帮我把急刹案例复制到一个文件夹")
    store.append_message(thread.id, "assistant", "已创建 outputs/exports/hard_braking_cases")

    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value="outputs/exports/hard_braking_cases",
        object_kind="folder",
        semantic_tags=["hard_braking", "export", "evidence_folder"],
        description="Folder created for exported hard_braking evidence cases",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )

    reopened = MemoryStore(db_path)
    messages = reopened.list_messages(thread.id)
    matches = reopened.query_memory(
        user_id=user.id,
        thread_id=thread.id,
        memory_type="path",
        object_kind="folder",
        semantic_tags=["hard_braking"],
    )

    assert [message.content for message in messages] == [
        "帮我把急刹案例复制到一个文件夹",
        "已创建 outputs/exports/hard_braking_cases",
    ]
    assert len(matches) == 1
    assert matches[0].value == "outputs/exports/hard_braking_cases"
    assert matches[0].status == "active"


def test_memory_store_marks_path_memory_deleted(tmp_path: Path) -> None:
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
        semantic_tags=["hard_braking"],
        description="Hard braking export folder",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )

    store.mark_path_deleted(user.id, ["outputs/exports/hard_braking_cases"])

    assert store.query_memory(user_id=user.id, thread_id=thread.id, semantic_tags=["hard_braking"]) == []
    deleted = store.query_memory(
        user_id=user.id,
        thread_id=thread.id,
        semantic_tags=["hard_braking"],
        status="deleted",
    )
    assert deleted[0].value == "outputs/exports/hard_braking_cases"


def test_memory_store_updates_thread_title(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "新对话")

    store.update_thread_title(thread.id, "找 5 个急刹案例")

    assert store.get_thread(thread.id).title == "找 5 个急刹案例"
