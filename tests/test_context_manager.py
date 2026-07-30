from pathlib import Path

from drivescene.memory.context import ContextManager
from drivescene.memory.store import MemoryStore


def _store_with_user_thread(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    user = store.create_user("alice", "secret")
    thread = store.create_thread(user.id, "Exports")
    return store, user, thread


def test_context_pack_uses_bounded_recent_messages(tmp_path: Path) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    for index in range(10):
        store.append_message(thread.id, "user", f"message {index}")
    manager = ContextManager(store, recent_message_limit=4)

    pack = manager.build_context_pack(user.id, thread.id, "继续")

    assert [message["content"] for message in pack.recent_messages] == [
        "message 6",
        "message 7",
        "message 8",
        "message 9",
    ]


def test_context_resolves_previous_hard_braking_folder_by_semantic_tags(
    tmp_path: Path,
) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value="outputs/exports/hard_braking_cases",
        object_kind="folder",
        semantic_tags=["hard_braking", "export"],
        description="Folder created for exported hard_braking evidence cases",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )
    manager = ContextManager(store)

    pack = manager.build_context_pack(user.id, thread.id, "将我之前建的包含急刹的文件夹删除掉")

    assert pack.resolved_context["paths"] == ["outputs/exports/hard_braking_cases"]
    assert pack.resolved_context["source"] == "structured_memory"
    assert pack.unresolved_references == []
    assert set(pack.structured_memory[0]["semantic_tags"]) == {"hard_braking", "export"}


def test_context_resolves_this_folder_to_latest_active_folder(tmp_path: Path) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value="outputs/exports/older",
        object_kind="folder",
        semantic_tags=["export"],
        description="Older export folder",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )
    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="artifact_folder",
        memory_type="path",
        value="outputs/exports/latest",
        object_kind="folder",
        semantic_tags=["export"],
        description="Latest export folder",
        source_tool="ArtifactOps.manage_artifacts",
        operation="create_folder",
    )
    manager = ContextManager(store)

    pack = manager.build_context_pack(user.id, thread.id, "将这个文件夹删除")

    assert pack.resolved_context["paths"] == ["outputs/exports/latest"]


def test_context_does_not_guess_when_tagged_folder_reference_is_ambiguous(
    tmp_path: Path,
) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    for path in ["outputs/exports/hard_a", "outputs/exports/hard_b"]:
        store.upsert_memory_item(
            user_id=user.id,
            thread_id=thread.id,
            key="artifact_folder",
            memory_type="path",
            value=path,
            object_kind="folder",
            semantic_tags=["hard_braking", "export"],
            description=f"Hard braking export folder {path}",
            source_tool="ArtifactOps.manage_artifacts",
            operation="create_folder",
        )
    manager = ContextManager(store)

    pack = manager.build_context_pack(user.id, thread.id, "删除之前建的急刹文件夹")

    assert "paths" not in pack.resolved_context
    assert pack.unresolved_references[0]["reason"] == "ambiguous_memory"
    assert pack.unresolved_references[0]["candidates"] == [
        "outputs/exports/hard_b",
        "outputs/exports/hard_a",
    ]


def test_context_resolves_this_event_to_latest_review_id(tmp_path: Path) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    store.upsert_memory_item(
        user_id=user.id,
        thread_id=thread.id,
        key="last_event",
        memory_type="entity",
        value={
            "review_id": "000123",
            "event_type": "hard_braking",
            "animation_path": "outputs/review_assets/000123/animation.gif",
        },
        object_kind="event",
        semantic_tags=["hard_braking"],
        description="Last event returned to the user",
        source_tool="EvidenceStore.query_index",
        operation="search",
    )
    manager = ContextManager(store)

    pack = manager.build_context_pack(user.id, thread.id, "给我它的文件地址")

    assert pack.resolved_context["review_ids"] == ["000123"]
    assert pack.resolved_context["event"]["event_type"] == "hard_braking"
    assert pack.resolved_context["event"]["animation_path"] == "outputs/review_assets/000123/animation.gif"


def test_context_identifies_dominant_event_type_from_thread_memory(tmp_path: Path) -> None:
    store, user, thread = _store_with_user_thread(tmp_path)
    for review_id in ["000123", "000124"]:
        store.upsert_memory_item(
            user_id=user.id,
            thread_id=thread.id,
            key="last_event",
            memory_type="entity",
            value={"review_id": review_id, "event_type": "hard_braking"},
            object_kind="event",
            semantic_tags=["hard_braking"],
            description="Hard braking event returned to the user",
            source_tool="EvidenceStore.query_index",
            operation="search",
        )
    manager = ContextManager(store)

    pack = manager.build_context_pack(user.id, thread.id, "我咨询你的主要是关于哪类事件？")

    assert pack.resolved_context["event_focus"]["event_type"] == "hard_braking"
    assert "急刹" in pack.resolved_context["direct_answer"]
