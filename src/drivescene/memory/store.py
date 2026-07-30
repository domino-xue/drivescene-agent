from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


PASSWORD_ITERATIONS = 210_000


@dataclass(frozen=True)
class User:
    id: int
    username: str
    created_at: str


@dataclass(frozen=True)
class Thread:
    id: int
    user_id: int
    title: str
    summary: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: int
    thread_id: int
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class MemoryItem:
    id: int
    user_id: int
    thread_id: int | None
    scope: str
    key: str
    memory_type: str
    value: Any
    object_kind: str | None
    semantic_tags: list[str]
    description: str
    source_tool: str | None
    operation: str | None
    status: str
    created_at: str
    updated_at: str


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_user(self, username: str, password: str) -> User:
        now = _utc_now()
        password_hash = _hash_password(password)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, now),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise ValueError("username already exists") from error
        return User(id=user_id, username=username, created_at=now)

    def authenticate_user(self, username: str, password: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None or not _verify_password(password, str(row["password_hash"])):
            return None
        return User(id=int(row["id"]), username=str(row["username"]), created_at=str(row["created_at"]))

    def create_thread(self, user_id: int, title: str) -> Thread:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO threads (user_id, title, summary, created_at, updated_at)
                VALUES (?, ?, '', ?, ?)
                """,
                (user_id, title, now, now),
            )
            thread_id = int(cursor.lastrowid)
        return Thread(
            id=thread_id,
            user_id=user_id,
            title=title,
            summary="",
            created_at=now,
            updated_at=now,
        )

    def list_threads(self, user_id: int) -> list[Thread]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, title, summary, created_at, updated_at
                FROM threads
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [_thread_from_row(row) for row in rows]

    def get_thread(self, thread_id: int) -> Thread | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title, summary, created_at, updated_at
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
        return _thread_from_row(row) if row is not None else None

    def update_thread_summary(self, thread_id: int, summary: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE threads SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, now, thread_id),
            )

    def update_thread_title(self, thread_id: int, title: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, thread_id),
            )

    def append_message(
        self,
        thread_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        now = _utc_now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (thread_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, role, content, metadata_json, now),
            )
            conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
            message_id = int(cursor.lastrowid)
        return Message(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=now,
        )

    def list_messages(self, thread_id: int, limit: int | None = None) -> list[Message]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT id, thread_id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY id ASC
                    """,
                    (thread_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, thread_id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (thread_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
        return [_message_from_row(row) for row in rows]

    def upsert_memory_item(
        self,
        user_id: int,
        key: str,
        memory_type: str,
        value: Any,
        description: str,
        semantic_tags: list[str] | None = None,
        thread_id: int | None = None,
        object_kind: str | None = None,
        source_tool: str | None = None,
        operation: str | None = None,
        scope: str | None = None,
        status: str = "active",
    ) -> MemoryItem:
        now = _utc_now()
        effective_scope = scope or ("thread" if thread_id is not None else "user")
        value_json = _json_dumps(value)
        tags_json = _json_dumps(sorted(set(semantic_tags or [])))
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM memory_items
                WHERE user_id = ?
                  AND IFNULL(thread_id, -1) = IFNULL(?, -1)
                  AND key = ?
                  AND memory_type = ?
                  AND value_json = ?
                """,
                (user_id, thread_id, key, memory_type, value_json),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_items (
                        user_id, thread_id, scope, key, memory_type, value_json, object_kind,
                        semantic_tags_json, description, source_tool, operation, status,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        thread_id,
                        effective_scope,
                        key,
                        memory_type,
                        value_json,
                        object_kind,
                        tags_json,
                        description,
                        source_tool,
                        operation,
                        status,
                        now,
                        now,
                    ),
                )
                memory_id = int(cursor.lastrowid)
            else:
                memory_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE memory_items
                    SET scope = ?, object_kind = ?, semantic_tags_json = ?, description = ?,
                        source_tool = ?, operation = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        effective_scope,
                        object_kind,
                        tags_json,
                        description,
                        source_tool,
                        operation,
                        status,
                        now,
                        memory_id,
                    ),
                )
        return self.get_memory_item(memory_id)

    def get_memory_item(self, memory_id: int) -> MemoryItem:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return _memory_from_row(row)

    def query_memory(
        self,
        user_id: int,
        thread_id: int | None = None,
        memory_type: str | None = None,
        object_kind: str | None = None,
        semantic_tags: list[str] | None = None,
        status: str = "active",
        include_user_scope: bool = False,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        clauses = ["user_id = ?", "status = ?"]
        params: list[Any] = [user_id, status]
        if thread_id is not None:
            if include_user_scope:
                clauses.append("(thread_id = ? OR scope = 'user')")
            else:
                clauses.append("thread_id = ?")
            params.append(thread_id)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if object_kind is not None:
            clauses.append("object_kind = ?")
            params.append(object_kind)

        query = "SELECT * FROM memory_items WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        required_tags = set(semantic_tags or [])
        items = [_memory_from_row(row) for row in rows]
        if required_tags:
            items = [item for item in items if required_tags.issubset(set(item.semantic_tags))]
        return items

    def mark_path_deleted(self, user_id: int, paths: list[str]) -> None:
        now = _utc_now()
        value_jsons = [_json_dumps(path) for path in paths]
        with self._connect() as conn:
            for value_json in value_jsons:
                conn.execute(
                    """
                    UPDATE memory_items
                    SET status = 'deleted', updated_at = ?, operation = 'delete'
                    WHERE user_id = ? AND memory_type = 'path' AND value_json = ?
                    """,
                    (now, user_id, value_json),
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    thread_id INTEGER REFERENCES threads(id) ON DELETE CASCADE,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    object_kind TEXT,
                    semantic_tags_json TEXT NOT NULL DEFAULT '[]',
                    description TEXT NOT NULL DEFAULT '',
                    source_tool TEXT,
                    operation TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_threads_user_updated
                    ON threads(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id
                    ON messages(thread_id, id);
                CREATE INDEX IF NOT EXISTS idx_memory_lookup
                    ON memory_items(user_id, thread_id, memory_type, object_kind, status, updated_at DESC);
                """
            )


def _thread_from_row(row: sqlite3.Row) -> Thread:
    return Thread(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        title=str(row["title"]),
        summary=str(row["summary"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    return Message(
        id=int(row["id"]),
        thread_id=int(row["thread_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        metadata=json.loads(str(row["metadata_json"])),
        created_at=str(row["created_at"]),
    )


def _memory_from_row(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        thread_id=int(row["thread_id"]) if row["thread_id"] is not None else None,
        scope=str(row["scope"]),
        key=str(row["key"]),
        memory_type=str(row["memory_type"]),
        value=json.loads(str(row["value_json"])),
        object_kind=str(row["object_kind"]) if row["object_kind"] is not None else None,
        semantic_tags=list(json.loads(str(row["semantic_tags_json"]))),
        description=str(row["description"]),
        source_tool=str(row["source_tool"]) if row["source_tool"] is not None else None,
        operation=str(row["operation"]) if row["operation"] is not None else None,
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations_text),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
