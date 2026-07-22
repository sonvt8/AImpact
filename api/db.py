from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user', 'viewer')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT NOT NULL UNIQUE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_user ON conversations(user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_message_conversation ON messages(conversation_id, id);
                """
            )

    def user_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password_hash: str, role: str) -> dict:
        created_at = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users(username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, created_at),
            )
            user_id = cursor.lastrowid
        return {"id": user_id, "username": username, "role": role, "created_at": created_at}

    def get_user_by_username(self, username: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY username COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    def count_role(self, role: str) -> int:
        with self.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM users WHERE role = ?", (role,)).fetchone()[0]
            )

    def update_user(self, user_id: int, *, role: str | None, password_hash: str | None) -> dict | None:
        updates = []
        values = []
        if role is not None:
            updates.append("role = ?")
            values.append(role)
        if password_hash is not None:
            updates.append("password_hash = ?")
            values.append(password_hash)
        if updates:
            values.append(user_id)
            with self.connect() as connection:
                connection.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)
        user = self.get_user(user_id)
        if not user:
            return None
        return {key: user[key] for key in ("id", "username", "role", "created_at")}

    def delete_user(self, user_id: int, current_user_id: int) -> None:
        if user_id == current_user_id:
            raise ValueError("You cannot delete your own account")
        with self.connect() as connection:
            row = connection.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                raise LookupError("User not found")
            if row["role"] == "admin":
                admins = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin'"
                ).fetchone()[0]
                if admins <= 1:
                    raise ValueError("The last administrator cannot be deleted")
            connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def store_refresh(self, token_hash: str, user_id: int, expires_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO refresh_tokens(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, expires_at, utc_now()),
            )

    def active_refresh(self, token_hash: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM refresh_tokens WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            ).fetchone()
        return dict(row) if row else None

    def revoke_refresh(self, token_hash: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE refresh_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE token_hash = ?",
                (utc_now(), token_hash),
            )

    def revoke_all_refresh(self, user_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE refresh_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE user_id = ?",
                (utc_now(), user_id),
            )

    def create_conversation(self, user_id: int, title: str) -> dict:
        conversation_id = str(uuid4())
        timestamp = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, title, timestamp, timestamp),
            )
        return {
            "id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def list_conversations(self, user_id: int) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def conversation(self, conversation_id: str, user_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not row:
                return None
            messages = connection.execute(
                "SELECT id, role, content, citations_json, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        result = dict(row)
        result["messages"] = [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "citations": json.loads(message["citations_json"]),
                "created_at": message["created_at"],
            }
            for message in messages
        ]
        return result

    def conversation_history(self, conversation_id: str, user_id: int, limit: int = 12) -> str | None:
        if not self.conversation(conversation_id, user_id):
            return None
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return "\n".join(
            f"{'Người dùng' if row['role'] == 'user' else 'Trợ lý'}: {row['content']}"
            for row in reversed(rows)
        )

    def add_message(self, conversation_id: str, role: str, content: str, citations: list | None = None) -> None:
        timestamp = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(citations or [], ensure_ascii=False), timestamp),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )

    def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        return cursor.rowcount > 0

    def add_audit(self, user: dict, action: str, resource_id: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO audit(user_id, username, action, resource_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user["id"], user["username"], action, resource_id, utc_now()),
            )

    def list_audit(self, limit: int = 500) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, username, action, resource_id, timestamp FROM audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
