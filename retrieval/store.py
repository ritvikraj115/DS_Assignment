from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from ingestion.models import utc_now


class AppStore:
    """Conversation and feedback persistence, separate from answer generation."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_conversation_session ON conversation_messages(session_id, id);
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY, session_id TEXT, question TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('up', 'down')), comment TEXT, created_at TEXT NOT NULL
            );
        """)
        self.connection.commit()

    def history(self, session_id: str | None, max_turns: int = 8) -> list[dict[str, str]]:
        if not session_id:
            return []
        rows = self.connection.execute(
            "SELECT role, content FROM conversation_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, max_turns * 2),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def save_exchange(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        now = utc_now()
        with self.connection:
            self.connection.executemany(
                "INSERT INTO conversation_messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(session_id, "user", question, now), (session_id, "assistant", answer, now)],
            )

    def save_feedback(self, session_id: str | None, question: str, rating: str, comment: str | None) -> dict[str, Any]:
        feedback_id, created_at = str(uuid.uuid4()), utc_now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (feedback_id, session_id, question, rating, comment, created_at),
            )
        return {"feedback_id": feedback_id, "status": "recorded", "created_at": created_at}
