from __future__ import annotations

import json
import hashlib
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
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_conversation_session ON conversation_messages(session_id, id);
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY, session_id TEXT, question TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('up', 'down')), comment TEXT, created_at TEXT NOT NULL
            );
        """)
        existing = {row["name"] for row in self.connection.execute("PRAGMA table_info(feedback)")}
        migrations = {
            "normalized_question": "TEXT",
            "question_embedding": "TEXT",
            "answer": "TEXT",
            "answer_hash": "TEXT",
            "confidence_json": "TEXT",
            "sources_json": "TEXT NOT NULL DEFAULT '[]'",
            "corpus_fingerprint": "TEXT",
            "scope_json": "TEXT NOT NULL DEFAULT '[]'",
            "response_status": "TEXT",
        }
        for name, declaration in migrations.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE feedback ADD COLUMN {name} {declaration}")
        message_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(conversation_messages)")}
        if "metadata_json" not in message_columns:
            self.connection.execute("ALTER TABLE conversation_messages ADD COLUMN metadata_json TEXT")
        self.connection.commit()

    def history(self, session_id: str | None, max_turns: int = 8) -> list[dict[str, str]]:
        if not session_id:
            return []
        rows = self.connection.execute(
            "SELECT role, content FROM conversation_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, max_turns * 2),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def recent_questions(self, limit: int = 100) -> list[str]:
        """Return distinct questions from prior conversations, newest first."""
        rows = self.connection.execute(
            """SELECT content, MAX(id) AS latest_id FROM conversation_messages
               WHERE role='user' AND TRIM(content) != ''
               GROUP BY content COLLATE NOCASE ORDER BY latest_id DESC LIMIT ?""",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
        return [str(row["content"]) for row in rows]

    def save_exchange(self, session_id: str | None, question: str, answer: str,
                      metadata: dict[str, Any] | None = None) -> None:
        if not session_id:
            return
        now = utc_now()
        with self.connection:
            self.connection.executemany(
                "INSERT INTO conversation_messages(session_id, role, content, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                [(session_id, "user", question, now, None),
                 (session_id, "assistant", answer, now, json.dumps(metadata) if metadata is not None else None)],
            )

    def verified_exchange(self, session_id: str | None, question: str, answer: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        row = self.connection.execute(
            """SELECT assistant.metadata_json FROM conversation_messages AS assistant
               JOIN conversation_messages AS user_message ON user_message.id = assistant.id - 1
               WHERE assistant.session_id=? AND assistant.role='assistant' AND assistant.content=?
                 AND user_message.session_id=assistant.session_id AND user_message.role='user'
                 AND user_message.content=? ORDER BY assistant.id DESC LIMIT 1""",
            (session_id, answer, question),
        ).fetchone()
        if not row or not row["metadata_json"]:
            return None
        return json.loads(row["metadata_json"])

    def delete_session(self, session_id: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM conversation_messages WHERE session_id=?", (session_id,))
            self.connection.execute("DELETE FROM feedback WHERE session_id=?", (session_id,))

    def save_feedback(self, session_id: str | None, question: str, rating: str, comment: str | None, *,
                      answer: str = "", confidence: dict[str, Any] | None = None,
                      sources: list[dict[str, Any]] | None = None,
                      question_embedding: list[float] | None = None,
                      corpus_fingerprint: str | None = None,
                      document_ids: list[str] | None = None,
                      response_status: str | None = None) -> dict[str, Any]:
        feedback_id, created_at = str(uuid.uuid4()), utc_now()
        normalized = " ".join(question.casefold().split())
        answer_hash = hashlib.sha256(answer.strip().encode("utf-8")).hexdigest() if answer.strip() else None
        with self.connection:
            self.connection.execute(
                """INSERT INTO feedback (
                    feedback_id, session_id, question, rating, comment, created_at,
                    normalized_question, question_embedding, answer, answer_hash,
                    confidence_json, sources_json, corpus_fingerprint, scope_json, response_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (feedback_id, session_id, question, rating, comment, created_at,
                 normalized, json.dumps(question_embedding) if question_embedding is not None else None,
                 answer, answer_hash, json.dumps(confidence) if confidence is not None else None,
                 json.dumps(sources or []), corpus_fingerprint,
                 json.dumps(sorted(document_ids)) if document_ids is not None else "[]", response_status),
            )
        return {"feedback_id": feedback_id, "status": "recorded", "created_at": created_at}

    def feedback_entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM feedback ORDER BY created_at DESC, rowid DESC").fetchall()
        entries = []
        for row in rows:
            item = dict(row)
            for field, default in (("question_embedding", None), ("confidence_json", None),
                                   ("sources_json", "[]"), ("scope_json", "[]")):
                raw = item.pop(field, None)
                target = field.removesuffix("_json")
                item[target] = json.loads(raw) if raw else default
            entries.append(item)
        return entries
