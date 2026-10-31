from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import Chunk, utc_now


class MetadataStore:
    """Metadata repository. MongoDB is selected only when MONGODB_URI is set."""

    def __init__(self, sqlite_path: Path, mongodb_uri: str | None = None, mongodb_database: str = "technical_docs"):
        self.mongo = None
        if mongodb_uri:
            try:
                from pymongo import MongoClient
            except ImportError as exc:
                raise RuntimeError("MONGODB_URI is configured but pymongo is not installed") from exc
            self.mongo = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)[mongodb_database]
            self.mongo.documents.create_index("document_id", unique=True)
            self.mongo.documents.create_index("chunks.vector_id")
            self.mongo.counters.update_one({"_id": "vector_id"}, {"$setOnInsert": {"value": 0}}, upsert=True)
        else:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(sqlite_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, source_url TEXT,
                    content_hash TEXT NOT NULL, version INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    text TEXT NOT NULL, section_json TEXT NOT NULL, content_hash TEXT NOT NULL,
                    vector_id INTEGER NOT NULL UNIQUE, embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO counters(name, value) VALUES ('vector_id', 0);
            """)
            self.connection.commit()

    def next_vector_ids(self, count: int) -> list[int]:
        if not count:
            return []
        if self.mongo is not None:
            from pymongo import ReturnDocument
            end = self.mongo.counters.find_one_and_update({"_id": "vector_id"}, {"$inc": {"value": count}}, return_document=ReturnDocument.AFTER)["value"]
            return list(range(end - count + 1, end + 1))
        with self.connection:
            row = self.connection.execute("SELECT value FROM counters WHERE name='vector_id'").fetchone()
            start = int(row[0]) + 1
            self.connection.execute("UPDATE counters SET value=? WHERE name='vector_id'", (start + count - 1,))
        return list(range(start, start + count))

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        if self.mongo is not None:
            doc = self.mongo.documents.find_one({"document_id": document_id}, {"_id": 0})
            return doc
        doc = self.connection.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        if doc is None:
            return None
        result = dict(doc)
        result["chunks"] = self.get_chunks(document_id)
        return result

    def get_chunks(self, document_id: str | None = None) -> list[dict[str, Any]]:
        if self.mongo is not None:
            docs = self.mongo.documents.find({} if document_id is None else {"document_id": document_id}, {"chunks": 1, "_id": 0})
            return [chunk for doc in docs for chunk in doc.get("chunks", [])]
        query = "SELECT * FROM chunks" + (" WHERE document_id=?" if document_id is not None else "")
        rows = self.connection.execute(query, (() if document_id is None else (document_id,))).fetchall()
        return [{"chunk_id": r["chunk_id"], "document_id": r["document_id"], "text": r["text"],
                 "section": json.loads(r["section_json"]), "content_hash": r["content_hash"],
                 "vector_id": r["vector_id"], "embedding": json.loads(r["embedding_json"]),
                 "created_at": r["created_at"], "updated_at": r["updated_at"]} for r in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        if self.mongo is not None:
            return list(self.mongo.documents.find({}, {"_id": 0}))
        return [dict(row) for row in self.connection.execute("SELECT * FROM documents").fetchall()]

    def replace_document(self, document: dict[str, Any], chunks: list[Chunk]) -> None:
        now = utc_now()
        if self.mongo is not None:
            value = dict(document)
            value["chunks"] = [chunk.to_dict() for chunk in chunks]
            self.mongo.documents.replace_one({"document_id": document["document_id"]}, value, upsert=True)
            return
        conn = self.connection
        with conn:
            conn.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (document["document_id"], document["filename"], document.get("source_url"),
                          document["content_hash"], document["version"], document["created_at"], document["updated_at"] or now))
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document["document_id"],))
            conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (c.chunk_id, document["document_id"], c.text, json.dumps(c.section), c.content_hash,
                 c.vector_id, json.dumps(c.embedding), c.created_at, c.updated_at) for c in chunks
            ])

    def delete_document(self, document_id: str) -> None:
        if self.mongo is not None:
            self.mongo.documents.delete_one({"document_id": document_id})
        else:
            with self.connection:
                self.connection.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                self.connection.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

    def close(self) -> None:
        if self.mongo is None:
            self.connection.close()
