from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Section:
    title: str
    level: int = 0
    path: list[str] = field(default_factory=list)


@dataclass
class ContentBlock:
    kind: str
    text: str
    section: Section


@dataclass
class ParsedDocument:
    text: str
    blocks: list[ContentBlock]
    filename: str
    source_url: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: dict[str, Any]
    content_hash: str
    vector_id: int | None = None
    embedding: list[float] | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionResult:
    document_id: str
    status: str
    chunk_count: int
    added_chunks: int = 0
    retained_chunks: int = 0
    removed_chunks: int = 0
    message: str = ""
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
