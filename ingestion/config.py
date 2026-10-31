from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/index.faiss")
    sqlite_path: Path = Path("data/metadata.sqlite3")
    mongodb_uri: str | None = None
    mongodb_database: str = "technical_docs"
    request_timeout: int = 20

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        data_dir = Path(os.getenv("INGESTION_DATA_DIR", "data"))
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            data_dir=data_dir,
            faiss_index_path=Path(os.getenv("FAISS_INDEX_PATH", str(data_dir / "index.faiss"))),
            sqlite_path=Path(os.getenv("SQLITE_PATH", str(data_dir / "metadata.sqlite3"))),
            mongodb_uri=os.getenv("MONGODB_URI") or None,
            mongodb_database=os.getenv("MONGODB_DATABASE", "technical_docs"),
            request_timeout=int(os.getenv("INGESTION_REQUEST_TIMEOUT", "20")),
        )

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
