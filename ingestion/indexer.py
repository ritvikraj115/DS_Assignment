from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


class FaissIndexer:
    """Builds replacement indexes off to the side and promotes them atomically."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.dimension: int | None = None

    @staticmethod
    def _faiss():
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Vector indexing requires faiss-cpu and numpy; install requirements.txt") from exc
        return faiss, np

    def prepare(self, chunks: list[dict[str, Any]]) -> Path | None:
        vectors = [c for c in chunks if c.get("embedding") is not None]
        if not vectors:
            if self.index_path.exists():
                temp = self.index_path.with_suffix(self.index_path.suffix + ".empty")
                faiss, _ = self._faiss()
                faiss.write_index(faiss.IndexIDMap2(faiss.IndexFlatIP(1)), str(temp))
                return temp
            return None
        faiss, np = self._faiss()
        dimension = len(vectors[0]["embedding"])
        if any(len(c["embedding"]) != dimension for c in vectors):
            raise ValueError("Embedding dimensions are inconsistent")
        base = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap2(base)
        matrix = np.asarray([c["embedding"] for c in vectors], dtype="float32")
        ids = np.asarray([c["vector_id"] for c in vectors], dtype="int64")
        index.add_with_ids(matrix, ids)
        fd, temp_name = tempfile.mkstemp(prefix="faiss-stage-", suffix=".index", dir=self.index_path.parent)
        os.close(fd)
        temp = Path(temp_name)
        faiss.write_index(index, str(temp))
        return temp

    def promote(self, staged_path: Path | None) -> None:
        if staged_path is not None:
            os.replace(staged_path, self.index_path)
        if self.index_path.exists():
            faiss, _ = self._faiss()
            self.index = faiss.read_index(str(self.index_path))

    def rebuild(self, chunks: list[dict[str, Any]]) -> None:
        staged = self.prepare(chunks)
        self.promote(staged)

    def search(self, vector: list[float], top_k: int) -> tuple[list[int], list[float]]:
        if self.index is None:
            return [], []
        _, np = self._faiss()
        query = np.asarray([vector], dtype="float32")
        scores, ids = self.index.search(query, max(1, top_k))
        return [int(value) for value in ids[0] if int(value) >= 0], [float(score) for value, score in zip(ids[0], scores[0]) if int(value) >= 0]
