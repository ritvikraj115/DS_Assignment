from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from time import perf_counter
from urllib.parse import urldefrag, urlparse

from .chunker import StructureAwareChunker
from .config import IngestionConfig
from .embedder import SentenceTransformerEmbedder
from .indexer import FaissIndexer
from .metadata import MetadataStore
from .models import Chunk, IngestionResult, utc_now
from .parsers import parse_file, parse_url

log = logging.getLogger(__name__)


class IngestionService:
    """Reusable ingestion entry point, ready to be called from a FastAPI route."""

    def __init__(self, config: IngestionConfig | None = None, embedder=None, metadata_store=None, indexer=None):
        self.config = config or IngestionConfig.from_env()
        self.config.validate()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.embedding_model)
        self.metadata = metadata_store or MetadataStore(self.config.sqlite_path, self.config.mongodb_uri, self.config.mongodb_database)
        self.indexer = indexer or FaissIndexer(self.config.faiss_index_path)
        # Metadata is the recovery source of truth after an interrupted index promotion.
        self.indexer.rebuild(self.metadata.get_chunks())
        self.chunker = StructureAwareChunker(self.config.chunk_size, self.config.chunk_overlap)

    @staticmethod
    def _file_id(path: Path) -> str:
        return "file:" + str(path.resolve()).casefold()

    @staticmethod
    def _url_id(url: str) -> str:
        normalized, _ = urldefrag(url)
        parsed = urlparse(normalized)
        normalized = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl()
        return "url:" + normalized

    def ingest_file(self, path: str | Path, document_id: str | None = None, filename: str | None = None) -> IngestionResult:
        file_path = Path(path)
        parsed = parse_file(file_path)
        if filename:
            parsed.filename = filename
        return self._ingest(parsed, document_id or self._file_id(file_path.resolve()))

    def ingest_url(self, url: str, document_id: str | None = None) -> IngestionResult:
        parsed = parse_url(url, self.config.request_timeout)
        return self._ingest(parsed, document_id or self._url_id(url))

    def _ingest(self, parsed, document_id: str) -> IngestionResult:
        started = perf_counter()
        log.info("ingestion_started document_id=%s filename=%s source_url=%s", document_id, parsed.filename, bool(parsed.source_url))
        if not parsed.text.strip() or not parsed.blocks:
            log.warning("ingestion_rejected document_id=%s reason=empty_document", document_id)
            raise ValueError("Document is empty after parsing")
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        previous = self.metadata.get_document(document_id)
        if previous and previous["content_hash"] == content_hash:
            log.info("ingestion_skipped_unchanged document_id=%s version=%s", document_id, previous["version"])
            return IngestionResult(document_id, "already_indexed", len(previous.get("chunks", [])),
                                   retained_chunks=len(previous.get("chunks", [])), message="Content is unchanged.",
                                   updated_at=previous.get("updated_at"))

        proposed = self.chunker.chunk(parsed.blocks, document_id)
        log.info("chunking_completed document_id=%s chunks=%s", document_id, len(proposed))
        old_chunks = previous.get("chunks", []) if previous else []
        # Keep vector IDs and embeddings for matching text hashes and section metadata.
        available: dict[tuple[str, str], list[dict]] = {}
        for old in old_chunks:
            key = (old["content_hash"], str(old["section"]))
            available.setdefault(key, []).append(old)
        retained = 0
        new_chunks: list[Chunk] = []
        for chunk in proposed:
            key = (chunk.content_hash, str(chunk.section))
            matches = available.get(key, [])
            if matches:
                old = matches.pop(0)
                chunk.vector_id, chunk.embedding = old["vector_id"], old["embedding"]
                chunk.chunk_id = old["chunk_id"]
                chunk.created_at = old.get("created_at", chunk.created_at)
                chunk.updated_at = old.get("updated_at", chunk.updated_at)
                retained += 1
            else:
                new_chunks.append(chunk)

        if new_chunks:
            log.info("embedding_started document_id=%s chunks=%s model=%s", document_id, len(new_chunks), self.config.embedding_model)
            vectors = self.embedder.encode([chunk.text for chunk in new_chunks])
            if len(vectors) != len(new_chunks):
                raise RuntimeError("Embedder returned an unexpected number of vectors")
            ids = self.metadata.next_vector_ids(len(new_chunks))
            for chunk, vector, vector_id in zip(new_chunks, vectors, ids):
                chunk.embedding, chunk.vector_id = [float(v) for v in vector], vector_id
            log.info("embedding_completed document_id=%s vectors=%s", document_id, len(vectors))
        chunks = proposed
        now = utc_now()
        document = {
            "document_id": document_id, "filename": parsed.filename, "source_url": parsed.source_url,
            "content_hash": content_hash, "version": (previous["version"] + 1) if previous else 1,
            "created_at": previous["created_at"] if previous else now, "updated_at": now,
        }
        existing_chunks = self.metadata.get_chunks()
        all_chunks = [c for c in existing_chunks if c["document_id"] != document_id]
        all_chunks.extend(c.to_dict() for c in chunks)
        staged = self.indexer.prepare(all_chunks)
        try:
            log.info("faiss_index_staged document_id=%s total_vectors=%s", document_id, len(all_chunks))
            self.metadata.replace_document(document, chunks)
            log.info("metadata_committed document_id=%s version=%s chunk_count=%s", document_id, document["version"], len(chunks))
            self.indexer.promote(staged)
            log.info("faiss_index_promoted document_id=%s", document_id)
        except Exception:
            log.exception("ingestion_commit_failed document_id=%s", document_id)
            if staged is not None and staged.exists():
                staged.unlink(missing_ok=True)
            # Restore prior metadata when index promotion fails. Startup also rebuilds from metadata.
            if previous:
                prior = [Chunk(c["chunk_id"], c["text"], c["section"], c["content_hash"], c["vector_id"], c["embedding"],
                               c.get("created_at", previous["updated_at"]), c.get("updated_at", previous["updated_at"])) for c in old_chunks]
                self.metadata.replace_document(previous, prior)
            else:
                self.metadata.delete_document(document_id)
            try:
                self.indexer.rebuild(existing_chunks)
            except Exception:
                # Persisted metadata remains authoritative; startup will repair the index.
                pass
            raise
        result = IngestionResult(document_id, "indexed" if not previous else "updated", len(chunks),
                               added_chunks=len(new_chunks), retained_chunks=retained,
                               removed_chunks=max(0, len(old_chunks) - retained), updated_at=now,
                               message="Document indexed successfully.")
        log.info("ingestion_completed document_id=%s status=%s added=%s retained=%s removed=%s duration_ms=%.1f",
                 document_id, result.status, result.added_chunks, result.retained_chunks, result.removed_chunks,
                 (perf_counter() - started) * 1000)
        return result
