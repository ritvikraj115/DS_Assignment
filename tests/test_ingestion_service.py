from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion.metadata import MetadataStore
from ingestion.service import IngestionService
from retrieval.store import AppStore


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeIndexer:
    def __init__(self):
        self.current = []
        self.staged = None

    def rebuild(self, chunks):
        self.current = list(chunks)

    def prepare(self, chunks):
        self.staged = list(chunks)
        return None

    def promote(self, staged):
        self.current = self.staged


def make_service(tmp_path: Path):
    config = IngestionConfig(chunk_size=120, chunk_overlap=10, data_dir=tmp_path,
                             faiss_index_path=tmp_path / "index.faiss", sqlite_path=tmp_path / "metadata.sqlite3")
    embedder, indexer = FakeEmbedder(), FakeIndexer()
    metadata = MetadataStore(config.sqlite_path)
    return IngestionService(config, embedder, metadata, indexer), embedder, indexer, metadata


def test_new_and_unchanged_document(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Setup\n\nInstall the package and configure the client.", encoding="utf-8")
    service, embedder, indexer, metadata = make_service(tmp_path / "state")

    first = service.ingest_file(source)
    second = service.ingest_file(source)

    assert first.status == "indexed"
    assert first.added_chunks == first.chunk_count
    assert second.status == "already_indexed"
    assert len(embedder.calls) == 1
    assert len(metadata.get_chunks()) == first.chunk_count
    assert {c["vector_id"] for c in indexer.current} == {c["vector_id"] for c in metadata.get_chunks()}


def test_update_reuses_unchanged_chunks_and_drops_obsolete_ones(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Setup\n\nInstall the package.\n\n# Usage\n\nRun the client.", encoding="utf-8")
    service, embedder, indexer, metadata = make_service(tmp_path / "state")
    first = service.ingest_file(source)
    first_chunks = {c["text"]: c for c in metadata.get_chunks(first.document_id)}

    source.write_text("# Setup\n\nInstall the package.\n\n# Usage\n\nRun the updated client.", encoding="utf-8")
    update = service.ingest_file(source)
    updated_chunks = metadata.get_chunks(first.document_id)

    assert update.status == "updated"
    assert update.retained_chunks >= 1
    assert update.added_chunks == 1
    assert update.removed_chunks == 1
    unchanged_text = next(text for text in first_chunks if "Install the package." in text)
    unchanged_now = next(c for c in updated_chunks if c["text"] == unchanged_text)
    assert unchanged_now["vector_id"] == first_chunks[unchanged_text]["vector_id"]
    assert len(embedder.calls) == 2
    assert len(indexer.current) == len(updated_chunks)
    assert {c["vector_id"] for c in indexer.current} == {c["vector_id"] for c in updated_chunks}
    assert all("Run the client." not in c["text"] for c in indexer.current)


def test_updated_chunks_invalidate_dependent_feedback_and_conversation_memory(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nOriginal setup instructions.", encoding="utf-8")
    service, _, _, metadata = make_service(tmp_path / "state")
    memory = AppStore(tmp_path / "state" / "app.sqlite3")
    service.memory_store = memory
    indexed = service.ingest_file(source)
    old_chunk = metadata.get_chunks(indexed.document_id)[0]
    document = metadata.get_document(indexed.document_id)
    sources = [{"document_id": indexed.document_id, "document_version": document["version"],
                "chunk_id": old_chunk["chunk_id"], "chunk_hash": old_chunk["content_hash"]}]
    memory.save_exchange("session-1", "How do I set this up?", "Original answer.", {"sources": sources})
    memory.save_feedback("session-1", "How do I set this up?", "down", None,
                         answer="Original answer.", confidence={"overall": 40}, sources=sources,
                         question_embedding=[1.0, 0.0], corpus_fingerprint="old",
                         response_status="unverified")
    memory.save_exchange("session-2", "Unrelated question", "Unrelated answer.", {
        "sources": [{"document_id": "other-doc", "chunk_id": "keep-chunk"}],
    })

    source.write_text("# Guide\n\nCompletely revised setup instructions.", encoding="utf-8")
    service.ingest_file(source)

    assert memory.feedback_entries() == []
    assert memory.history("session-1") == []
    assert memory.history("session-2") == [
        {"role": "user", "content": "Unrelated question"},
        {"role": "assistant", "content": "Unrelated answer."},
    ]
