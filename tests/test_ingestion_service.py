from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion.metadata import MetadataStore
from ingestion.service import IngestionService


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
