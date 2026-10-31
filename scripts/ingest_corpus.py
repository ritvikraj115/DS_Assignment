"""Fetch the small public FastAPI documentation corpus into the configured index."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from ingestion.service import IngestionService  # noqa: E402

log = logging.getLogger("corpus_ingestion")


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    manifest = ROOT / "corpus" / "sources.json"
    sources = json.loads(manifest.read_text(encoding="utf-8"))
    service = IngestionService()
    failures = 0
    try:
        for source in sources:
            log.info("corpus_source_started title=%s url=%s", source["title"], source["url"])
            try:
                result = service.ingest_url(source["url"], document_id=source["id"])
                log.info("corpus_source_completed title=%s status=%s chunks=%s", source["title"], result.status, result.chunk_count)
                print(f"{result.status:16} {result.chunk_count:4} chunks  {source['title']}")
            except Exception:
                failures += 1
                log.exception("corpus_source_failed title=%s", source["title"])
    finally:
        service.metadata.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
