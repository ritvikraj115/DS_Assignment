"""Optional FastAPI adapter. Import this module only in applications using FastAPI."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .service import IngestionService


def create_ingest_router(service: IngestionService) -> APIRouter:
    router = APIRouter()

    @router.post("/ingest")
    async def ingest(file: UploadFile | None = File(default=None), url: str | None = Form(default=None)):
        if (file is None) == (url is None):
            raise HTTPException(status_code=400, detail="Provide exactly one of file or url")
        try:
            if url:
                return service.ingest_url(url).to_dict()
            suffix = Path(file.filename or "").suffix
            with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(await file.read())
                temp_path = Path(temporary.name)
            try:
                return service.ingest_file(temp_path,
                    document_id=f"upload:{(file.filename or 'document').casefold()}",
                    filename=file.filename or "document").to_dict()
            finally:
                temp_path.unlink(missing_ok=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return router
