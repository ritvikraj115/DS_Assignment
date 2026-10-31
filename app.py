from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from ingestion.config import IngestionConfig
from ingestion.metadata import MetadataStore
from ingestion.service import IngestionService
from retrieval.config import RetrievalConfig
from retrieval.service import RAGService
from retrieval.store import AppStore

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=100)


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    rating: Literal["up", "down"]
    session_id: str | None = Field(default=None, max_length=128)
    comment: str | None = Field(default=None, max_length=2000)


def create_app(rag_service: RAGService | None = None, ingestion_service: IngestionService | None = None,
               app_store: AppStore | None = None) -> FastAPI:
    app = FastAPI(title="Technical Documentation Assistant", version="1.0.0")
    cors_origins = [origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=False)
    app.state.rag_service = rag_service
    app.state.ingestion_service = ingestion_service or (rag_service.ingestion if rag_service else None)
    app.state.metadata_store = app.state.ingestion_service.metadata if app.state.ingestion_service else None
    app.state.app_store = app_store or (rag_service.app_store if rag_service else None)

    def runtime(require_ingestion: bool = False, require_rag: bool = False, require_metadata: bool = False):
        if not hasattr(app.state, "ingestion_config"):
            app.state.ingestion_config = IngestionConfig.from_env()
        if app.state.app_store is None:
            app.state.app_store = AppStore(app.state.ingestion_config.sqlite_path)
        if (require_metadata or require_ingestion or require_rag) and app.state.metadata_store is None:
            app.state.metadata_store = MetadataStore(app.state.ingestion_config.sqlite_path,
                app.state.ingestion_config.mongodb_uri, app.state.ingestion_config.mongodb_database)
        if require_ingestion and app.state.ingestion_service is None:
            app.state.ingestion_service = IngestionService(app.state.ingestion_config, metadata_store=app.state.metadata_store)
        if require_rag and app.state.rag_service is None:
            if app.state.ingestion_service is None:
                app.state.ingestion_service = IngestionService(app.state.ingestion_config, metadata_store=app.state.metadata_store)
            app.state.rag_service = RAGService(app.state.ingestion_service, RetrievalConfig.from_env(), app_store=app.state.app_store)
        elif app.state.rag_service is not None and app.state.rag_service.app_store is None:
            app.state.rag_service.app_store = app.state.app_store
        return app.state.rag_service, app.state.ingestion_service, app.state.app_store

    @app.get("/health")
    def health():
        log.info("health_check status=ok")
        return {"status": "ok"}

    @app.middleware("http")
    async def request_logging(request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started = perf_counter()
        log.info("http_request_started request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("http_request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
            raise
        response.headers["X-Request-ID"] = request_id
        log.info("http_request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                 request_id, request.method, request.url.path, response.status_code, (perf_counter() - started) * 1000)
        return response

    @app.post("/query")
    def query(http_request: Request, request: QueryRequest):
        try:
            rag, _, _ = runtime(require_ingestion=True, require_rag=True)
            question = request.question.strip()
            if len(question) < 2:
                raise HTTPException(status_code=422, detail="question must contain at least two non-whitespace characters")
            log.info("query_pipeline_started request_id=%s session_id=%s", http_request.state.request_id, request.session_id or "new")
            response = rag.query(question, request.session_id, document_ids=request.document_ids,
                                 top_k=request.top_k, request_id=http_request.state.request_id)
            log.info("query_pipeline_completed request_id=%s status=%s retries=%s generation_retries=%s",
                     http_request.state.request_id, response.get("status"), response.get("retry_count"), response.get("generation_retry_count"))
            if response.get("error"):
                return app_exception_response(response, status_code=503)
            return response
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as exc:
            log.exception("Query service unavailable")
            raise HTTPException(status_code=503, detail="The query service is temporarily unavailable") from exc
        except Exception as exc:
            log.exception("Query failed")
            raise HTTPException(status_code=500, detail="Query failed") from exc

    @app.post("/query/stream")
    def query_stream(http_request: Request, request: QueryRequest):
        try:
            rag, _, _ = runtime(require_ingestion=True, require_rag=True)
            question = request.question.strip()
            if len(question) < 2:
                raise HTTPException(status_code=422, detail="question must contain at least two non-whitespace characters")
            request_id = http_request.state.request_id
            log.info("query_stream_started request_id=%s session_id=%s", request_id, request.session_id or "new")

            def event_stream():
                for event in rag.query_stream(question, request.session_id, document_ids=request.document_ids,
                                              top_k=request.top_k, request_id=request_id):
                    event_name = event.pop("type")
                    yield f"event: {event_name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as exc:
            log.exception("Streaming query service unavailable")
            raise HTTPException(status_code=503, detail="The query service is temporarily unavailable") from exc
        except Exception as exc:
            log.exception("Could not start streaming query")
            raise HTTPException(status_code=500, detail="Query failed") from exc

    @app.post("/ingest")
    async def ingest(request: Request, file: UploadFile | None = File(default=None), url: str | None = Form(default=None)):
        if (file is None) == (url is None):
            raise HTTPException(status_code=400, detail="Provide exactly one of file or url")
        try:
            _, ingestion, _ = runtime(require_ingestion=True)
            log.info("ingestion_request_started request_id=%s kind=%s", request.state.request_id, "url" if url is not None else "file")
            if url is not None:
                result = ingestion.ingest_url(url).to_dict()
                log.info("ingestion_request_completed request_id=%s status=%s document_id=%s chunks=%s", request.state.request_id, result.get("status"), result.get("document_id"), result.get("chunk_count"))
                return result
            suffix = Path(file.filename or "").suffix
            with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(await file.read())
                temp_path = Path(temporary.name)
            try:
                result = ingestion.ingest_file(temp_path,
                    document_id=f"upload:{(file.filename or 'document').casefold()}",
                    filename=file.filename or "document").to_dict()
                log.info("ingestion_request_completed request_id=%s status=%s document_id=%s chunks=%s", request.state.request_id, result.get("status"), result.get("document_id"), result.get("chunk_count"))
                return result
            finally:
                temp_path.unlink(missing_ok=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Ingestion failed")
            raise HTTPException(status_code=503, detail="Document ingestion is temporarily unavailable") from exc

    @app.get("/documents")
    def documents():
        try:
            _, _, _ = runtime(require_metadata=True)
            metadata = app.state.metadata_store
            chunks = metadata.get_chunks()
            counts: dict[str, int] = {}
            for chunk in chunks:
                counts[chunk["document_id"]] = counts.get(chunk["document_id"], 0) + 1
            result = {"documents": [{
                "document_id": doc["document_id"], "filename": doc["filename"], "source_url": doc.get("source_url"),
                "content_hash": doc["content_hash"], "version": doc["version"], "created_at": doc["created_at"],
                "updated_at": doc["updated_at"], "chunk_count": counts.get(doc["document_id"], 0),
            } for doc in metadata.list_documents()]}
            log.info("documents_listed count=%s", len(result["documents"]))
            return result
        except Exception as exc:
            log.exception("Could not list indexed documents")
            raise HTTPException(status_code=503, detail="Document metadata is temporarily unavailable") from exc

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        try:
            _, _, store = runtime()
            result = store.save_feedback(request.session_id, request.question.strip(), request.rating, request.comment)
            log.info("feedback_recorded feedback_id=%s rating=%s", result["feedback_id"], request.rating)
            return result
        except Exception as exc:
            log.exception("Could not store feedback")
            raise HTTPException(status_code=503, detail="Feedback could not be stored") from exc

    return app


def app_exception_response(response: dict, status_code: int):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status_code, content=response)


app = create_app()
