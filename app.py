from __future__ import annotations

import logging
import os
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    stream: bool = False


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    rating: Literal["up", "down"]
    session_id: str | None = Field(default=None, max_length=128)
    comment: str | None = Field(default=None, max_length=2000)
    answer: str = Field(default="", max_length=40000)
    confidence: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    response_status: str | None = Field(default=None, max_length=40)
    document_ids: list[str] | None = Field(default=None, max_length=100)


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
        if app.state.ingestion_service is not None:
            app.state.ingestion_service.memory_store = app.state.app_store
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
            if request.stream or "text/event-stream" in http_request.headers.get("accept", ""):
                request_id = http_request.state.request_id

                def event_stream():
                    for event in rag.query_stream(question, request.session_id, document_ids=request.document_ids,
                                                  top_k=request.top_k, request_id=request_id):
                        event_name = event.pop("type")
                        yield f"event: {event_name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
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
            rag, ingestion, store = runtime(require_ingestion=True, require_rag=True)
            question = request.question.strip()
            verified = store.verified_exchange(request.session_id, question, request.answer)
            if not verified:
                raise HTTPException(status_code=409, detail="Feedback must refer to an answer from this conversation")
            documents = ingestion.metadata.list_documents()
            scoped = documents if request.document_ids is None else [
                document for document in documents if document["document_id"] in set(request.document_ids)
            ]
            corpus_snapshot = sorted((doc["document_id"], doc.get("version"), doc.get("content_hash")) for doc in scoped)
            corpus_fingerprint = hashlib.sha256(json.dumps(corpus_snapshot, separators=(",", ":")).encode("utf-8")).hexdigest()
            feedback_question = verified.get("feedback_question") or question
            vectors = ingestion.embedder.encode([feedback_question])
            if len(vectors) != 1:
                raise RuntimeError("Could not encode question for semantic feedback matching")
            canonical_question = rag.canonical_feedback_question(
                feedback_question, request.rating, request.document_ids, vectors[0])
            canonical_vectors = ingestion.embedder.encode([canonical_question]) if canonical_question != feedback_question else vectors
            if len(canonical_vectors) != 1:
                raise RuntimeError("Could not encode canonical feedback question")
            # Answer metadata comes from the server-side conversation record, not client-submitted source fields.
            result = store.save_feedback(
                request.session_id, canonical_question, request.rating, request.comment,
                answer=request.answer, confidence=verified.get("confidence"),
                sources=verified.get("sources") or verified.get("memory_sources", []),
                question_embedding=canonical_vectors[0], corpus_fingerprint=corpus_fingerprint,
                document_ids=request.document_ids, response_status=verified.get("status"),
            )
            log.info("feedback_recorded feedback_id=%s rating=%s", result["feedback_id"], request.rating)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Could not store feedback")
            raise HTTPException(status_code=503, detail="Feedback could not be stored") from exc

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        try:
            _, _, store = runtime()
            store.delete_session(session_id)
            log.info("conversation_deleted session_id=%s", session_id)
            return {"status": "deleted", "session_id": session_id}
        except Exception as exc:
            log.exception("Could not delete conversation session")
            raise HTTPException(status_code=503, detail="Conversation could not be deleted") from exc

    return app


def app_exception_response(response: dict, status_code: int):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status_code, content=response)


app = create_app()
