from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    original_query: str
    standalone_query: str
    rewritten_query: str
    session_id: str | None
    request_id: str | None
    chat_history: list[dict[str, str]]
    document_ids: list[str] | None
    top_k: int
    retrieved_documents: list[dict[str, Any]]
    graded_documents: list[dict[str, Any]]
    relevant_documents: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    memory_sources: list[dict[str, Any]]
    evaluation_scores: dict[str, Any]
    confidence_score: int | None
    retry_count: int
    generation_retry_count: int
    fallback_used: bool
    error: str | None
    response_status: str
    correction_feedback: str
    should_retry: bool
    should_regenerate: bool
    stream_output: bool
    progress_callback: Any
    cancel_event: Any
    search_scope_empty: bool

