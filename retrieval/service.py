from __future__ import annotations

import json
import hashlib
import logging
import re
from queue import Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from ingestion.service import IngestionService

from .config import RetrievalConfig
from .llm import GeminiClient, ModelClient, ModelError
from .models import RAGState
from .prompts import (EVALUATION_PROMPT, FEEDBACK_ANSWER_PROMPT, GENERATION_PROMPT,
                      GRADE_PROMPT, QUERY_REWRITE_PROMPT)
from .schemas import (AnswerEvaluationOutput, DocumentGradingOutput, FeedbackAnswerOutput,
                      QueryRewriteOutput, StrictOutput,
                      parse_structured_output)

log = logging.getLogger(__name__)
INSUFFICIENT = "I couldn't find enough relevant documentation to answer that question reliably."
UNVERIFIED = "I couldn't verify an answer against the documentation right now, so I won't present an unverified response."


class QueryCancelled(Exception):
    """Raised between pipeline steps after an SSE client disconnects."""


class RAGService:
    def __init__(self, ingestion: IngestionService, config: RetrievalConfig | None = None,
                 model_client: ModelClient | None = None, app_store=None):
        self.ingestion = ingestion
        self.config = config or RetrievalConfig.from_env()
        self.config.validate()
        self.model_client = model_client or GeminiClient(
            self.config.gemini_api_key, self.config.gemini_timeout_ms,
            model_fallbacks=self._fallback_model_map(self.config),
        )
        self.app_store = app_store
        self.graph = self._build_graph()

    @staticmethod
    def _fallback_model_map(config: RetrievalConfig) -> dict[str, tuple[str, ...]]:
        """Merge role fallback lists when both roles share a primary model."""
        merged: dict[str, list[str]] = {}
        for primary, fallbacks in (
            (config.generation_model, config.generation_fallback_models),
            (config.judge_model, config.judge_fallback_models),
        ):
            chain = merged.setdefault(primary, [])
            chain.extend(model for model in fallbacks if model not in chain and model != primary)
        return {primary: tuple(chain) for primary, chain in merged.items()}

    @staticmethod
    def _progress(state: RAGState, stage: str, detail: str) -> None:
        cancel_event = state.get("cancel_event")
        if cancel_event and cancel_event.is_set():
            raise QueryCancelled()
        callback = state.get("progress_callback")
        if callback:
            callback({"type": "stage", "stage": stage, "detail": detail})

    def _generate_structured(self, model: str, fallbacks: tuple[str, ...], prompt: str,
                             schema: type[StrictOutput], validate=None):
        """Try configured role fallbacks when a model returns JSON that fails its schema."""
        candidates = list(dict.fromkeys(candidate for candidate in (model, *fallbacks) if candidate))
        last_error = None
        for candidate in candidates:
            raw = self.model_client.generate(candidate, prompt, json_mode=True)
            try:
                data = parse_structured_output(raw, schema)
                return validate(data) if validate else data
            except ModelError as exc:
                last_error = exc
                log.warning("structured_output_rejected model=%s schema=%s reason=%s",
                            candidate, schema.__name__, exc)
        raise last_error or ModelError(f"No model configured for {schema.__name__}")

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("query_analysis", self._query_analysis)
        graph.add_node("retrieval", self._retrieval)
        graph.add_node("document_grading", self._document_grading)
        graph.add_node("generation", self._generation)
        graph.add_node("answer_evaluation", self._answer_evaluation)
        graph.add_node("insufficient_evidence", self._insufficient_evidence)
        graph.add_node("safe_response", self._safe_response)
        graph.add_node("final_response", self._final_response)
        graph.add_edge(START, "query_analysis")
        graph.add_edge("query_analysis", "retrieval")
        graph.add_edge("retrieval", "document_grading")
        graph.add_conditional_edges("document_grading", self._after_grading, {
            "generate": "generation", "rewrite": "query_analysis", "insufficient": "insufficient_evidence", "safe": "safe_response",
        })
        graph.add_conditional_edges("generation", lambda state: "safe" if state.get("error") else "evaluate",
                                    {"evaluate": "answer_evaluation", "safe": "safe_response"})
        graph.add_conditional_edges("answer_evaluation", self._after_evaluation, {
            "passed": "final_response", "regenerate": "generation", "safe": "safe_response",
        })
        graph.add_edge("insufficient_evidence", END)
        graph.add_edge("safe_response", END)
        graph.add_edge("final_response", END)
        return graph.compile()

    def _query_analysis(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        original = state["original_query"]
        self._progress(state, "query_analysis", "Understanding your question and conversation context…")
        if state.get("retry_count", 0):
            self._progress(state, "retrieval_retry", f"The first search found no suitable sources. Trying a revised search ({state['retry_count'] + 1})…")
        log.info("graph_node_started request_id=%s node=query_analysis retry_count=%s history_messages=%s",
                 state.get("request_id"), state.get("retry_count", 0), len(state.get("chat_history", [])))
        indexed_chunks = self.ingestion.metadata.get_chunks()
        requested_documents = state.get("document_ids")
        requested_set = set(requested_documents) if requested_documents is not None else None
        if not indexed_chunks or (requested_set is not None and not any(
                chunk["document_id"] in requested_set for chunk in indexed_chunks)):
            self._progress(state, "insufficient_evidence", "There are no indexed chunks in the selected document scope.")
            return {"standalone_query": state.get("standalone_query") or original,
                    "rewritten_query": state.get("rewritten_query") or original,
                    "search_scope_empty": True, "error": None}
        try:
            prompt = QUERY_REWRITE_PROMPT.format(
                original_query=original, standalone_query=state.get("standalone_query", original),
                retry_count=state.get("retry_count", 0), chat_history=json.dumps(state.get("chat_history", []), ensure_ascii=False),
            )

            def validate_rewrite(data):
                if not data.standalone_query.strip() or not data.rewritten_query.strip():
                    raise ModelError("Query rewrite response contains empty questions")
                return data

            data = self._generate_structured(self.config.generation_model,
                                             self.config.generation_fallback_models,
                                             prompt, QueryRewriteOutput, validate_rewrite)
            standalone = data.standalone_query.strip()
            rewritten = data.rewritten_query.strip()
            log.info("graph_node_completed request_id=%s node=query_analysis retry_count=%s duration_ms=%.1f",
                     state.get("request_id"), state.get("retry_count", 0), (perf_counter() - started) * 1000)
            return {"standalone_query": standalone, "rewritten_query": rewritten, "error": None}
        except ModelError as exc:
            if state.get("retry_count", 0) == 0:
                # Follow-up resolution can fail open to the original query; retrieval is still grounded.
                log.warning("Query analysis failed; using original question: %s", exc)
                return {"standalone_query": state.get("standalone_query") or original,
                        "rewritten_query": state.get("standalone_query") or original, "error": None}
            # Retrieval can still use the last standalone query when rewriting is unavailable.
            log.warning("Query rewrite unavailable; retaining prior question: %s", exc)
            return {"standalone_query": state.get("standalone_query") or original,
                    "rewritten_query": state.get("standalone_query") or original, "error": None}

    def _retrieval(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        self._progress(state, "retrieval", "Searching the indexed documentation…")
        log.info("graph_node_started request_id=%s node=retrieval retry_count=%s top_k=%s", state.get("request_id"), state.get("retry_count", 0), state.get("top_k"))
        if state.get("search_scope_empty"):
            return {"retrieved_documents": [], "search_scope_empty": True}
        if state.get("error"):
            return {"retrieved_documents": []}
        all_chunks = self.ingestion.metadata.get_chunks()
        document_ids = state.get("document_ids")
        allowed = set(document_ids) if document_ids is not None else None
        if not all_chunks or (allowed is not None and not any(chunk["document_id"] in allowed for chunk in all_chunks)):
            return {"retrieved_documents": [], "search_scope_empty": True}
        query = state.get("rewritten_query") or state["standalone_query"]
        try:
            vectors = self.ingestion.embedder.encode([query])
        except Exception as exc:
            log.exception("Query embedding failed")
            return {"error": "embedding_failed", "retrieved_documents": []}
        if len(vectors) != 1:
            return {"error": "embedding_failed", "retrieved_documents": []}
        by_vector = {int(c["vector_id"]): c for c in all_chunks}
        doc_info = {d["document_id"]: d for d in self.ingestion.metadata.list_documents()}
        try:
            if allowed is not None:
                # Filtering candidates before final top-k keeps document filtering exact.
                candidate_ids = {int(c["vector_id"]) for c in all_chunks if c["document_id"] in allowed}
                if not candidate_ids:
                    return {"retrieved_documents": []}
                search_ids, scores = self.ingestion.indexer.search(vectors[0], len(all_chunks))
                hits = [(vid, score) for vid, score in zip(search_ids, scores) if vid in candidate_ids][:state.get("top_k", self.config.top_k)]
            else:
                search_ids, scores = self.ingestion.indexer.search(vectors[0], state.get("top_k", self.config.top_k))
                hits = list(zip(search_ids, scores))
        except Exception:
            log.exception("FAISS retrieval failed")
            return {"error": "retrieval_unavailable", "retrieved_documents": []}
        result = []
        for vector_id, score in hits:
            chunk = by_vector.get(vector_id)
            if not chunk:
                continue
            doc = doc_info.get(chunk["document_id"], {})
            result.append({
                "document_id": chunk["document_id"], "document": doc.get("filename", chunk["document_id"]),
                "source_url": doc.get("source_url"), "document_version": doc.get("version"),
                "section": chunk.get("section", {}).get("title", "Document"),
                "section_path": chunk.get("section", {}).get("path", []),
                "chunk_id": chunk["chunk_id"], "chunk_hash": chunk.get("content_hash"),
                "vector_id": vector_id, "text": chunk["text"], "similarity": score,
            })
        log.info("graph_node_completed request_id=%s node=retrieval hits=%s retry_count=%s duration_ms=%.1f",
                 state.get("request_id"), len(result), state.get("retry_count", 0), (perf_counter() - started) * 1000)
        return {"retrieved_documents": result, "search_scope_empty": False}

    def _document_grading(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        docs = state.get("retrieved_documents", [])
        self._progress(state, "source_review", f"Checking {len(docs)} retrieved source{'s' if len(docs) != 1 else ''} for relevance…")
        log.info("graph_node_started request_id=%s node=document_grading retrieved=%s", state.get("request_id"), len(docs))
        if not docs:
            retry = (not state.get("search_scope_empty", False)
                     and state.get("retry_count", 0) < self.config.max_retrieval_retries)
            log.info("graph_node_completed request_id=%s node=document_grading relevant=0 retry=%s duration_ms=%.1f",
                     state.get("request_id"), retry, (perf_counter() - started) * 1000)
            return {"graded_documents": [], "relevant_documents": [], "should_retry": retry,
                    "retry_count": state.get("retry_count", 0) + int(retry),
                    "fallback_used": state.get("fallback_used", False) or retry}
        compact = [{k: d[k] for k in ("chunk_id", "document", "section", "text")} for d in docs]
        try:
            prompt = GRADE_PROMPT.format(
                question=state["standalone_query"], documents=json.dumps(compact, ensure_ascii=False),
            )

            def attach_grades(data):
                by_id = {grade.chunk_id: grade for grade in data.grades}
                expected_ids = {doc["chunk_id"] for doc in docs}
                if set(by_id) != expected_ids:
                    raise ModelError("Document grading response does not match retrieved chunks")
                return [{**doc, "relevant": by_id[doc["chunk_id"]].relevant,
                         "reason": by_id[doc["chunk_id"]].reason} for doc in docs]

            graded = self._generate_structured(
                self.config.judge_model, self.config.judge_fallback_models, prompt,
                DocumentGradingOutput, attach_grades,
            )
            relevant = [doc for doc in graded if doc["relevant"]]
            retry = not relevant and state.get("retry_count", 0) < self.config.max_retrieval_retries
            log.info("graph_node_completed request_id=%s node=document_grading relevant=%s retry=%s duration_ms=%.1f",
                     state.get("request_id"), len(relevant), retry, (perf_counter() - started) * 1000)
            return {"graded_documents": graded, "relevant_documents": relevant, "error": None,
                    "should_retry": retry, "retry_count": state.get("retry_count", 0) + int(retry),
                    "fallback_used": state.get("fallback_used", False) or retry}
        except ModelError as exc:
            log.error("Document judge unavailable after %.1f ms: %s", (perf_counter() - started) * 1000, exc)
            return {"error": "judge_unavailable", "graded_documents": [], "relevant_documents": []}

    def _after_grading(self, state: RAGState) -> str:
        if state.get("error"):
            return "safe"
        if state.get("relevant_documents"):
            return "generate"
        if state.get("should_retry"):
            return "rewrite"
        return "insufficient"

    @staticmethod
    def _source_list(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen, sources = set(), []
        for doc in docs:
            key = doc["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            sources.append({"document": doc["document"], "section": doc["section"],
                            "chunk_id": doc["chunk_id"], "document_id": doc["document_id"],
                            "document_version": doc.get("document_version"), "chunk_hash": doc.get("chunk_hash"),
                            **({"source_url": doc["source_url"]} if doc.get("source_url") else {})})
        return sources

    @staticmethod
    def _remove_inline_source_tags(answer: str) -> str:
        return re.sub(r"\s*\[S\d*\]?", "", answer, flags=re.IGNORECASE).strip()

    def _stream_model_answer(self, model: str, prompt: str, progress_callback=None, cancel_event=None) -> str:
        stream_generator = getattr(self.model_client, "generate_stream", None)
        if not stream_generator:
            return self._remove_inline_source_tags(self.model_client.generate(model, prompt))
        answer_parts: list[str] = []
        pending = [""]

        def reset_for_fallback():
            answer_parts.clear()
            pending[0] = ""
            if progress_callback:
                progress_callback({"type": "reset_draft"})

        try:
            stream = stream_generator(model, prompt, on_fallback=reset_for_fallback)
        except TypeError:  # Support simple model clients with the older two-argument stream method.
            stream = stream_generator(model, prompt)
        try:
            for piece in stream:
                if cancel_event and cancel_event.is_set():
                    raise QueryCancelled()
                answer_parts.append(piece)
                pending[0] += piece
                partial_tag = re.search(r"\[(?:S\d*)?$", pending[0], flags=re.IGNORECASE)
                safe_end = partial_tag.start() if partial_tag else len(pending[0])
                pending[0] = pending[0][safe_end:]
        finally:
            if cancel_event and cancel_event.is_set() and hasattr(stream, "close"):
                stream.close()
        return self._remove_inline_source_tags("".join(answer_parts))

    def _save_exchange(self, session_id: str | None, question: str, answer: str,
                       metadata: dict[str, Any] | None = None,
                       question_embedding: list[float] | None = None) -> None:
        if not self.app_store or not session_id:
            return
        if question_embedding is None:
            try:
                vectors = self.ingestion.embedder.encode([question])
                question_embedding = vectors[0] if len(vectors) == 1 else None
            except Exception:
                log.exception("Could not embed question for conversation memory persistence")
        try:
            self.app_store.save_exchange(session_id, question, answer, metadata,
                                         question_embedding=question_embedding)
        except TypeError:
            # Keep lightweight third-party/test stores compatible during migration.
            self.app_store.save_exchange(session_id, question, answer, metadata)

    @staticmethod
    def _feedback_corpus_fingerprint(documents: list[dict[str, Any]]) -> str:
        snapshot = sorted((doc["document_id"], doc.get("version"), doc.get("content_hash")) for doc in documents)
        return hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = sum(float(value) ** 2 for value in left) ** 0.5
        right_norm = sum(float(value) ** 2 for value in right) ** 0.5
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def canonical_feedback_question(self, question: str, rating: str, document_ids: list[str] | None,
                                    embedding: list[float]) -> str:
        return self.find_canonical_feedback_question(
            self.app_store, self.config, question, rating, document_ids, embedding)

    @staticmethod
    def find_canonical_feedback_question(app_store, config: RetrievalConfig, question: str,
                                         rating: str, document_ids: list[str] | None,
                                         embedding: list[float]) -> str:
        if not app_store:
            return question
        scope = sorted(document_ids) if document_ids is not None else []
        # Canonical question identity is rating-independent. Shortcut decisions
        # still use their stricter positive/negative thresholds at query time.
        threshold = config.feedback_positive_similarity_threshold
        candidates = []
        for entry in app_store.feedback_entries():
            if entry.get("scope", []) != scope:
                continue
            stored = entry.get("question_embedding")
            if stored is None:
                continue
            similarity = RAGService._cosine(embedding, stored)
            if similarity >= threshold:
                candidates.append((similarity, entry["question"]))
        return max(candidates, default=(0.0, question), key=lambda item: item[0])[1]

    def _feedback_shortcut(self, question: str, session_id: str | None,
                           document_ids: list[str] | None, *, progress_callback=None,
                           cancel_event=None) -> dict[str, Any] | None:
        if progress_callback:
            progress_callback({"type": "stage", "stage": "memory_check",
                               "detail": "Checking prior questions and feedback for a semantic match?"})
        if not self.app_store:
            if progress_callback:
                progress_callback({"type": "stage", "stage": "memory_none",
                                   "detail": "No conversation memory is available; continuing to document search."})
            return None
        entries = self.app_store.feedback_entries()
        try:
            vectors = self.ingestion.embedder.encode([question])
            if len(vectors) != 1:
                return None
            query_embedding = vectors[0]
        except Exception:
            log.exception("Could not embed question for feedback-memory matching")
            return None
        if cancel_event and cancel_event.is_set():
            raise QueryCancelled()

        selected = set(document_ids) if document_ids is not None else None
        documents = self.ingestion.metadata.list_documents()
        scoped_documents = [doc for doc in documents if selected is None or doc["document_id"] in selected]
        corpus_fingerprint = self._feedback_corpus_fingerprint(scoped_documents)
        scope = sorted(document_ids) if document_ids is not None else []
        matches = []
        shortcut_matches = []
        for entry in entries:
            if entry.get("scope", []) != scope or entry.get("question_embedding") is None:
                continue
            similarity = self._cosine(query_embedding, entry["question_embedding"])
            threshold = (self.config.feedback_positive_similarity_threshold if entry.get("rating") == "up"
                         else self.config.feedback_negative_similarity_threshold)
            # Question display/deduplication uses the canonical threshold even
            # when this rating is not yet eligible for a shortcut.
            if similarity >= self.config.feedback_positive_similarity_threshold:
                matches.append((entry, similarity))
            if similarity >= threshold:
                shortcut_matches.append((entry, similarity))

        history_matches = []
        try:
            records_method = getattr(self.app_store, "recent_question_records", None)
            if records_method:
                records = records_method(limit=100)
                missing = [record for record in records if not record.get("question_embedding")]
                if missing:
                    vectors = self.ingestion.embedder.encode([record["question"] for record in missing])
                    if len(vectors) == len(missing):
                        for record, embedding in zip(missing, vectors):
                            record["question_embedding"] = embedding
                            store_embedding = getattr(self.app_store, "store_question_embedding", None)
                            if store_embedding:
                                store_embedding(record["id"], embedding)
                history_matches = []
                for record in records:
                    record_embedding = record.get("question_embedding")
                    if not record_embedding:
                        continue
                    resolved = " ".join((record.get("feedback_question") or "").casefold().split())
                    related = [entry for entry in entries
                               if entry.get("scope", []) == scope and entry.get("question_embedding")
                               and (" ".join(entry["question"].casefold().split()) == resolved
                                    or self._cosine(record_embedding, entry["question_embedding"])
                                    >= self.config.feedback_positive_similarity_threshold)]
                    ups = sum(entry.get("rating") == "up" for entry in related)
                    downs = sum(entry.get("rating") == "down" for entry in related)
                    history_matches.append({
                        "question": record["question"],
                        "similarity": self._cosine(query_embedding, record_embedding),
                        "rating": ("mixed" if ups and downs else "up" if ups else "down" if downs else None),
                        "feedback_score": ups - downs,
                        "response_count": len({entry.get("answer_hash") for entry in related if entry.get("answer_hash")}),
                        "_feedback_ids": [entry["feedback_id"] for entry in related],
                        "_embedding": record_embedding,
                    })
                history_matches = [item for item in history_matches
                                   if item["similarity"] >= self.config.feedback_positive_similarity_threshold]
                for item in history_matches:
                    related_ids = set(item.get("_feedback_ids", []))
                    for entry in entries:
                        if entry.get("feedback_id") not in related_ids:
                            continue
                        pair = (entry, item["similarity"])
                        if all(existing[0].get("feedback_id") != entry["feedback_id"] for existing in matches):
                            matches.append(pair)
                        threshold = (self.config.feedback_positive_similarity_threshold if entry.get("rating") == "up"
                                     else self.config.feedback_negative_similarity_threshold)
                        if (entry.get("rating") == "up" and item["similarity"] >= threshold and all(
                                existing[0].get("feedback_id") != entry["feedback_id"]
                                for existing in shortcut_matches)):
                            shortcut_matches.append(pair)
        except Exception:
            log.exception("Could not compare prior conversation questions for memory display")

        if cancel_event and cancel_event.is_set():
            raise QueryCancelled()
        visible = {}
        for entry, similarity in matches:
            key = " ".join(entry["question"].casefold().split())
            item = visible.setdefault(key, {"question": entry["question"], "similarity": similarity,
                                             "upvotes": 0, "downvotes": 0, "answers": set(),
                                             "_embedding": entry.get("question_embedding")})
            item["similarity"] = max(item["similarity"], similarity)
            item["upvotes"] += int(entry.get("rating") == "up")
            item["downvotes"] += int(entry.get("rating") == "down")
            if entry.get("answer_hash"):
                item["answers"].add(entry["answer_hash"])
        for key, item in visible.items():
            item["rating"] = ("mixed" if item["upvotes"] and item["downvotes"] else
                               "up" if item["upvotes"] else "down" if item["downvotes"] else None)
            item["feedback_score"] = item["upvotes"] - item["downvotes"]
            item["response_count"] = len(item["answers"])
            del item["answers"]
        for item in history_matches:
            key = " ".join(item["question"].casefold().split())
            previous = visible.get(key)
            feedback_group = max(
                (candidate for candidate in visible.values()
                 if candidate.get("rating") and self._cosine(item.get("_embedding") or [],
                                                               candidate.get("_embedding") or [])
                 >= self.config.feedback_positive_similarity_threshold),
                key=lambda candidate: candidate["similarity"], default=None,
            )
            if previous is None or item["similarity"] > previous["similarity"]:
                visible[key] = {**item, "rating": (previous or {}).get("rating") or item.get("rating") or
                                (feedback_group or {}).get("rating"),
                                "feedback_score": (previous or {}).get("feedback_score",
                                                   item.get("feedback_score", (feedback_group or {}).get("feedback_score"))),
                                "response_count": (previous or {}).get("response_count",
                                                      item.get("response_count", (feedback_group or {}).get("response_count", 0)))}
        nearest = sorted(visible.values(), key=lambda item: item["similarity"], reverse=True)[:3]
        if progress_callback:
            if nearest:
                closest = nearest[0]
                detail = (f"Found {len(visible)} similar previous question(s). Closest: {closest['question']} "
                          f"({closest['similarity']:.0%} similar); "
                          f"{'reviewing grouped feedback and source freshness.' if matches else 'continuing to document search.'}")
                progress_callback({"type": "stage", "stage": "memory_match", "detail": detail,
                                   "matches": [{key: value for key, value in item.items() if not key.startswith("_")}
                                               | {"similarity": round(item["similarity"], 3)} for item in nearest]})
            elif not matches:
                progress_callback({"type": "stage", "stage": "memory_none",
                                   "detail": "No similar previous questions found; continuing to document search."})
        if not shortcut_matches:
            return None

        shortcut_feedback_ids = {entry["feedback_id"] for entry, _ in shortcut_matches}
        question_groups: list[dict[str, Any]] = []
        for entry, _ in sorted(matches, key=lambda item: item[1], reverse=True):
            normalized = " ".join(entry["question"].casefold().split())
            entry_embedding = entry.get("question_embedding") or []
            threshold = self.config.feedback_positive_similarity_threshold
            question_group = next((group for group in question_groups
                                   if group["scope"] == entry.get("scope", [])
                                   and (group["normalized"] == normalized
                                        or self._cosine(entry_embedding, group["question_embedding"]) >= threshold)), None)
            if question_group is None:
                question_group = {"question": entry["question"], "normalized": normalized,
                                  "question_embedding": entry_embedding, "scope": entry.get("scope", []),
                                  "entries": [], "shortcut_entries": [], "responses": {}}
                question_groups.append(question_group)
            question_group["entries"].append(entry)
            if any(candidate["feedback_id"] == entry["feedback_id"] for candidate, _ in shortcut_matches):
                question_group["shortcut_entries"].append(entry)
            source_signature = json.dumps(sorted(entry.get("sources") or [],
                                                  key=lambda source: (source.get("document_id", ""), source.get("chunk_id", ""))),
                                          sort_keys=True, separators=(",", ":"))
            response_key = f"{entry.get('answer_hash') or ''}:{source_signature}"
            answer_group = question_group["responses"].setdefault(response_key, {
                "answer": entry.get("answer", ""), "sources": entry.get("sources") or [],
                "feedback_score": 0, "entries": [],
            })
            answer_group["feedback_score"] += 1 if entry.get("rating") == "up" else -1
            answer_group["entries"].append(entry)

        chunks = self.ingestion.metadata.get_chunks()
        chunks_by_document = {(chunk["document_id"], chunk["chunk_id"]): chunk for chunk in chunks}
        docs_by_id = {doc["document_id"]: doc for doc in documents}
        for group in question_groups:
            current_entries = [entry for entry in group["entries"]
                               if entry.get("corpus_fingerprint") == corpus_fingerprint
                               and entry["feedback_id"] in shortcut_feedback_ids]
            eligible_entries = [entry for entry in group["shortcut_entries"]
                                if entry.get("corpus_fingerprint") == corpus_fingerprint]
            if not eligible_entries:
                continue
            feedback_score = sum(1 if entry.get("rating") == "up" else -1 for entry in current_entries)
            confidence_values = [entry["confidence"]["overall"] for entry in current_entries
                                 if entry.get("rating") == "down" and isinstance(entry.get("confidence"), dict)
                                 and isinstance(entry["confidence"].get("overall"), (int, float))]
            mean_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
            if (any(entry.get("rating") == "down" for entry in eligible_entries)
                    and feedback_score < -1 and mean_confidence is not None
                    and mean_confidence < self.config.minimum_overall_score):
                response = {
                    "answer": INSUFFICIENT,
                    "confidence": {"overall": None, "faithfulness": None, "relevance": None,
                                   "correctness": None, "status": "insufficient_evidence",
                                   "reason": "Similar answers repeatedly had low confidence and negative feedback."},
                    "sources": [], "session_id": session_id, "status": "insufficient_evidence",
                    "retry_count": 0, "generation_retry_count": 0, "fallback_used": False,
                    "error": None, "feedback_shortcut": "low_confidence_negative_feedback",
                    "feedback_question": group["question"], "feedback_score": feedback_score,
                    "feedback_responses": [{"answer": item["answer"], "sources": item["sources"],
                                            "feedback_score": item["feedback_score"]}
                                           for item in group["responses"].values()],
                }
                self._save_exchange(session_id, question, response["answer"], response, query_embedding)
                log.info("feedback_shortcut_applied kind=low_confidence_negative_feedback score=%s mean_confidence=%.1f",
                         feedback_score, mean_confidence)
                return response

            if not any(entry.get("rating") == "up" for entry in eligible_entries):
                continue

            response_context = []
            current_docs: dict[tuple[str, str], dict[str, Any]] = {}
            for answer_group in group["responses"].values():
                eligible_response_entries = [entry for entry in answer_group["entries"]
                                             if entry["feedback_id"] in shortcut_feedback_ids
                                             and entry.get("corpus_fingerprint") == corpus_fingerprint]
                answer_feedback_score = sum(1 if entry.get("rating") == "up" else -1
                                            for entry in eligible_response_entries)
                if answer_feedback_score < 1 or not answer_group["answer"]:
                    continue
                valid_entries = [entry for entry in eligible_response_entries
                                 if entry.get("rating") == "up"
                                 and entry.get("response_status") == "passed"
                                 and isinstance((entry.get("confidence") or {}).get("overall"), (int, float))
                                 and (entry.get("confidence") or {})["overall"] >= self.config.minimum_overall_score]
                if not valid_entries:
                    continue
                entry = max(valid_entries, key=lambda item: item.get("created_at", ""))
                cited_sources = entry.get("sources") or []
                if not cited_sources or not all(
                    source.get("document_id") in docs_by_id
                    and docs_by_id[source["document_id"]].get("version") == source.get("document_version")
                    and (source.get("document_id"), source.get("chunk_id")) in chunks_by_document
                    and chunks_by_document[(source["document_id"], source["chunk_id"])].get("content_hash") == source.get("chunk_hash")
                    for source in cited_sources
                ):
                    continue
                response_context.append({"answer": answer_group["answer"],
                                         "positive_feedback_score": answer_feedback_score})
                for source in cited_sources:
                    chunk = chunks_by_document[(source["document_id"], source["chunk_id"])]
                    doc = docs_by_id[source["document_id"]]
                    key = (source["document_id"], source["chunk_id"])
                    current_docs[key] = {
                        "document_id": source["document_id"], "document": doc["filename"],
                        "source_url": doc.get("source_url"), "document_version": doc.get("version"),
                        "section": source.get("section") or chunk.get("section", {}).get("title", "Document"),
                        "section_path": chunk.get("section", {}).get("path", []),
                        "chunk_id": source["chunk_id"], "chunk_hash": source.get("chunk_hash"),
                        "vector_id": chunk.get("vector_id"), "text": chunk["text"], "similarity": 1.0,
                    }

            if not response_context or not current_docs:
                continue
            if progress_callback:
                progress_callback({"type": "stage", "stage": "feedback_review",
                                   "detail": f"Checking {len(response_context)} positively rated answer variant(s) against their current sources."})
            try:
                memory_prompt = FEEDBACK_ANSWER_PROMPT.format(
                    question=question, canonical_question=group["question"],
                    answers=json.dumps(response_context[:10], ensure_ascii=False),
                    documents=json.dumps([{"document": doc["document"], "section": doc["section"],
                                           "chunk_id": doc["chunk_id"], "text": doc["text"]}
                                          for doc in list(current_docs.values())[:10]], ensure_ascii=False),
                )
                decision = self._generate_structured(self.config.generation_model,
                                                     self.config.generation_fallback_models,
                                                     memory_prompt, FeedbackAnswerOutput)
            except ModelError as exc:
                log.warning("Positive feedback answer could not be reviewed; continuing to retrieval: %s", exc)
                decision = None
            if not decision or not decision.can_answer or not decision.answer.strip():
                if progress_callback:
                    progress_callback({"type": "stage", "stage": "feedback_rejected",
                                       "detail": "The positively rated answers do not safely resolve this wording; continuing to document retrieval."})
                continue

            cited_docs = list(current_docs.values())[:10]
            evaluation_state = {
                "original_query": question, "standalone_query": question, "answer": decision.answer.strip(),
                "relevant_documents": cited_docs, "generation_retry_count": 1,
                "progress_callback": progress_callback, "request_id": None,
            }
            evaluation = self._answer_evaluation(evaluation_state)
            if evaluation.get("error") or not evaluation.get("evaluation_scores", {}).get("passed"):
                if progress_callback:
                    progress_callback({"type": "stage", "stage": "feedback_rejected",
                                       "detail": "The answer from feedback memory did not pass source verification; continuing to document retrieval."})
                continue

            confidence_data = evaluation["evaluation_scores"]
            confidence = {"overall": evaluation["confidence_score"],
                          "faithfulness": confidence_data["faithfulness"],
                          "relevance": confidence_data["relevance"],
                          "correctness": confidence_data["correctness"],
                          "status": "passed", "reasons": confidence_data["reasons"],
                          "unsupported_claims": confidence_data["unsupported_claims"]}
            response = {
                "answer": self._remove_inline_source_tags(decision.answer.strip()),
                "confidence": confidence, "sources": self._source_list(cited_docs),
                "session_id": session_id, "status": "passed", "retry_count": 0,
                "generation_retry_count": 0, "fallback_used": False, "error": None,
                "feedback_shortcut": "upvoted_answer", "feedback_score": feedback_score,
                "feedback_question": group["question"],
                "feedback_responses": [{"answer": item["answer"], "sources": item["sources"],
                                         "feedback_score": item["feedback_score"]}
                                        for item in group["responses"].values()],
            }
            if progress_callback:
                progress_callback({"type": "stage", "stage": "feedback_approved",
                                   "detail": "The reframed answer passed evaluation against its current cited documentation."})
            self._save_exchange(session_id, question, response["answer"], response, query_embedding)
            log.info("feedback_shortcut_applied kind=upvoted_answer_reframed feedback_score=%s confidence=%s",
                     feedback_score, confidence.get("overall"))
            return response
        return None

    def _generation(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        docs = state.get("relevant_documents", [])
        self._progress(state, "generation", "Writing a draft grounded in the relevant sources…")
        log.info("graph_node_started request_id=%s node=generation relevant=%s regeneration=%s", state.get("request_id"), len(docs), state.get("generation_retry_count", 0))
        sources = self._source_list(docs)
        excerpt = [{"document": d["document"], "section": d["section"],
                    "chunk_id": d["chunk_id"], "text": d["text"]} for d in docs]
        correction = ""
        if state.get("generation_retry_count", 0):
            correction = "Correct the prior draft according to this evaluator feedback. Do not preserve unsupported claims: " + state.get("correction_feedback", "")
            callback = state.get("progress_callback")
            if callback:
                callback({"type": "reset_draft"})
        try:
            prompt = GENERATION_PROMPT.format(
                correction=correction, feedback_context="", question=state["standalone_query"],
                documents=json.dumps(excerpt, ensure_ascii=False),
            )
            if state.get("stream_output"):
                answer = self._stream_model_answer(self.config.generation_model, prompt,
                                                   state.get("progress_callback"), state.get("cancel_event"))
            else:
                answer = self._remove_inline_source_tags(self.model_client.generate(self.config.generation_model, prompt))
            if not answer:
                raise ModelError("Generation response was empty")
            log.info("graph_node_completed request_id=%s node=generation sources=%s regeneration=%s duration_ms=%.1f",
                     state.get("request_id"), len(sources), state.get("generation_retry_count", 0), (perf_counter() - started) * 1000)
            return {"answer": answer, "sources": sources, "error": None}
        except ModelError as exc:
            log.error("Answer generation failed after %.1f ms: %s", (perf_counter() - started) * 1000, exc)
            return {"error": "generation_unavailable", "answer": "", "sources": sources}

    def _answer_evaluation(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        self._progress(state, "evaluation", "Verifying the draft against the documentation before showing it…")
        docs = [{"document": d["document"], "section": d["section"], "chunk_id": d["chunk_id"], "text": d["text"]}
                for d in state.get("relevant_documents", [])]
        try:
            prompt = EVALUATION_PROMPT.format(
                question=state["original_query"],
                standalone_query=state.get("standalone_query") or state["original_query"],
                answer=state.get("answer", ""), documents=json.dumps(docs, ensure_ascii=False),
            )
            data = self._generate_structured(self.config.judge_model,
                                             self.config.judge_fallback_models,
                                             prompt, AnswerEvaluationOutput)
            scores, reasons = {}, {}
            for dimension in ("faithfulness", "relevance", "correctness"):
                item = getattr(data, dimension)
                scores[dimension] = item.score
                reasons[dimension] = item.reason
            unsupported = data.unsupported_claims
            overall = round(.45 * scores["faithfulness"] + .30 * scores["relevance"] + .25 * scores["correctness"])
            passed = not unsupported and overall >= self.config.minimum_overall_score and all(
                scores[key] >= self.config.minimum_dimension_score for key in scores
            )
            evaluation = {**scores, "overall": overall, "reasons": reasons,
                          "unsupported_claims": unsupported,
                          "feedback": data.feedback, "passed": passed,
                          "status": "passed" if passed else "failed"}
            log.info("graph_node_completed request_id=%s node=answer_evaluation status=%s overall=%s faithfulness=%s relevance=%s correctness=%s duration_ms=%.1f",
                     state.get("request_id"), evaluation["status"], overall, scores["faithfulness"], scores["relevance"], scores["correctness"],
                     (perf_counter() - started) * 1000)
            regenerate = not passed and state.get("generation_retry_count", 0) < 1
            correction_feedback = data.feedback or "Improve grounding and answer relevance."
            if unsupported:
                correction_feedback += " Remove or rewrite these unsupported claims: " + json.dumps(unsupported, ensure_ascii=False)
            if regenerate:
                self._progress(state, "answer_retry", "The draft did not meet the evidence checks. Revising it against evaluator feedback…")
            elif not passed:
                self._progress(state, "unverified", "The draft did not pass verification. Showing a safe response instead.")
            return {"evaluation_scores": evaluation, "confidence_score": overall, "error": None,
                    "should_regenerate": regenerate,
                    "generation_retry_count": state.get("generation_retry_count", 0) + int(regenerate),
                    "correction_feedback": correction_feedback}
        except ModelError as exc:
            log.error("Answer evaluation failed after %.1f ms: %s", (perf_counter() - started) * 1000, exc)
            return {"evaluation_scores": {"status": "unverified", "reason": str(exc)},
                    "confidence_score": None, "error": "judge_unavailable"}

    def _after_evaluation(self, state: RAGState) -> str:
        if state.get("error"):
            return "safe"
        if state.get("evaluation_scores", {}).get("passed"):
            return "passed"
        if state.get("should_regenerate"):
            return "regenerate"
        return "safe"

    def _insufficient_evidence(self, state: RAGState) -> dict[str, Any]:
        self._progress(state, "insufficient_evidence", "No sufficiently relevant documentation was found.")
        return {"answer": INSUFFICIENT, "sources": [], "evaluation_scores": {"status": "insufficient_evidence"},
                "confidence_score": None, "response_status": "insufficient_evidence"}

    def _safe_response(self, state: RAGState) -> dict[str, Any]:
        self._progress(state, "unverified", "The answer could not be verified. Showing a safe response instead.")
        return {"answer": UNVERIFIED, "sources": [],
                "memory_sources": state.get("sources") or self._source_list(state.get("relevant_documents", [])),
                "evaluation_scores": state.get("evaluation_scores") or {"status": "unverified"},
                "confidence_score": state.get("confidence_score"), "response_status": "unverified"}

    @staticmethod
    def _final_response(state: RAGState) -> dict[str, Any]:
        callback = state.get("progress_callback")
        if callback:
            callback({"type": "stage", "stage": "approved", "detail": "The streamed draft passed verification."})
        return {"response_status": "passed"}

    def _initial_state(self, question: str, session_id: str | None, *, document_ids: list[str] | None = None,
                       top_k: int | None = None, chat_history: list[dict[str, str]] | None = None,
                       request_id: str | None = None, stream_output: bool = False, progress_callback=None,
                       cancel_event=None) -> RAGState:
        history = chat_history or (self.app_store.history(session_id, self.config.max_history_turns) if self.app_store else [])
        return {
            "original_query": question, "standalone_query": question, "rewritten_query": question,
            "session_id": session_id, "request_id": request_id, "chat_history": history, "document_ids": document_ids,
            "top_k": top_k or self.config.top_k, "retrieved_documents": [], "graded_documents": [],
            "relevant_documents": [], "answer": "", "sources": [], "evaluation_scores": {},
            "confidence_score": None, "retry_count": 0, "generation_retry_count": 0,
            "fallback_used": False, "error": None, "response_status": "pending", "correction_feedback": "",
            "stream_output": stream_output, "progress_callback": progress_callback,
            "cancel_event": cancel_event, "search_scope_empty": False,
        }

    def _format_response(self, result: dict[str, Any], session_id: str | None) -> dict[str, Any]:
        evaluation = result.get("evaluation_scores", {})
        confidence = {
            "overall": result.get("confidence_score"),
            "faithfulness": evaluation.get("faithfulness"), "relevance": evaluation.get("relevance"),
            "correctness": evaluation.get("correctness"), "status": evaluation.get("status", result.get("response_status")),
        }
        for key in ("reasons", "unsupported_claims", "reason"):
            if key in evaluation:
                confidence[key] = evaluation[key]
        response = {"answer": result.get("answer", ""), "confidence": confidence, "sources": result.get("sources", []),
                    "session_id": session_id, "status": result.get("response_status", "unverified"),
                    "retry_count": result.get("retry_count", 0), "generation_retry_count": result.get("generation_retry_count", 0),
                    "fallback_used": result.get("fallback_used", False), "error": result.get("error"),
                    "feedback_question": result.get("standalone_query") or result.get("original_query", "")}
        if self.app_store:
            metadata = {**response, "memory_sources": result.get("memory_sources") or result.get("sources", [])}
            self._save_exchange(session_id, result.get("original_query", ""), response["answer"], metadata)
        return response

    def query(self, question: str, session_id: str | None = None, *, document_ids: list[str] | None = None,
              top_k: int | None = None, chat_history: list[dict[str, str]] | None = None,
              request_id: str | None = None) -> dict[str, Any]:
        shortcut = self._feedback_shortcut(question, session_id, document_ids)
        if shortcut:
            return shortcut
        state = self._initial_state(question, session_id, document_ids=document_ids, top_k=top_k,
                                   chat_history=chat_history, request_id=request_id)
        result = self.graph.invoke(state)
        return self._format_response(result, session_id)

    def query_stream(self, question: str, session_id: str | None = None, *, document_ids: list[str] | None = None,
                     top_k: int | None = None, chat_history: list[dict[str, str]] | None = None,
                     request_id: str | None = None):
        """Yield progress events, then approved answer pieces and the final response."""
        events: Queue = Queue()
        finished = object()
        cancel_event = Event()

        def publish(event):
            if not cancel_event.is_set():
                events.put(event)

        def run_graph():
            try:
                shortcut = self._feedback_shortcut(question, session_id, document_ids,
                                                   progress_callback=publish,
                                                   cancel_event=cancel_event)
                if cancel_event.is_set():
                    return
                if shortcut:
                    publish({"type": "stage", "stage": "feedback_shortcut",
                             "detail": ("Using a positively rated answer reframed and verified against current citations; document search was skipped."
                                        if shortcut["status"] == "passed" else
                                        "Similar low-confidence answers received repeated negative feedback; retrieval was skipped.")})
                    if shortcut.get("status") == "passed":
                        for character in shortcut["answer"]:
                            publish({"type": "token", "text": character})
                    publish({"type": "result", "response": shortcut})
                    return
                state = self._initial_state(question, session_id, document_ids=document_ids, top_k=top_k,
                                            chat_history=chat_history, request_id=request_id,
                                            stream_output=True, progress_callback=publish,
                                            cancel_event=cancel_event)
                result = self.graph.invoke(state)
                if cancel_event.is_set():
                    return
                response = self._format_response(result, session_id)
                if response["status"] == "passed":
                    for character in response["answer"]:
                        publish({"type": "token", "text": character})
                publish({"type": "result", "response": response})
            except QueryCancelled:
                log.info("query_stream_cancelled request_id=%s", request_id)
            except Exception:
                log.exception("Streaming query failed request_id=%s", request_id)
                publish({"type": "error", "message": "The query could not be completed. Please try again."})
            finally:
                events.put(finished)

        Thread(target=run_graph, name=f"rag-stream-{request_id or 'query'}", daemon=True).start()
        try:
            yield {"type": "stage", "stage": "started", "detail": "Starting the documentation assistant…"}
            while True:
                event = events.get()
                if event is finished:
                    break
                yield event
        finally:
            cancel_event.set()
