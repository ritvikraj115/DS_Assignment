from __future__ import annotations

import json
import hashlib
import logging
import re
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from ingestion.service import IngestionService

from .config import RetrievalConfig
from .llm import GeminiClient, ModelClient, ModelError, parse_json_response
from .models import RAGState
from .prompts import EVALUATION_PROMPT, GENERATION_PROMPT, GRADE_PROMPT, QUERY_REWRITE_PROMPT

log = logging.getLogger(__name__)
INSUFFICIENT = "I couldn't find enough relevant documentation to answer that question reliably."
UNVERIFIED = "I couldn't verify an answer against the documentation right now, so I won't present an unverified response."


class RAGService:
    def __init__(self, ingestion: IngestionService, config: RetrievalConfig | None = None,
                 model_client: ModelClient | None = None, app_store=None):
        self.ingestion = ingestion
        self.config = config or RetrievalConfig.from_env()
        self.config.validate()
        self.model_client = model_client or GeminiClient(
            self.config.gemini_api_key, self.config.gemini_timeout_ms,
            model_fallbacks={
                self.config.generation_model: self.config.generation_fallback_models,
                self.config.judge_model: self.config.judge_fallback_models,
            },
        )
        self.app_store = app_store
        self.graph = self._build_graph()

    @staticmethod
    def _progress(state: RAGState, stage: str, detail: str) -> None:
        callback = state.get("progress_callback")
        if callback:
            callback({"type": "stage", "stage": stage, "detail": detail})

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
        try:
            raw = self.model_client.generate(self.config.generation_model, QUERY_REWRITE_PROMPT.format(
                original_query=original, standalone_query=state.get("standalone_query", original),
                retry_count=state.get("retry_count", 0), chat_history=json.dumps(state.get("chat_history", []), ensure_ascii=False),
            ), json_mode=True)
            data = parse_json_response(raw)
            standalone = str(data.get("standalone_query", "")).strip()
            rewritten = str(data.get("rewritten_query", standalone)).strip()
            if not standalone or not rewritten:
                raise ModelError("Query rewrite response is missing required fields")
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
        if state.get("error"):
            return {"retrieved_documents": []}
        query = state.get("rewritten_query") or state["standalone_query"]
        try:
            vectors = self.ingestion.embedder.encode([query])
        except Exception as exc:
            log.exception("Query embedding failed")
            return {"error": "embedding_failed", "retrieved_documents": []}
        if len(vectors) != 1:
            return {"error": "embedding_failed", "retrieved_documents": []}
        all_chunks = self.ingestion.metadata.get_chunks()
        by_vector = {int(c["vector_id"]): c for c in all_chunks}
        doc_info = {d["document_id"]: d for d in self.ingestion.metadata.list_documents()}
        document_ids = state.get("document_ids")
        try:
            if document_ids is not None:
                allowed = set(document_ids)
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
        return {"retrieved_documents": result}

    def _document_grading(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        docs = state.get("retrieved_documents", [])
        self._progress(state, "source_review", f"Checking {len(docs)} retrieved source{'s' if len(docs) != 1 else ''} for relevance…")
        log.info("graph_node_started request_id=%s node=document_grading retrieved=%s", state.get("request_id"), len(docs))
        if not docs:
            retry = state.get("retry_count", 0) < self.config.max_retrieval_retries
            log.info("graph_node_completed request_id=%s node=document_grading relevant=0 retry=%s duration_ms=%.1f",
                     state.get("request_id"), retry, (perf_counter() - started) * 1000)
            return {"graded_documents": [], "relevant_documents": [], "should_retry": retry,
                    "retry_count": state.get("retry_count", 0) + int(retry),
                    "fallback_used": state.get("fallback_used", False) or retry}
        compact = [{k: d[k] for k in ("chunk_id", "document", "section", "text")} for d in docs]
        try:
            raw = self.model_client.generate(self.config.judge_model, GRADE_PROMPT.format(
                question=state["standalone_query"], documents=json.dumps(compact, ensure_ascii=False),
            ), json_mode=True)
            data = parse_json_response(raw)
            grades = data.get("grades")
            if not isinstance(grades, list):
                raise ModelError("Document grading response has no grades list")
            by_id = {str(g.get("chunk_id")): g for g in grades if isinstance(g, dict)}
            graded = []
            for doc in docs:
                grade = by_id.get(doc["chunk_id"])
                if grade is None or not isinstance(grade.get("relevant"), bool):
                    raise ModelError("Document grading response omitted a retrieved chunk")
                graded.append({**doc, "relevant": grade["relevant"], "reason": str(grade.get("reason", ""))})
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

    def _stream_model_answer(self, model: str, prompt: str, progress_callback=None) -> str:
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
        for piece in stream:
            answer_parts.append(piece)
            pending[0] += piece
            partial_tag = re.search(r"\[(?:S\d*)?$", pending[0], flags=re.IGNORECASE)
            safe_end = partial_tag.start() if partial_tag else len(pending[0])
            visible = self._remove_inline_source_tags(pending[0][:safe_end])
            pending[0] = pending[0][safe_end:]
            if visible and progress_callback:
                progress_callback({"type": "draft_token", "text": visible})
        final_piece = self._remove_inline_source_tags(pending[0])
        if final_piece and progress_callback:
            progress_callback({"type": "draft_token", "text": final_piece})
        return self._remove_inline_source_tags("".join(answer_parts))

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

    def _synthesize_upvoted_responses(self, question: str, groups: list[dict[str, Any]],
                                      docs_by_id: dict[str, dict[str, Any]],
                                      chunks_by_document: dict[tuple[str, str], dict[str, Any]],
                                      session_id: str | None, *, stream_output: bool = False,
                                      progress_callback=None) -> dict[str, Any]:
        document_lookup = {doc["document_id"]: doc for doc in self.ingestion.metadata.list_documents()}
        selected_chunks: dict[tuple[str, str], dict[str, Any]] = {}
        for group in groups:
            for source in group["entry"].get("sources", []):
                key = (source["document_id"], source["chunk_id"])
                chunk = chunks_by_document[key]
                doc = docs_by_id[source["document_id"]]
                document = document_lookup.get(source["document_id"], {})
                selected_chunks.setdefault(key, {
                    "document_id": source["document_id"], "document": document.get("filename", source["document_id"]),
                    "source_url": document.get("source_url"), "document_version": document.get("version"),
                    "section": chunk.get("section", {}).get("title", "Document"),
                    "section_path": chunk.get("section", {}).get("path", []), "chunk_id": chunk["chunk_id"],
                    "chunk_hash": chunk.get("content_hash"), "vector_id": chunk.get("vector_id"),
                    "text": chunk["text"], "similarity": 1.0,
                })
        docs = list(selected_chunks.values())
        sources = self._source_list(docs)
        evidence = [{"document": doc["document"], "section": doc["section"],
                     "chunk_id": doc["chunk_id"], "text": doc["text"]} for doc in docs]
        prior_answers = [{"answer": group["answer"], "upvotes": group["count"],
                          "sources": [{"document": src["document"], "section": src["section"], "chunk_id": src["chunk_id"]}
                                      for src in group["entry"].get("sources", [])]}
                         for group in groups]
        feedback_context = (
            "The following distinct responses were previously upvoted for this question. Synthesize their useful points, "
            "but treat votes as preference signals rather than proof. Resolve conflicts using only the current evidence below. "
            "Do not add source markers such as [S1]. Upvoted responses: "
            + json.dumps(prior_answers, ensure_ascii=False)
        )
        if progress_callback:
            progress_callback({"type": "stage", "stage": "feedback_synthesis",
                               "detail": f"Synthesizing {len(groups)} upvoted responses from their current source chunks; skipping search."})
        correction = "Use only the current evidence. Incorporate the useful information from the previously upvoted responses, resolve any differences against the evidence, and do not repeat unsupported claims."
        answer = ""
        evaluation_result: dict[str, Any] = {}
        for attempt in range(2):
            if attempt:
                correction += " Revise the prior synthesis using evaluator feedback: " + str(evaluation_result.get("correction_feedback", ""))
                if progress_callback:
                    progress_callback({"type": "reset_draft"})
                    progress_callback({"type": "stage", "stage": "answer_retry",
                                       "detail": "Revising the synthesis against evaluator feedback."})
            if progress_callback:
                progress_callback({"type": "stage", "stage": "generation",
                                   "detail": "Writing from the upvoted responses and their current documentation evidence."})
            prompt = GENERATION_PROMPT.format(
                correction=correction, feedback_context=feedback_context, question=question,
                documents=json.dumps(evidence, ensure_ascii=False),
            )
            try:
                if stream_output:
                    answer = self._stream_model_answer(self.config.generation_model, prompt, progress_callback)
                else:
                    answer = self._remove_inline_source_tags(self.model_client.generate(self.config.generation_model, prompt))
                if not answer:
                    raise ModelError("Feedback synthesis returned an empty answer")
            except ModelError as exc:
                log.error("Upvoted-answer synthesis failed: %s", exc)
                answer = UNVERIFIED
                break

            evaluation_result = self._answer_evaluation({
                "original_query": question, "answer": answer, "relevant_documents": docs,
                "generation_retry_count": attempt, "request_id": None, "correction_feedback": "",
                "progress_callback": progress_callback,
            })
            evaluation = evaluation_result.get("evaluation_scores", {})
            if evaluation.get("passed"):
                confidence = {key: evaluation.get(key) for key in ("overall", "faithfulness", "relevance", "correctness", "status", "reasons", "unsupported_claims") if key in evaluation}
                response = {"answer": answer, "confidence": confidence, "sources": sources,
                            "session_id": session_id, "status": "passed", "retry_count": 0,
                            "generation_retry_count": attempt, "fallback_used": False, "error": None,
                            "feedback_shortcut": "upvoted_synthesis", "feedback_votes": sum(group["count"] for group in groups)}
                if self.app_store:
                    self.app_store.save_exchange(session_id, question, answer, response)
                if progress_callback:
                    progress_callback({"type": "stage", "stage": "approved",
                                       "detail": "The synthesized answer passed verification against current cited chunks."})
                log.info("feedback_shortcut_applied kind=upvoted_synthesis responses=%s votes=%s",
                         len(groups), response["feedback_votes"])
                return response
            if evaluation_result.get("error") or not evaluation_result.get("should_regenerate"):
                answer = UNVERIFIED
                break

        response = {"answer": UNVERIFIED, "confidence": {"overall": None, "faithfulness": None,
                    "relevance": None, "correctness": None, "status": "unverified"},
                    "sources": [], "session_id": session_id, "status": "unverified", "retry_count": 0,
                    "generation_retry_count": 0, "fallback_used": False, "error": evaluation_result.get("error"),
                    "feedback_shortcut": "upvoted_synthesis_unverified"}
        if self.app_store:
            self.app_store.save_exchange(session_id, question, response["answer"], response)
        if progress_callback:
            progress_callback({"type": "stage", "stage": "unverified",
                               "detail": "The synthesized answer could not be verified. Showing a safe response."})
        return response

    def _feedback_shortcut(self, question: str, session_id: str | None,
                           document_ids: list[str] | None, *, stream_output: bool = False,
                           progress_callback=None) -> dict[str, Any] | None:
        if progress_callback:
            progress_callback({"type": "stage", "stage": "memory_check",
                               "detail": "Checking prior questions and feedback for a semantic match…"})
        if not self.app_store:
            if progress_callback:
                progress_callback({"type": "stage", "stage": "memory_none",
                                   "detail": "No conversation memory is available; continuing to document search."})
            return None
        entries = self.app_store.feedback_entries()
        try:
            vector = self.ingestion.embedder.encode([question])
        except Exception:
            log.exception("Could not embed question for feedback-memory matching")
            if progress_callback:
                progress_callback({"type": "stage", "stage": "memory_unavailable",
                                   "detail": "Could not compare feedback memory; continuing to document search."})
            return None
        if len(vector) != 1:
            if progress_callback:
                progress_callback({"type": "stage", "stage": "memory_unavailable",
                                   "detail": "Could not compare feedback memory; continuing to document search."})
            return None
        query_embedding = vector[0]
        selected = set(document_ids) if document_ids is not None else None
        documents = self.ingestion.metadata.list_documents()
        scoped_documents = [doc for doc in documents if selected is None or doc["document_id"] in selected]
        corpus_fingerprint = self._feedback_corpus_fingerprint(scoped_documents)
        scope = sorted(document_ids) if document_ids is not None else []
        matches = []
        for entry in entries:  # Store returns newest first, which is needed for consecutive downvotes.
            if entry.get("scope", []) != scope:
                continue
            stored_embedding = entry.get("question_embedding")
            if stored_embedding is None:
                continue
            similarity = self._cosine(query_embedding, stored_embedding)
            if similarity >= self.config.feedback_similarity_threshold:
                matches.append((entry, similarity))

        # Show prior questions even when they have no thumbs feedback. Only feedback
        # records above can affect answer reuse or downvote suppression.
        history_matches = []
        try:
            recent_questions = self.app_store.recent_questions(limit=100) if hasattr(self.app_store, "recent_questions") else []
            recent_questions = [text for text in recent_questions if text.strip()]
            if recent_questions:
                history_vectors = self.ingestion.embedder.encode(recent_questions)
                if len(history_vectors) == len(recent_questions):
                    history_matches = [
                        {"question": text, "similarity": self._cosine(query_embedding, stored), "rating": None}
                        for text, stored in zip(recent_questions, history_vectors)
                    ]
                    history_matches = [item for item in history_matches
                                       if item["similarity"] >= self.config.feedback_similarity_threshold]
        except Exception:
            log.exception("Could not compare prior conversation questions for memory display")

        visible_matches = {
            " ".join(entry["question"].casefold().split()):
                {"question": entry["question"], "similarity": similarity, "rating": entry.get("rating")}
            for entry, similarity in matches
        }
        for item in history_matches:
            key = " ".join(item["question"].casefold().split())
            current = visible_matches.get(key)
            if current is None or item["similarity"] > current["similarity"]:
                visible_matches[key] = {**item, "rating": item.get("rating") or (current or {}).get("rating")}
        nearest = sorted(visible_matches.values(), key=lambda item: item["similarity"], reverse=True)[:3]
        if not matches:
            if progress_callback:
                if nearest:
                    closest = nearest[0]
                    progress_callback({"type": "stage", "stage": "memory_match",
                                       "detail": (f"Found {len(visible_matches)} similar previous question(s). Closest: “{closest['question']}” "
                                                  f"({closest['similarity']:.0%} similar); continuing to document search."),
                                       "matches": [{**item, "similarity": round(item["similarity"], 3)} for item in nearest]})
                else:
                    progress_callback({"type": "stage", "stage": "memory_none",
                                       "detail": "No similar previous questions found; continuing to document search."})
            return None
        if progress_callback:
            closest = nearest[0]
            progress_callback({"type": "stage", "stage": "memory_match",
                               "detail": (f"Found {len(visible_matches)} similar previous question(s). Closest: “{closest['question']}” "
                                          f"({closest['similarity']:.0%} similar); checking feedback and source freshness."),
                               "matches": [{**item, "similarity": round(item["similarity"], 3)} for item in nearest]})

        # A changed corpus invalidates old negative-feedback suppression.
        consecutive_downvotes = 0
        for entry, similarity in matches:
            if similarity < self.config.feedback_direct_similarity_threshold:
                continue
            if entry.get("corpus_fingerprint") != corpus_fingerprint:
                continue
            if entry["rating"] == "down":
                consecutive_downvotes += 1
                if consecutive_downvotes == 3:
                    response = {
                        "answer": INSUFFICIENT,
                        "confidence": {"overall": None, "faithfulness": None, "relevance": None,
                                       "correctness": None, "status": "insufficient_evidence",
                                       "reason": "Recent feedback repeatedly rated answers to this question as unhelpful."},
                        "sources": [], "session_id": session_id, "status": "insufficient_evidence",
                        "retry_count": 0, "generation_retry_count": 0, "fallback_used": False,
                        "error": None, "feedback_shortcut": "downvote_suppression",
                    }
                    if self.app_store:
                        self.app_store.save_exchange(session_id, question, response["answer"], response)
                    log.info("feedback_shortcut_applied kind=downvote_suppression consecutive_votes=%s", consecutive_downvotes)
                    return response
            else:
                consecutive_downvotes = 0

        # Reuse an answer only if the exact cited document and chunk versions are still indexed.
        chunks = self.ingestion.metadata.get_chunks()
        chunks_by_document = {(chunk["document_id"], chunk["chunk_id"]): chunk for chunk in chunks}
        docs_by_id = {doc["document_id"]: doc for doc in documents}
        positive_groups: dict[tuple[str, str], dict[str, Any]] = {}
        for entry, similarity in matches:
            if entry.get("rating") != "up" or entry.get("response_status") != "passed" or not entry.get("answer"):
                continue
            sources = entry.get("sources") or []
            if not sources:
                continue
            if not all(
                source.get("document_id") in docs_by_id
                and docs_by_id[source["document_id"]].get("version") == source.get("document_version")
                and (source.get("document_id"), source.get("chunk_id")) in chunks_by_document
                and chunks_by_document[(source["document_id"], source["chunk_id"])].get("content_hash") == source.get("chunk_hash")
                for source in sources
            ):
                continue
            canonical_sources = sorted(
                ({key: value for key, value in source.items() if key != "label"} for source in sources),
                key=lambda source: (source.get("document_id", ""), source.get("chunk_id", "")),
            )
            source_signature = json.dumps(canonical_sources, sort_keys=True, separators=(",", ":"))
            clean_answer = self._remove_inline_source_tags(entry["answer"])
            clean_hash = hashlib.sha256(clean_answer.encode("utf-8")).hexdigest()
            key = (clean_hash, source_signature)
            group = positive_groups.setdefault(key, {"count": 0, "direct_count": 0,
                                                       "entry": entry, "answer": clean_answer})
            group["count"] += 1
            if similarity >= self.config.feedback_direct_similarity_threshold:
                group["direct_count"] += 1
        valid_groups = list(positive_groups.values())
        upvote_count = sum(group["count"] for group in valid_groups)
        if upvote_count < 3 or not valid_groups:
            return None
        if len(valid_groups) > 1 or valid_groups[0]["direct_count"] < 3:
            return self._synthesize_upvoted_responses(question, valid_groups, docs_by_id,
                chunks_by_document, session_id, stream_output=stream_output, progress_callback=progress_callback)
        best = valid_groups[0]
        if best["direct_count"] < 3:
            return None
        entry = best["entry"]
        confidence = entry.get("confidence") or {}
        confidence = {**confidence, "status": "passed"}
        sources = [{key: value for key, value in source.items() if key != "label"} for source in entry["sources"]]
        response = {"answer": best["answer"], "confidence": confidence, "sources": sources,
                    "session_id": session_id, "status": "passed", "retry_count": 0,
                    "generation_retry_count": 0, "fallback_used": False, "error": None,
                    "feedback_shortcut": "upvoted_answer", "feedback_votes": best["count"]}
        if self.app_store:
            self.app_store.save_exchange(session_id, question, response["answer"], response)
        log.info("feedback_shortcut_applied kind=upvoted_answer votes=%s", best["count"])
        return response

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
                answer = self._stream_model_answer(self.config.generation_model, prompt, state.get("progress_callback"))
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
            raw = self.model_client.generate(self.config.judge_model, EVALUATION_PROMPT.format(
                question=state["original_query"], answer=state.get("answer", ""), documents=json.dumps(docs, ensure_ascii=False),
            ), json_mode=True)
            data = parse_json_response(raw)
            scores, reasons = {}, {}
            for dimension in ("faithfulness", "relevance", "correctness"):
                item = data.get(dimension)
                if not isinstance(item, dict) or not isinstance(item.get("score"), int) or isinstance(item.get("score"), bool):
                    raise ModelError(f"Evaluation response has invalid {dimension} score")
                score = item["score"]
                if not 0 <= score <= 100:
                    raise ModelError(f"Evaluation {dimension} score must be between 0 and 100")
                scores[dimension] = score
                reasons[dimension] = str(item.get("reason", ""))
            unsupported = data.get("unsupported_claims", [])
            if not isinstance(unsupported, list) or not all(isinstance(item, str) for item in unsupported):
                raise ModelError("Evaluation unsupported_claims must be a list of strings")
            overall = round(.45 * scores["faithfulness"] + .30 * scores["relevance"] + .25 * scores["correctness"])
            passed = overall >= self.config.minimum_overall_score and all(
                scores[key] >= self.config.minimum_dimension_score for key in scores
            )
            evaluation = {**scores, "overall": overall, "reasons": reasons,
                          "unsupported_claims": unsupported,
                          "feedback": str(data.get("feedback", "")), "passed": passed,
                          "status": "passed" if passed else "failed"}
            log.info("graph_node_completed request_id=%s node=answer_evaluation status=%s overall=%s faithfulness=%s relevance=%s correctness=%s duration_ms=%.1f",
                     state.get("request_id"), evaluation["status"], overall, scores["faithfulness"], scores["relevance"], scores["correctness"],
                     (perf_counter() - started) * 1000)
            regenerate = not passed and state.get("generation_retry_count", 0) < 1
            if regenerate:
                self._progress(state, "answer_retry", "The draft did not meet the evidence checks. Revising it against evaluator feedback…")
            elif not passed:
                self._progress(state, "unverified", "The draft did not pass verification. Showing a safe response instead.")
            return {"evaluation_scores": evaluation, "confidence_score": overall, "error": None,
                    "should_regenerate": regenerate,
                    "generation_retry_count": state.get("generation_retry_count", 0) + int(regenerate),
                    "correction_feedback": str(data.get("feedback", "Improve grounding and answer relevance."))}
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
                "evaluation_scores": state.get("evaluation_scores") or {"status": "unverified"},
                "confidence_score": None, "response_status": "unverified"}

    @staticmethod
    def _final_response(state: RAGState) -> dict[str, Any]:
        callback = state.get("progress_callback")
        if callback:
            callback({"type": "stage", "stage": "approved", "detail": "The streamed draft passed verification."})
        return {"response_status": "passed"}

    def _initial_state(self, question: str, session_id: str | None, *, document_ids: list[str] | None = None,
                       top_k: int | None = None, chat_history: list[dict[str, str]] | None = None,
                       request_id: str | None = None, stream_output: bool = False, progress_callback=None) -> RAGState:
        history = chat_history or (self.app_store.history(session_id, self.config.max_history_turns) if self.app_store else [])
        return {
            "original_query": question, "standalone_query": question, "rewritten_query": question,
            "session_id": session_id, "request_id": request_id, "chat_history": history, "document_ids": document_ids,
            "top_k": top_k or self.config.top_k, "retrieved_documents": [], "graded_documents": [],
            "relevant_documents": [], "answer": "", "sources": [], "evaluation_scores": {},
            "confidence_score": None, "retry_count": 0, "generation_retry_count": 0,
            "fallback_used": False, "error": None, "response_status": "pending", "correction_feedback": "",
            "stream_output": stream_output, "progress_callback": progress_callback,
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
                    "fallback_used": result.get("fallback_used", False), "error": result.get("error")}
        if self.app_store:
            self.app_store.save_exchange(session_id, result.get("original_query", ""), response["answer"], response)
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

        def publish(event):
            events.put(event)

        def run_graph():
            try:
                shortcut = self._feedback_shortcut(question, session_id, document_ids,
                                                   stream_output=True, progress_callback=publish)
                if shortcut:
                    if not shortcut.get("feedback_shortcut", "").startswith("upvoted_synthesis"):
                        publish({"type": "stage", "stage": "feedback_shortcut",
                                 "detail": ("Using a version-checked upvoted answer; retrieval was skipped."
                                            if shortcut["status"] == "passed" else
                                            "Recent matching answers received consecutive downvotes; retrieval was skipped." )})
                    if shortcut.get("feedback_shortcut") == "upvoted_answer":
                        for index in range(0, len(shortcut["answer"]), 32):
                            publish({"type": "token", "text": shortcut["answer"][index:index + 32]})
                    publish({"type": "result", "response": shortcut})
                    return
                state = self._initial_state(question, session_id, document_ids=document_ids, top_k=top_k,
                                            chat_history=chat_history, request_id=request_id,
                                            stream_output=True, progress_callback=publish)
                result = self.graph.invoke(state)
                response = self._format_response(result, session_id)
                publish({"type": "result", "response": response})
            except Exception:
                log.exception("Streaming query failed request_id=%s", request_id)
                publish({"type": "error", "message": "The query could not be completed. Please try again."})
            finally:
                events.put(finished)

        Thread(target=run_graph, name=f"rag-stream-{request_id or 'query'}", daemon=True).start()
        yield {"type": "stage", "stage": "started", "detail": "Starting the documentation assistant…"}
        while True:
            event = events.get()
            if event is finished:
                break
            yield event
