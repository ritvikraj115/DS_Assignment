import json
import threading
import time
import types

import pytest

from retrieval.config import RetrievalConfig
from retrieval.llm import GeminiClient, ModelError
from retrieval.schemas import (AnswerEvaluationOutput, DocumentGradingOutput, QueryRewriteOutput,
                               parse_structured_output)
from retrieval.service import QueryCancelled, RAGService, UNVERIFIED
from retrieval.store import AppStore


CHUNKS = [
    {"chunk_id": "chunk_001", "document_id": "doc-1", "text": "FastAPI uses Depends() to declare dependencies.",
     "section": {"title": "Dependency Injection", "path": ["Guide", "Dependency Injection"]},
     "content_hash": "a", "vector_id": 11, "embedding": [1.0, 0.0]},
    {"chunk_id": "chunk_002", "document_id": "doc-2", "text": "FastAPI applications can declare route handlers.",
     "section": {"title": "Routes", "path": ["Guide", "Routes"]},
     "content_hash": "b", "vector_id": 12, "embedding": [0.0, 1.0]},
]
DOCUMENTS = [
    {"document_id": "doc-1", "filename": "fastapi_dependencies.md", "source_url": None, "version": 1,
     "content_hash": "doc-hash-1", "created_at": "2026-10-31T00:00:00Z", "updated_at": "2026-10-31T00:00:00Z"},
    {"document_id": "doc-2", "filename": "fastapi_routes.md", "source_url": None, "version": 1,
     "content_hash": "doc-hash-2", "created_at": "2026-10-31T00:00:00Z", "updated_at": "2026-10-31T00:00:00Z"},
]


@pytest.mark.parametrize(("schema", "payload"), [
    (QueryRewriteOutput, {"standalone_query": "follow-up"}),
    (DocumentGradingOutput, {"grades": [{"chunk_id": "c1", "relevant": "yes", "reason": "ok"}]}),
    (DocumentGradingOutput, {"grades": [
        {"chunk_id": "c1", "relevant": True, "reason": "ok"},
        {"chunk_id": "c1", "relevant": False, "reason": "duplicate"},
    ]}),
    (AnswerEvaluationOutput, {
        "faithfulness": {"score": 101, "reason": "ok"},
        "relevance": {"score": 90, "reason": "ok"},
        "correctness": {"score": 90, "reason": "ok"},
        "unsupported_claims": [], "feedback": "",
    }),
])
def test_structured_model_outputs_are_pydantic_validated(schema, payload):
    with pytest.raises(ModelError, match="failed .* validation"):
        parse_structured_output(json.dumps(payload), schema)


class FakeMetadata:
    def __init__(self, chunks=None, documents=None):
        self.chunks = CHUNKS[:1] if chunks is None else chunks
        self.documents = DOCUMENTS[:1] if documents is None else documents

    def get_chunks(self):
        return self.chunks

    def list_documents(self):
        return self.documents


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def encode(self, texts):
        self.queries.extend(texts)
        return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]


class FakeIndexer:
    def __init__(self, hits=None):
        self.hits = hits or [[(11, 0.92)]]
        self.calls = 0

    def search(self, vector, top_k):
        selected = self.hits[min(self.calls, len(self.hits) - 1)]
        self.calls += 1
        selected = selected[:top_k]
        return [item[0] for item in selected], [item[1] for item in selected]


def evaluation(faith=95, relevance=90, correctness=90, feedback="Remove the unsupported detail.", unsupported=None):
    return {"faithfulness": {"score": faith, "reason": "Claims match evidence."},
            "relevance": {"score": relevance, "reason": "It answers the question."},
            "correctness": {"score": correctness, "reason": "Consistent with docs."},
            "unsupported_claims": unsupported or [], "feedback": feedback}


class ScriptedModel:
    def __init__(self, grades=None, evaluations=None, answers=None, rewrites=None, fail_at=None):
        self.grades = list(grades or [True])
        self.evaluations = list(evaluations or [evaluation()])
        self.answers = list(answers or ["FastAPI uses `Depends()` to declare dependencies. [S1]"])
        self.rewrites = list(rewrites or [])
        self.fail_at = fail_at
        self.calls = []

    def generate(self, model, prompt, *, json_mode=False):
        self.calls.append((model, prompt, json_mode))
        if self.fail_at and self.fail_at in prompt:
            raise ModelError("Gemini rate limit (429)")
        if prompt.startswith("Rewrite"):
            if self.rewrites:
                return json.dumps(self.rewrites.pop(0))
            return json.dumps({"standalone_query": "How do I use FastAPI Depends?", "rewritten_query": "FastAPI Depends dependency declaration"})
        if prompt.startswith("Judge"):
            question_chunks = json.loads(prompt.split("Chunks: ", 1)[1])
            is_relevant = self.grades.pop(0) if self.grades else True
            return json.dumps({"grades": [{"chunk_id": chunk["chunk_id"], "relevant": is_relevant,
                                            "reason": "Relevant documentation." if is_relevant else "Different topic."}
                               if not isinstance(is_relevant, dict) else
                               {"chunk_id": chunk["chunk_id"], "relevant": is_relevant.get(chunk["chunk_id"], False),
                                "reason": "Relevant documentation." if is_relevant.get(chunk["chunk_id"], False) else "Different topic."}
                               for chunk in question_chunks]})
        if prompt.startswith("Answer"):
            return self.answers.pop(0) if self.answers else "Corrected answer. [S1]"
        if prompt.startswith("Evaluate"):
            return json.dumps(self.evaluations.pop(0) if self.evaluations else evaluation())
        raise AssertionError(f"Unexpected prompt: {prompt[:50]}")

    def generate_stream(self, model, prompt, *, on_fallback=None):
        text = self.generate(model, prompt)
        for start in range(0, len(text), 12):
            yield text[start:start + 12]


def make_rag(model=None, *, hits=None, chunks=None, documents=None, app_store=None, retries=2):
    class FakeIngestion:
        pass
    ingestion = FakeIngestion()
    ingestion.embedder = FakeEmbedder()
    ingestion.metadata = FakeMetadata(chunks, documents)
    ingestion.indexer = FakeIndexer(hits)
    model = model or ScriptedModel()
    config = RetrievalConfig(generation_model="test-generation", judge_model="test-judge", top_k=5,
                             max_retrieval_retries=retries)
    return RAGService(ingestion, config, model, app_store), model, ingestion


def test_relevant_retrieval_generation_evaluation_and_real_citations():
    rag, model, _ = make_rag()
    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "passed"
    assert result["confidence"] == {
        "overall": 92, "faithfulness": 95, "relevance": 90, "correctness": 90,
        "status": "passed", "reasons": {
            "faithfulness": "Claims match evidence.", "relevance": "It answers the question.",
            "correctness": "Consistent with docs.",
        }, "unsupported_claims": [],
    }
    assert result["sources"][0]["document"] == "fastapi_dependencies.md"
    assert result["sources"][0]["section"] == "Dependency Injection"
    assert result["sources"][0]["chunk_id"] == "chunk_001"
    assert "test-generation" in [call[0] for call in model.calls]
    assert "test-judge" in [call[0] for call in model.calls]


def test_irrelevant_first_retrieval_rewrites_and_succeeds():
    model = ScriptedModel(grades=[False, True])
    rag, _, ingestion = make_rag(model, hits=[[(11, .9)], [(12, .8)]], chunks=CHUNKS, documents=DOCUMENTS)
    result = rag.query("What is route declaration?")

    assert result["status"] == "passed"
    assert result["retry_count"] == 1
    assert result["fallback_used"] is True
    assert ingestion.embedder.queries[-1] == "FastAPI Depends dependency declaration"


def test_mixed_grades_filter_irrelevant_chunks_before_generation():
    model = ScriptedModel(grades=[{"chunk_001": True, "chunk_002": False}])
    rag, _, _ = make_rag(model, hits=[[(11, .92), (12, .81)]], chunks=CHUNKS, documents=DOCUMENTS)
    result = rag.query("How does FastAPI declare dependencies?")
    generation_prompt = next(prompt for _, prompt, _ in model.calls if prompt.startswith("Answer"))

    assert result["status"] == "passed"
    assert [source["chunk_id"] for source in result["sources"]] == ["chunk_001"]
    assert "chunk_001" in generation_prompt
    assert "chunk_002" not in generation_prompt


def test_retrieval_retries_exhausted_returns_insufficient_without_generation():
    model = ScriptedModel(grades=[False, False, False])
    rag, _, _ = make_rag(model, retries=2)
    result = rag.query("Unrelated question")

    assert result["status"] == "insufficient_evidence"
    assert result["retry_count"] == 2
    assert result["answer"].startswith("I couldn't find enough")
    assert result["sources"] == []
    assert not any(prompt.startswith("Answer") for _, prompt, _ in model.calls)


def test_failed_answer_is_regenerated_once_with_judge_feedback():
    model = ScriptedModel(evaluations=[evaluation(40, 70, 60), evaluation()],
                          answers=["Depends() always caches results. [S1]", "Use Depends() to declare dependencies. [S1]"])
    rag, _, _ = make_rag(model)
    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "passed"
    assert result["generation_retry_count"] == 1
    assert len([1 for _, prompt, _ in model.calls if prompt.startswith("Answer")]) == 2
    assert "unsupported detail" in [prompt for _, prompt, _ in model.calls if prompt.startswith("Answer")][1]


def test_regenerated_answer_failure_returns_safe_unverified_response():
    model = ScriptedModel(evaluations=[evaluation(30, 60, 50), evaluation(40, 60, 50)],
                          answers=["Unsupported claim. [S1]", "Still unsupported. [S1]"])
    rag, _, _ = make_rag(model)
    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "unverified"
    assert result["answer"] == UNVERIFIED
    assert result["confidence"]["overall"] == 48
    assert result["confidence"]["status"] == "failed"
    assert result["sources"] == []


def test_failed_regeneration_keeps_numeric_confidence_and_internal_source_references(tmp_path):
    model = ScriptedModel(evaluations=[evaluation(30, 60, 50), evaluation(40, 60, 50)],
                          answers=["Unsupported claim.", "Still unsupported."])
    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(model, app_store=store)

    response = rag.query("How does FastAPI declare dependencies?", session_id="low-confidence")
    saved = store.verified_exchange("low-confidence", "How does FastAPI declare dependencies?", response["answer"])

    assert response["confidence"]["overall"] == 48
    assert response["sources"] == []
    assert saved["memory_sources"][0]["chunk_id"] == "chunk_001"


def test_unsupported_claim_forces_regeneration_even_with_high_scores():
    model = ScriptedModel(
        evaluations=[evaluation(98, 98, 98, unsupported=["The dependency is cached forever."]), evaluation()],
        answers=["The dependency is cached forever.", "Use Depends() to declare a dependency."],
    )
    rag, _, _ = make_rag(model)

    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "passed"
    assert result["generation_retry_count"] == 1
    generation_prompts = [prompt for _, prompt, _ in model.calls if prompt.startswith("Answer")]
    assert "The dependency is cached forever" in generation_prompts[1]


def test_empty_index_skips_rewrite_and_retrieval_retries():
    rag, model, ingestion = make_rag(chunks=[], documents=[])

    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "insufficient_evidence"
    assert result["retry_count"] == 0
    assert not model.calls
    assert ingestion.indexer.calls == 0


def test_follow_up_uses_session_history_to_resolve_standalone_query():
    model = ScriptedModel(rewrites=[{"standalone_query": "How do I declare FastAPI dependencies?",
                                    "rewritten_query": "FastAPI dependency declaration Depends"}])
    rag, _, ingestion = make_rag(model)
    rag.query("How does FastAPI declare dependencies?", session_id="s1",
              chat_history=[{"role": "user", "content": "Tell me about FastAPI dependencies."},
                            {"role": "assistant", "content": "They use Depends."}])

    rewrite_prompt = next(prompt for _, prompt, _ in model.calls if prompt.startswith("Rewrite"))
    evaluation_prompt = next(prompt for _, prompt, _ in model.calls if prompt.startswith("Evaluate"))
    assert "They use Depends." in rewrite_prompt
    assert ingestion.embedder.queries[-1] == "FastAPI dependency declaration Depends"
    assert "Original user question: How does FastAPI declare dependencies?" in evaluation_prompt
    assert "Resolved standalone question: How do I declare FastAPI dependencies?" in evaluation_prompt


def test_judge_rate_limit_returns_unverified_and_never_passes():
    model = ScriptedModel(fail_at="Judge whether")
    rag, _, _ = make_rag(model)
    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "unverified"
    assert result["confidence"]["overall"] is None
    assert result["error"] == "judge_unavailable"
    assert result["answer"] == UNVERIFIED


def test_generation_provider_failure_is_safe():
    model = ScriptedModel(fail_at="Answer the user's")
    rag, _, _ = make_rag(model)
    result = rag.query("How does FastAPI declare dependencies?")

    assert result["status"] == "unverified"
    assert result["answer"] == UNVERIFIED
    assert result["confidence"]["overall"] is None


def test_feedback_is_stored_separately(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    result = store.save_feedback("session-1", "Was this helpful?", "down", "The answer was too broad.")

    assert result["status"] == "recorded"
    row = store.connection.execute("SELECT * FROM feedback WHERE feedback_id=?", (result["feedback_id"],)).fetchone()
    assert row["rating"] == "down"
    assert row["comment"] == "The answer was too broad."
    assert store.history("session-1") == []


def test_prior_question_memory_is_checked_without_feedback(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    store.save_exchange("session-1", "How does FastAPI declare dependencies?", "They use Depends.")
    rag, _, _ = make_rag(app_store=store)
    events = []

    result = rag._feedback_shortcut(
        "How does FastAPI declare dependencies?", "session-2", None,
        progress_callback=events.append,
    )

    assert result is None
    match_event = next(event for event in events if event["stage"] == "memory_match")
    assert match_event["matches"][0]["question"] == "How does FastAPI declare dependencies?"
    assert match_event["matches"][0]["rating"] is None
    stored = store.recent_question_records()[0]
    assert stored["question_embedding"] == [1.0, 0.0]
    embedded_count = len(rag.ingestion.embedder.queries)
    rag._feedback_shortcut("How does FastAPI declare dependencies?", "session-3", None)
    assert len(rag.ingestion.embedder.queries) == embedded_count + 1


def test_feedback_submission_updates_existing_vote_instead_of_counting_twice(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    first = store.save_feedback("session-1", "How do I declare a dependency?", "up", None,
                                answer="Use Depends().")
    second = store.save_feedback("session-1", "  HOW DO I DECLARE A DEPENDENCY?  ", "down", None,
                                 answer="Use Depends().")

    rows = store.feedback_entries()
    assert second["status"] == "updated"
    assert first["feedback_id"] == second["feedback_id"]
    assert len(rows) == 1
    assert rows[0]["rating"] == "down"


def test_store_migration_keeps_newest_of_legacy_duplicate_votes(tmp_path):
    path = tmp_path / "app.sqlite3"
    store = AppStore(path)
    store.save_feedback("session-1", "How do I declare a dependency?", "up", None,
                        answer="Use Depends().")
    store.connection.execute("DROP INDEX ux_feedback_session_question_answer")
    original = dict(store.connection.execute("SELECT * FROM feedback").fetchone())
    original["feedback_id"] = "legacy-duplicate"
    original["rating"] = "down"
    original["created_at"] = "9999-01-01T00:00:00Z"
    columns = list(original)
    store.connection.execute(
        f"INSERT INTO feedback ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [original[column] for column in columns],
    )
    store.connection.commit()

    migrated = AppStore(path)

    rows = migrated.feedback_entries()
    assert len(rows) == 1
    assert rows[0]["rating"] == "down"
    assert rows[0]["feedback_id"] == "legacy-duplicate"


def test_feedback_entries_are_bounded(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    for index in range(4):
        store.save_feedback(f"session-{index}", f"Question {index}", "up", None,
                            answer=f"Answer {index}")

    assert len(store.feedback_entries(limit=2)) == 2


def test_feedback_answer_cache_is_rejected_when_source_chunk_changes(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    question = "How does FastAPI declare dependencies?"
    answer = "Use Depends() to declare a dependency."
    fingerprint = rag._feedback_corpus_fingerprint(DOCUMENTS[:1])
    sources = [{"document": "fastapi_dependencies.md", "section": "Dependency Injection",
                "chunk_id": "chunk_001", "document_id": "doc-1", "document_version": 1,
                "chunk_hash": "a"}]
    store.save_feedback("vote-1", question, "up", None, answer=answer, sources=sources,
                        confidence={"overall": 90, "status": "passed"},
                        question_embedding=[1.0, 0.0], corpus_fingerprint=fingerprint,
                        response_status="passed")

    cached = rag._feedback_shortcut(question, "query-session", None)
    assert cached["feedback_shortcut"] == "upvoted_answer"

    changed_chunks = [{**CHUNKS[0], "content_hash": "new-chunk-hash"}]
    changed_rag, _, _ = make_rag(chunks=changed_chunks, documents=DOCUMENTS[:1], app_store=store)
    assert changed_rag._feedback_shortcut(question, "query-session", None) is None


def test_feedback_shortcut_needs_one_confident_positive_and_keeps_response_array(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    question = "How does FastAPI declare dependencies?"
    fingerprint = rag._feedback_corpus_fingerprint(DOCUMENTS[:1])
    sources = [{"document": "fastapi_dependencies.md", "section": "Dependency Injection",
                "chunk_id": "chunk_001", "document_id": "doc-1", "document_version": 1,
                "chunk_hash": "a"}]
    for index, answer in enumerate(("Use Depends() for dependencies.", "Declare Depends() in a route.")):
        store.save_feedback(f"voter-{index}", question, "up", None, answer=answer, sources=sources,
                            confidence={"overall": 90, "status": "passed"}, question_embedding=[1.0, 0.0],
                            corpus_fingerprint=fingerprint, response_status="passed")

    result = rag._feedback_shortcut(question, "query-session", None)

    assert result["feedback_shortcut"] == "upvoted_answer"
    assert result["feedback_score"] == 1
    assert len(result["feedback_responses"]) == 2


def test_negative_feedback_requires_net_below_minus_one_and_low_confidence(tmp_path):
    question = "How does FastAPI declare dependencies?"

    def seed_store(confidence):
        store = AppStore(tmp_path / f"{confidence}.sqlite3")
        rag, _, _ = make_rag(app_store=store)
        fingerprint = rag._feedback_corpus_fingerprint(DOCUMENTS[:1])
        for session_id in ("voter-1", "voter-2"):
            store.save_feedback(session_id, question, "down", None, answer="Unverified answer.",
                                confidence={"overall": confidence, "status": "failed"},
                                question_embedding=[1.0, 0.0], corpus_fingerprint=fingerprint,
                                response_status="unverified")
        return rag

    low_confidence = seed_store(60)._feedback_shortcut(question, "query-session", None)
    high_confidence = seed_store(90)._feedback_shortcut(question, "query-session", None)

    assert low_confidence["status"] == "insufficient_evidence"
    assert high_confidence is None


def test_feedback_similarity_thresholds_deduplicate_to_canonical_question(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    store.save_feedback("voter-1", "How do I declare dependencies?", "up", None,
                        answer="Use Depends().", confidence={"overall": 90},
                        question_embedding=[1.0, 0.0])

    canonical = rag.canonical_feedback_question(
        "How do I use Depends in FastAPI?", "down", None, [1.0, 0.0])

    assert rag.config.feedback_positive_similarity_threshold == 0.72
    assert rag.config.feedback_negative_similarity_threshold == 0.80
    assert canonical == "How do I declare dependencies?"


def test_positive_and_negative_feedback_use_their_own_similarity_cutoffs(tmp_path):
    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    store.save_feedback("voter-1", "How do I declare dependencies?", "up", None,
                        answer="Use Depends().", question_embedding=[0.75, (1 - 0.75 ** 2) ** 0.5])

    positive_match = rag.canonical_feedback_question("Similar follow-up", "up", None, [1.0, 0.0])
    negative_match = rag.canonical_feedback_question("Similar follow-up", "down", None, [1.0, 0.0])

    assert positive_match == "How do I declare dependencies?"
    assert negative_match == "Similar follow-up"


def test_followup_feedback_is_saved_under_resolved_question(tmp_path):
    from fastapi.testclient import TestClient
    from app import create_app

    store = AppStore(tmp_path / "app.sqlite3")
    store.save_exchange("session-1", "How do I use FastAPI dependencies?", "Use Depends() in the route.")
    model = ScriptedModel(rewrites=[{"standalone_query": "How do I declare FastAPI dependencies?",
                                    "rewritten_query": "FastAPI dependency declaration Depends"}])
    rag, _, _ = make_rag(model, app_store=store)
    client = TestClient(create_app(rag_service=rag, app_store=store))
    response = client.post("/query", json={"question": "Retry that answer", "session_id": "session-1"})
    assert response.status_code == 200
    assert response.json()["feedback_question"] == "How do I declare FastAPI dependencies?"

    feedback = client.post("/feedback", json={
        "question": "Retry that answer", "rating": "up", "session_id": "session-1",
        "answer": response.json()["answer"],
    })

    assert feedback.status_code == 200
    assert store.feedback_entries()[0]["question"] == "How do I declare FastAPI dependencies?"


def test_shared_primary_model_merges_fallback_chains_and_client_uses_them():
    config = RetrievalConfig(generation_model="shared", judge_model="shared",
                             generation_fallback_models=("gen-lite", "common-lite"),
                             judge_fallback_models=("judge-lite", "gen-lite"))
    fallbacks = RAGService._fallback_model_map(config)
    assert fallbacks == {"shared": ("gen-lite", "common-lite", "judge-lite")}

    client = GeminiClient.__new__(GeminiClient)
    client.model_fallbacks = fallbacks
    calls = []

    def generate_once(self, model, prompt, *, json_mode=False):
        calls.append(model)
        if model in {"shared", "gen-lite"}:
            raise ModelError("temporary failure", retryable=True)
        return "fallback response"

    client._generate_once = types.MethodType(generate_once, client)
    assert client.generate("shared", "test prompt") == "fallback response"
    assert calls == ["shared", "gen-lite", "common-lite"]


def test_streaming_sends_answer_only_after_evaluation_approves_it():
    rag, _, _ = make_rag()

    events = list(rag.query_stream("How does FastAPI declare dependencies?"))

    assert events[-1]["type"] == "result"
    assert events[-1]["response"]["status"] == "passed"
    assert not any(event["type"] == "draft_token" for event in events)
    approved_index = next(index for index, event in enumerate(events)
                          if event["type"] == "stage" and event.get("stage") == "approved")
    token_indices = [index for index, event in enumerate(events) if event["type"] == "token"]
    assert token_indices and min(token_indices) > approved_index
    streamed = "".join(event["text"] for event in events if event["type"] == "token")
    assert streamed == events[-1]["response"]["answer"]


def test_stream_fallback_resets_partial_model_output_without_emitting_draft():
    class FallbackStreamClient:
        def generate_stream(self, model, prompt, *, on_fallback=None):
            yield "partial from failed model"
            on_fallback()
            yield "approved fallback answer"

    rag, _, _ = make_rag()
    rag.model_client = FallbackStreamClient()
    events = []

    answer = rag._stream_model_answer("primary", "prompt", events.append)

    assert answer == "approved fallback answer"
    assert events == [{"type": "reset_draft"}]


def test_sse_disconnect_stops_before_retrieval_after_current_model_call():
    entered = threading.Event()
    release = threading.Event()

    class BlockingRewriteModel(ScriptedModel):
        def generate(self, model, prompt, *, json_mode=False):
            if prompt.startswith("Rewrite"):
                entered.set()
                release.wait(timeout=3)
            return super().generate(model, prompt, json_mode=json_mode)

    model = BlockingRewriteModel()
    rag, _, ingestion = make_rag(model)
    stream = rag.query_stream("How does FastAPI declare dependencies?")
    assert next(stream)["stage"] == "started"
    while True:
        event = next(stream)
        if event.get("stage") == "query_analysis":
            break
    assert entered.wait(timeout=1)
    stream.close()
    release.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(thread.name == "rag-stream-query" for thread in threading.enumerate()):
        time.sleep(.01)
    assert ingestion.indexer.calls == 0


def test_fastapi_sse_streams_only_approved_answer_text(tmp_path):
    from fastapi.testclient import TestClient
    from app import create_app

    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    client = TestClient(create_app(rag_service=rag, app_store=store))

    response = client.post("/query", json={
        "question": "How does FastAPI declare dependencies?",
        "session_id": "stream-session", "stream": True,
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = []
    for frame in response.text.split("\n\n"):
        name, data = None, None
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if name and data is not None:
            parsed.append((name, data))

    assert parsed[-1][0] == "result"
    assert parsed[-1][1]["response"]["status"] == "passed"
    assert all(name != "draft_token" for name, _ in parsed)
    approved = next(index for index, (name, data) in enumerate(parsed)
                    if name == "stage" and data.get("stage") == "approved")
    token_positions = [index for index, (name, _) in enumerate(parsed) if name == "token"]
    assert token_positions and min(token_positions) > approved


def test_fastapi_health_query_documents_and_feedback_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from app import create_app

    store = AppStore(tmp_path / "app.sqlite3")
    rag, _, _ = make_rag(app_store=store)
    client = TestClient(create_app(rag_service=rag, app_store=store))

    health = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert health.json() == {"status": "ok"}
    assert health.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert health.headers.get("x-request-id")
    query_response = client.post("/query", json={"question": "How does FastAPI declare dependencies?", "session_id": "api-session"})
    assert query_response.status_code == 200
    assert query_response.json()["confidence"]["status"] == "passed"
    docs = client.get("/documents").json()["documents"]
    assert docs[0]["filename"] == "fastapi_dependencies.md"
    assert "chunks" not in docs[0]
    feedback_response = client.post("/feedback", json={
        "question": "How does FastAPI declare dependencies?", "rating": "up", "session_id": "api-session",
        "answer": query_response.json()["answer"],
    })
    assert feedback_response.status_code == 200
    assert feedback_response.json()["status"] == "recorded"
    assert len(store.history("api-session")) == 2
