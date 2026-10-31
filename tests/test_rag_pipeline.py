import json

from retrieval.config import RetrievalConfig
from retrieval.llm import ModelError
from retrieval.service import RAGService, UNVERIFIED
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


class FakeMetadata:
    def __init__(self, chunks=None, documents=None):
        self.chunks = chunks or CHUNKS[:1]
        self.documents = documents or DOCUMENTS[:1]

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
    assert result["confidence"]["overall"] is None
    assert result["confidence"]["status"] == "failed"
    assert result["sources"] == []


def test_follow_up_uses_session_history_to_resolve_standalone_query():
    model = ScriptedModel(rewrites=[{"standalone_query": "How do I declare FastAPI dependencies?",
                                    "rewritten_query": "FastAPI dependency declaration Depends"}])
    rag, _, ingestion = make_rag(model)
    rag.query("How does FastAPI declare dependencies?", session_id="s1",
              chat_history=[{"role": "user", "content": "Tell me about FastAPI dependencies."},
                            {"role": "assistant", "content": "They use Depends."}])

    rewrite_prompt = next(prompt for _, prompt, _ in model.calls if prompt.startswith("Rewrite"))
    assert "They use Depends." in rewrite_prompt
    assert ingestion.embedder.queries[-1] == "FastAPI dependency declaration Depends"


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
    feedback_response = client.post("/feedback", json={"question": "Was that useful?", "rating": "up", "session_id": "api-session"})
    assert feedback_response.status_code == 200
    assert feedback_response.json()["status"] == "recorded"
    assert len(store.history("api-session")) == 2
