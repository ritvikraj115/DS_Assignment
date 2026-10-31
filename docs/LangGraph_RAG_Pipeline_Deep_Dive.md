# LangGraph RAG Pipeline Deep Dive

## Technical map, worked examples, edge cases, and code review

**Project:** DocuMind — Technical Documentation Assistant  
**Repository:** DS-Intern Assignment  
**Prepared:** 2026-10-31  
**Purpose:** Explain the implementation in enough detail to debug it, verify its behavior, and plan improvements.

> This guide describes the repository as inspected for this review. It distinguishes the LangGraph itself from the feedback-memory shortcut and the FastAPI/SSE wrapper, because those execute before and around the graph.

## Contents

1. Executive summary and architecture
2. Request lifecycle outside and inside the graph
3. Graph topology, nodes, and conditional edges
4. RAGState field reference
5. Worked traces and edge cases
6. Feedback memory and cached-answer rules
7. Streaming protocol and frontend behavior
8. Model fallback, confidence, and safe responses
9. Debugging playbook
10. Review findings and improvement roadmap
11. Test results and source map

## 1. Executive summary and architecture

DocuMind is a retrieval-augmented generation (RAG) assistant. It searches locally indexed documentation, asks a judge model to select relevant chunks, asks a generation model to draft from those chunks, then asks a judge model to score that draft against the evidence. The graph returns a passed answer only when configured score gates are met. It allows a bounded retry to rewrite a search query when no chunks are relevant and one regeneration when answer evaluation fails.

A feedback-memory check runs before the graph. It can report similar previous questions, reuse a sufficiently supported upvoted answer, synthesize several upvoted answers against still-current cited chunks, or suppress an answer after consecutive strong downvotes while the corpus snapshot is unchanged.

```text
POST /query
  ├─ Validate request and initialize runtime
  ├─ Check question/feedback memory
  │    ├─ No reusable shortcut → continue
  │    ├─ Version-valid upvoted answer → return cached response
  │    ├─ Several upvoted variants → synthesize + evaluate
  │    └─ Three strong consecutive downvotes → insufficient evidence
  └─ LangGraph
       query_analysis → retrieval → document_grading
                                  ├─ relevant → generation → answer_evaluation
                                  │                           ├─ pass → final_response
                                  │                           ├─ fail, first time → generation
                                  │                           └─ fail/outage → safe_response
                                  ├─ no relevant, retries remain → query_analysis
                                  ├─ no relevant, exhausted → insufficient_evidence
                                  └─ operational error → safe_response
```

LangGraph contributes the explicit node and edge routing. Nodes are ordinary Python functions that read state and return partial state updates. The progress callback is a side channel for UI events, not durable graph state. The compiled StateGraph controls which node runs next.

**Source map:** graph and memory logic are in `retrieval/service.py`; state schema in `retrieval/models.py`; model transport/fallback in `retrieval/llm.py`; prompts in `retrieval/prompts.py`; FastAPI and SSE framing in `app.py`; conversation and feedback persistence in `retrieval/store.py`.

## 2. Request lifecycle: wrapper and graph

### 2.1 FastAPI entry point

1. The browser sends `POST /query` with `question`, optional `session_id`, optional `document_ids`/`top_k`, and `stream: true` when it wants SSE.
2. FastAPI validates `QueryRequest`: raw question length is 2–4000 characters. The route trims whitespace and rejects fewer than two remaining characters.
3. `runtime()` lazily creates metadata, ingestion/index service, `AppStore`, and `RAGService`. Health being green proves only that the process responded; it does not prove FAISS, the embedding model, or Gemini is ready.
4. If `stream=true` or `Accept` contains `text/event-stream`, the route returns a `StreamingResponse`; otherwise it calls `RAGService.query` and returns JSON.

### 2.2 Pre-graph feedback-memory branch

Both `query()` and `query_stream()` call `_feedback_shortcut()` before building or invoking the graph. A direct cached response therefore skips query rewriting, FAISS retrieval, chunk grading, normal generation, and sometimes answer evaluation. The multiple-answer synthesis route still generates and evaluates.

The matcher compares the query embedding to saved feedback embeddings and embeddings of up to 100 recent distinct conversation questions. Candidate/display matching uses cosine threshold `FEEDBACK_SIMILARITY_THRESHOLD` (default 0.80). A stricter `FEEDBACK_DIRECT_SIMILARITY_THRESHOLD` (default 0.93) gates direct high-confidence feedback actions. Only feedback-backed entries with current sources can influence reuse or downvote suppression.

> **Memory display versus answer shortcut:** A prior question may be shown even when it has no rating. An unvoted history record never supplies an answer and never triggers downvote suppression; the normal RAG graph still runs.

### 2.3 Graph invocation and persistence

When no shortcut returns an answer, `_initial_state()` creates the request state. Unless explicit `chat_history` is supplied, history comes from `AppStore.history(session_id, max_history_turns)`. The graph executes through `graph.invoke()`. In streaming mode it runs in a worker thread and publishes progress/token/result events into a queue. `_format_response()` converts evaluation fields to the API shape and saves the exchange when `session_id` exists.

## 3. Graph topology, nodes, and conditional edges

### 3.1 Exact graph

```text
START → query_analysis → retrieval → document_grading
                                      ├─ state.error → safe_response → END
                                      ├─ relevant docs → generation
                                      │                   ├─ state.error → safe_response → END
                                      │                   └─ answer_evaluation
                                      │                       ├─ error → safe_response → END
                                      │                       ├─ pass → final_response → END
                                      │                       ├─ fail, retry allowed → generation
                                      │                       └─ fail again → safe_response → END
                                      ├─ no relevant + should_retry → query_analysis
                                      └─ no relevant + exhausted → insufficient_evidence → END
```

| From | Condition | To | Meaning |
|---|---|---|---|
| START | Always | query_analysis | Resolve references and optimize query wording. |
| query_analysis | Always | retrieval | Search with rewritten query or fallback. |
| retrieval | Always | document_grading | Empty result and retrieval error are handled downstream. |
| document_grading | `state.error` | safe_response | Do not generate after an operational failure. |
| document_grading | `relevant_documents` non-empty | generation | At least one chunk passed the relevance judge. |
| document_grading | no relevant and `should_retry` | query_analysis | Bounded search rewrite/retry loop. |
| document_grading | no relevant and no retry | insufficient_evidence | Search exhausted; do not invent an answer. |
| generation | `state.error` | safe_response | Generation failed. |
| generation | no error | answer_evaluation | Draft requires independent evidence checking. |
| answer_evaluation | error | safe_response | Judge unavailable or malformed response. |
| answer_evaluation | passed | final_response | Mark passed and finish. |
| answer_evaluation | `should_regenerate` | generation | One corrective regeneration. |
| answer_evaluation | otherwise | safe_response | No answer survived evaluation. |

There is no edge from a terminal node back into the graph. The only loops are the retrieval rewrite loop through `query_analysis` and one evaluation regeneration loop through `generation`.

### 3.2 Node 1 — `query_analysis`

Implementation: `RAGService._query_analysis()`. It uses `QUERY_REWRITE_PROMPT` with the generation model in JSON mode. Inputs are original question, current standalone wording, retry number, and serialized history. It expects `standalone_query` and `rewritten_query`.

| Case | State update | Routing consequence |
|---|---|---|
| Valid JSON and nonblank fields | Set standalone and rewritten query; clear error | Continue to retrieval. |
| First rewrite failure or malformed response | Use current standalone/original for both | Continue with grounded retrieval. |
| Retry rewrite failure | Retain the prior standalone wording | Search may repeat; retry budget remains bounded. |

Example: history says “FastAPI dependencies use Depends.” The user asks “How do I add one to a route?” The model can resolve “one” into a standalone dependency question and produce a search-optimized phrase. It does not answer at this stage.

### 3.3 Node 2 — `retrieval`

Implementation: `RAGService._retrieval()`. It embeds `rewritten_query` (falling back to standalone), searches FAISS, then maps vector IDs to stored chunks and documents. A result contains document identity/version, section/path, chunk ID/hash, vector ID, text, and similarity.

- No filter: ask FAISS for `top_k` neighbors.
- With `document_ids`: search up to the indexed chunk count, filter to allowed document IDs, then take `top_k`. This is exact but can cost more than metadata-aware search.
- An FAISS ID without metadata is skipped.
- Embedding/FAISS failures set `state.error`; the graph routes to `safe_response` after grading.

Example: FAISS returns vectors 11–15. A filter allows only the documents containing vectors 12 and 14, so only those two reach grading.

Edge cases: no indexed chunks or a filter matching no chunks returns an empty list. The current router may spend its bounded retry budget rewriting a query even though an empty corpus cannot yield evidence. This is a candidate optimization.

### 3.4 Node 3 — `document_grading`

Implementation: `RAGService._document_grading()`. This is the first judge decision. Each chunk is sent with chunk ID, document, section, and text. The judge must return one boolean relevance decision per supplied chunk.

| Situation | State update | Next edge |
|---|---|---|
| No chunks | Empty grades; `should_retry = retry_count < max`; increment retry when used | Rewrite or insufficient. |
| Valid grades; some relevant | Set graded and relevant documents; `should_retry=false` | Generate. |
| Valid grades; none relevant | Empty relevant docs; retry if budget remains | Rewrite or insufficient. |
| Missing grade, wrong type, invalid JSON, model outage | `error=judge_unavailable` | Safe response. |

The judge is instructed to reject topical similarity without evidence. Generation sees only `relevant_documents`. Edge case: the lookup is keyed by `chunk_id`; duplicate IDs could associate a judge grade with the wrong chunk, so chunk IDs should be globally unique.

### 3.5 Router after grading

Implementation: `_after_grading()`. Order matters: error first, then relevant documents, then retry flag, then insufficient evidence. Thus an embedding/index/judge error cannot enter generation. `MAX_RETRIEVAL_RETRIES=2` means at most three retrieval/grading passes: the initial pass plus two retries.

### 3.6 Node 4 — `generation`

Implementation: `RAGService._generation()`. It builds source metadata from relevant chunks and sends their document, section, chunk ID, and text to `GENERATION_PROMPT`. The prompt says to use only the excerpts, state evidence gaps, and omit inline `[S1]` tags. Source metadata is returned separately from prose.

When `generation_retry_count` is nonzero, prior evaluator feedback is added to the correction prompt. In streaming mode `_stream_model_answer()` emits `draft_token` while generating. On provider fallback or regeneration it can emit `reset_draft` to clear partial text.

Edge cases: blank model output becomes a generation error; provider failure routes to safe response. **Draft tokens can be visible before answer evaluation passes**; see review findings.

```text
Evidence: fastapi_dependencies.md / Dependency Injection / chunk_001
         “FastAPI uses Depends() to declare dependencies.”
Draft:   “Use Depends() in the path operation signature to declare a dependency.”
The source list is built from the actual evidence chunk; answer text needs no source label.
```

### 3.7 Node 5 — `answer_evaluation`

Implementation: `RAGService._answer_evaluation()`. It gives the judge the **original** question, draft, and relevant evidence. The JSON contract requires integer scores for faithfulness, relevance, correctness, unsupported claims, and feedback.

`overall = round(0.45 × faithfulness + 0.30 × relevance + 0.25 × correctness)`.

A pass requires `overall >= MINIMUM_OVERALL_SCORE` and **each** dimension `>= MINIMUM_DIMENSION_SCORE`. Defaults: 75 overall and 70 per dimension. A high overall cannot compensate for one dimension below its floor.

| Dimension | Question answered |
|---|---|
| Faithfulness | Are claims supported by supplied chunks? |
| Relevance | Does the answer address the user’s original question? |
| Correctness | Is it consistent with the evidence? |

The evaluator validates integer type/range and unsupported-claim list types. It returns `should_regenerate=true` only when failed and `generation_retry_count < 1`; otherwise the graph goes safe. One improvement: `unsupported_claims` is returned but does not independently veto a numeric pass, so define an explicit policy.

### 3.8 Router after evaluation

| Condition | Edge | Result |
|---|---|---|
| Error present | safe_response | Judge failure cannot be passed. |
| `evaluation_scores.passed` | final_response | Successful terminal branch. |
| `should_regenerate` | generation | One correction attempt. |
| Otherwise | safe_response | No passing answer after allowed retry. |

The router does not recalculate the score; `_answer_evaluation()` owns the pass decision.

### 3.9 Terminal nodes

| Node | Behavior | API meaning |
|---|---|---|
| `final_response` | Sets `response_status=passed`; emits approved progress. | Evaluated answer, confidence, and source metadata. |
| `insufficient_evidence` | Fixed insufficient text, no sources. | No relevant documentation after bounded search. |
| `safe_response` | Fixed unverified text, no sources. | Operational failure or draft not approved. |

The safe branches intentionally discard unapproved prose and return no citations, since there is no approved claim to cite.

## 4. RAGState field reference

| Field | Role | Lifecycle / notes |
|---|---|---|
| `original_query` | Exact question used for evaluation and persistence. | Set once; never replaced by search wording. |
| `standalone_query` | Question with conversational references resolved. | Updated by query analysis; falls back to original. |
| `rewritten_query` | Search-oriented wording. | Used by retrieval; can equal standalone. |
| `session_id`, `request_id` | Conversation key and log correlation. | Optional; no session means no saved exchange. |
| `chat_history` | Prior role/content messages passed to rewrite. | Up to configured history turns. |
| `document_ids`, `top_k` | Retrieval scope and result limit. | API top_k range 1–100; service config default otherwise. |
| `retrieved_documents` | FAISS results with evidence/metadata. | Replaced for each retrieval pass. |
| `graded_documents` | Retrieved chunks plus relevance and reason. | Created by grader. |
| `relevant_documents` | Evidence used for generation/evaluation. | Must be non-empty for normal generation. |
| `answer`, `sources` | Draft and its source metadata. | Draft may stream before final verdict. |
| `evaluation_scores`, `confidence_score` | Judge dimensions, aggregate, status. | Overall is null if judge unavailable/insufficient. |
| `retry_count` | Retrieval rewrites already used. | Initial retrieval is not counted; configured max 0–2. |
| `generation_retry_count` | Corrective generation attempts. | At most one. |
| `fallback_used` | Whether retrieval had to retry. | Does **not** mean Gemini switched models. |
| `error`, `response_status` | Operational error and response status. | Router checks error before success. |
| `correction_feedback` | Judge feedback for regeneration. | Populated by answer evaluation. |
| `should_retry`, `should_regenerate` | Router flags. | Set by the node owning the decision. |
| `stream_output`, `progress_callback` | Streaming mode and event side channel. | Callback is not persisted graph state. |

## 5. Worked traces and edge cases

### Trace A — straightforward answer passes

1. User asks “How does FastAPI declare dependencies?”
2. Query analysis creates standalone/search wording.
3. Retrieval embeds it and returns top-k chunks.
4. Grading marks `chunk_001` relevant.
5. Generation drafts from `chunk_001`.
6. Evaluation returns 95/90/90. Overall is `round(42.75 + 27 + 22.5) = 92`; all dimensions exceed 70 and overall exceeds 75.
7. `final_response` marks passed and the response contains a source for `chunk_001`.

### Trace B — first search misses, rewrite finds evidence

```text
Initial query: “Where do I add a route dependency?”
Pass 1: unrelated chunks → relevant_documents=[]
retry_count 0 < max 2 → retry=true; retry_count becomes 1
Router → query_analysis → revised search query
Pass 2: dependency chunk is relevant → generation/evaluation
fallback_used=true (retrieval retry, not model fallback)
```

If pass 2 also misses, retry count becomes 2 and a third search happens. After that, insufficient evidence is terminal. If query rewriting fails, the previous query is reused; repeated search is possible but bounded.

### Trace C — score failure, one regeneration

```text
Draft 1: faithfulness=40, relevance=70, correctness=60
Overall = round(18 + 21 + 15) = 54 → fail
answer_evaluation saves correction_feedback; should_regenerate=true
generation_retry_count becomes 1; graph returns to generation
Draft 2 → pass: final_response; fail/judge outage: safe_response
```

### Trace D — judge or generation unavailable

GeminiClient tries configured fallback models only for retryable rate-limit or temporary-outage failures. If the complete chain fails, grading/evaluation routes to safe response. Query rewriting is more permissive: it reuses original/prior query and retrieval can continue. Fallbacks cannot fix an invalid key, exhausted project quota across all candidates, or provider-wide outage.

### Trace E — similar previous question

The query is embedded for memory matching. The matcher checks feedback rows in the same document scope and embeds up to 100 recent distinct conversation questions. It emits `memory_check`, then `memory_match` or `memory_none`. An unvoted match is shown but does not shortcut; normal FAISS search still runs. A supported, current answer with enough strong upvotes may be reused. Multiple upvoted variants are synthesized from their currently cited chunks and judged again.

### Trace F — corpus changed since feedback

Every candidate answer source is checked against current document version, chunk ID, and chunk content hash. Missing/changed sources disqualify the candidate. A changed corpus fingerprint also disables old downvote suppression. This is conservative: a whole-document version bump can invalidate a cache even if one cited chunk is unchanged, favoring freshness over latency.

### Edge-case matrix

| Case | Current handling | Expected behavior |
|---|---|---|
| Whitespace question | Trimmed route rejects fewer than two chars. | HTTP 422. |
| No history/session | Rewrite still uses original. | Retrieval proceeds without contextual resolution. |
| No memory | Emit no-match and continue. | Normal retrieval. |
| History match with no rating | Show match only. | Normal retrieval, no shortcut/suppression. |
| Feedback embedding absent | Skip row; history matching may still show question. | Normal retrieval unless another candidate qualifies. |
| Memory embedding failure | Log unavailable and continue to graph. | Normal RAG if retrieval embedding works. |
| Retrieval embedding fails | Set error; route to safe response. | Unverified response. |
| FAISS vector lacks metadata | Drop hit. | Use other mapped chunks. |
| Filter matches no chunks | Empty results may retry. | Eventually insufficient; empty-index fast path is recommended. |
| Judge omits grade | Treat as judge failure. | Safe response. |
| No relevant chunks | Retry until budget exhausted. | Insufficient evidence. |
| Blank generation output | Treat as generation error. | Safe response. |
| Invalid score type/range | Reject evaluation output. | Safe response. |
| One dimension under threshold | Per-dimension gate fails even if overall high. | Regenerate once, else safe. |
| Unsupported list non-empty but scores pass | Current numeric pass can still succeed. | Define explicit unsupported-claim policy. |
| Client disconnect during SSE | Worker has no cancellation signal. | Work may continue; cancellation is a roadmap item. |
| No session ID | Exchange is not persisted; feedback verification cannot locate it. | No persistent experience memory for that request. |

## 6. Feedback memory and cached-answer rules

| Setting | Default | Purpose |
|---|---:|---|
| `FEEDBACK_SIMILARITY_THRESHOLD` | 0.80 | Candidate detection and similar-question display. |
| `FEEDBACK_DIRECT_SIMILARITY_THRESHOLD` | 0.93 | Higher-confidence direct feedback actions. |

Cosine similarity is not a probability. Lower thresholds find more paraphrases but increase false matches.

### Positive feedback flow

1. Feedback is accepted only for an exchange the server can verify in that session.
2. Server computes question embedding and stores answer, confidence, server-owned sources, version/hash, corpus fingerprint, scope, and rating.
3. At query time, only passed upvotes with non-empty current sources are candidates.
4. At least three strong direct votes for one answer/source signature can reuse that answer.
5. If three or more votes span distinct answer groups, or matches are broader semantic matches, a lightweight model synthesizes against current chunks and the judge verifies.

Votes are preference signals, not evidence. Current documentation remains the synthesis evidence and the evaluator is the final gate.

### Negative feedback flow

The matcher scans newest-first entries. It counts strong (>=0.93) semantically matching downvotes only when their stored corpus fingerprint equals the current corpus. Three consecutive qualifying downvotes return fixed insufficient-evidence text; a qualifying upvote resets the counter. Lower-similarity candidates do not affect this action.

Product tradeoff: a downvote may mean poor phrasing, not absent evidence. Suppressing an answer may mislead if the docs can answer it. Consider forcing normal retrieval/evaluation on downvotes rather than returning insufficient evidence.

### Limitations

- History display includes at most 100 newest distinct stored questions.
- That history is re-embedded on each query; simple for a small assignment but not scalable.
- Feedback rows are loaded as a full list and currently lack an idempotent/unique vote constraint.
- Matching is document-scope-sensitive: feedback saved under a different `document_ids` scope is not eligible.
- Version/hash checks prevent stale cache use; popularity never overrides source freshness.

## 7. Streaming protocol and frontend behavior

```text
POST /query {stream:true}
Content-Type: text/event-stream
event: stage        {stage, detail, optional matches[]}
event: draft_token  {text}        # emitted while evaluator has not approved yet
event: reset_draft                 # clear partial text after fallback/regeneration
event: token         {text}        # direct cached-answer chunks
event: result        {response}    # authoritative final JSON response
event: error         {message}
```

`query_stream()` starts a daemon worker thread. The worker publishes to a queue and the endpoint yields SSE frames. `streamQuestion()` parses frames, updates the current step, appends text, clears text on reset, and completes only after `result`. The progress `<details>` disclosure is closed by default. It shows current step; expansion shows prior steps and similar questions. `App.jsx` clears progress on success/error, and `ChatMessage` only renders it while the message is streaming.

Example event sequence: `memory_check → memory_match → query_analysis → retrieval → source_review → generation → draft_token… → evaluation → approved → result`. If evaluation fails, `reset_draft` may clear the draft and the final answer becomes the safe response.

> **Streaming contract:** result is authoritative. Text arriving before result is a draft. The UI labels it as pending verification, but users have already seen the text. If strict approved-only output is required, buffer server-side until the answer passes, then stream the approved text.

## 8. Model fallback, confidence, and safe responses

Generation and judge models are configured independently. `GeminiClient` deduplicates model names and tries the next configured model only for rate limits and temporary service outages. Permanent request failures do not trigger fallback. JSON is parsed and node-specific fields are validated.

Example output:

```json
{
  "answer": "Use Depends() in a path operation to declare a dependency.",
  "confidence": {
    "overall": 92,
    "faithfulness": 95,
    "relevance": 90,
    "correctness": 90,
    "status": "passed"
  },
  "sources": [
    {"document": "fastapi_dependencies.md", "section": "Dependency Injection", "chunk_id": "chunk_001"}
  ]
}
```

These values are judge scores and a weighted aggregate, not calibrated probabilities. Unverified and insufficient responses return null score values rather than inventing confidence.

| Status | Typical cause | Returned behavior |
|---|---|---|
| passed | Evidence found and all score gates met. | Answer with sources and scores. |
| insufficient_evidence | No relevant chunks after bounded retries, or feedback suppression. | Fixed insufficient documentation statement. |
| unverified | Generation/judge error or answer fails after one regeneration. | Fixed cannot-verify response. |

## 9. Debugging playbook

| Symptom | Inspect | Interpretation |
|---|---|---|
| 503 from `/query` | request_id logs, runtime initialization, key, FAISS/metadata, judge error | Health may be green while a model/index dependency is not ready. |
| No memory match | `recent_questions()`, feedback rows, session ID, threshold, model | Only 100 newest distinct history questions are compared. |
| History never appears | Confirm previous request had `session_id` and completed | Without a session the exchange is not persisted. |
| Cached response not used | rating/status, vote count, similarity, source versions/hashes | Any stale/missing source disqualifies it. |
| Repeated search | retry_count, rewritten_query, retrieved/relevant counts | Rewrite failure can reuse the same query; loop is bounded. |
| Chunks retrieved but no answer | grading result and judge schema | Retrieved does not mean relevant; all may be rejected. |
| Draft appears and disappears | draft_token, reset_draft, evaluation, result status | Draft was shown before judge approval and then replaced. |
| Null confidence | terminal status and evaluation error | Expected on insufficient evidence or judge outage. |

Useful log events: `query_pipeline_started`, `http_request_started`, `graph_node_started/completed`, `gemini_request_started/completed`, `gemini_model_fallback`, and `feedback_shortcut_applied`. Correlate with `request_id`. Current logging avoids prompts, document text, and keys.

## 10. Review findings and improvement roadmap

### Fixed during this review

| Finding | Impact | Change | Verification |
|---|---|---|---|
| Matcher returned early when feedback table was empty, so unvoted history questions were never compared or shown. | UI could report no memory although prior conversations existed. | Removed early return; history matching runs independently of feedback. | Added `test_prior_question_memory_is_checked_without_feedback`; suite passes. |
| Existing feedback API test sent a question different from the stored exchange and omitted the answer. | Correct server verification returned 409; test payload was stale. | Updated it to submit exact question and returned answer. | Backend suite passes. |

### Open findings

| Priority | Finding | Why it matters | Recommendation |
|---|---|---|---|
| P1 — trust | `draft_token` is published by generation before evaluation passes. | Unsupported text can be seen before it is rejected; reset/final replacement happens later. | Decide between approved-only buffered streaming and explicitly visible draft streaming; keep UI and API contract aligned. |
| P1 — feedback integrity | `/feedback` inserts a row each time; no unique vote key/idempotency. | Repeated submissions can reach the three-vote shortcut threshold. | Add unique session/question/answer key and define vote-change semantics. |
| P2 — evaluator | `unsupported_claims` is validated/returned but does not independently veto a numeric pass. | Judge may list unsupported claims and still pass. | Explicitly reject or reconcile claims. |
| P2 — follow-up evaluation | Generation uses `standalone_query`, while answer evaluation scores against `original_query` only. | For anaphoric follow-ups such as “what about async?”, the evaluator may score an answer against a question that lacks its resolved subject. | Give evaluation both original wording and resolved standalone wording plus relevant history, and clarify which wording is authoritative. |
| P2 — fallback configuration | Fallbacks are keyed by primary model name. If `GENERATION_MODEL` and `JUDGE_MODEL` are configured to the same model with different chains, the dictionary entry for the later role overwrites the earlier chain. | One role can silently use the other role's fallback list. | Merge chains deterministically or validate that shared primary models use the same chain. |
| P2 — memory scaling | Every query re-embeds up to 100 history questions; all feedback rows are read. | Latency and work grow with history/votes. | Persist question embeddings; bound feedback reads/retention or add a nearest-neighbor store. |
| P2 — SSE lifecycle | Daemon query thread has no client-disconnect cancellation. | Expensive work may continue after browser stops listening. | Pass a cancellation event through model/retrieval iteration. |
| P2 — empty corpus | Empty index can consume rewrite retries. | Adds model calls despite no possible evidence. | Detect no indexed chunks before graph retries. |
| P2 — documentation drift | README still says threshold 0.92 and shows an S1 source label not emitted by current source list. | Setup and response examples disagree with runtime. | Update README defaults/schema examples. |
| P3 — test coverage | Tests do not cover SSE resets, shortcut freshness invalidation, duplicate votes, or real fallback chains. | Important branches may regress. | Add focused fake-based tests and one SSE integration test. |

Suggested order: (1) make votes idempotent; (2) choose streaming contract; (3) define unsupported-claim policy; (4) add empty-corpus short-circuit and edge tests; (5) persist embeddings/add SSE cancellation; (6) refresh README.

## 11. Test results and source map

### Verification performed

Command: `.venv\Scripts\python.exe -m pytest -q`  
Result: **14 passed**. This includes the no-feedback conversation-memory regression check and the corrected feedback endpoint request. No live Gemini integration test was run; scripted clients are used in unit tests.

### Source map

| File | Responsibility | Key symbols |
|---|---|---|
| `retrieval/service.py` | Graph nodes/edges, feedback shortcut, query invocation, SSE worker. | `_build_graph`, `_query_analysis`, `_retrieval`, `_document_grading`, `_generation`, `_answer_evaluation`, `_feedback_shortcut`, `query_stream` |
| `retrieval/models.py` | Typed graph state. | `RAGState` |
| `retrieval/config.py` | Models, thresholds, retries, score settings. | `RetrievalConfig` |
| `retrieval/prompts.py` | Rewrite, grade, generation, evaluation contracts. | Four prompt constants |
| `retrieval/llm.py` | Gemini requests, fallback classification, JSON parsing. | `GeminiClient`, `ModelError`, `parse_json_response` |
| `retrieval/store.py` | SQLite conversation/feedback persistence. | `history`, `recent_questions`, `save_exchange`, `save_feedback`, `feedback_entries` |
| `app.py` | FastAPI routes and SSE framing. | `create_app`, `QueryRequest`, `FeedbackRequest` |
| `frontend/src/services/api.js` | SSE reader and stage labels. | `streamQuestion` |
| `frontend/src/App.jsx` | Chat streaming state lifecycle. | `submitQuestion` |
| `frontend/src/components/ChatMessage.jsx` | Progress disclosure, answer/source rendering, feedback. | `ChatMessage` |
| `tests/test_rag_pipeline.py` | Graph, memory, API tests with fakes. | Pipeline tests and scripted model |

### Configuration quick reference

| Variable | Default/range | Meaning |
|---|---|---|
| `RETRIEVAL_TOP_K` | 5; API 1–100 | FAISS candidates sent to judge. |
| `MAX_RETRIEVAL_RETRIES` | 2; valid 0–2 | Query rewrites after initial retrieval. |
| `MINIMUM_DIMENSION_SCORE` | 70; 0–100 | Floor for each judge dimension. |
| `MINIMUM_OVERALL_SCORE` | 75; 0–100 | Weighted score floor. |
| `MAX_HISTORY_TURNS` | 8 | Prior turns passed to rewrite. |
| `FEEDBACK_SIMILARITY_THRESHOLD` | 0.80; 0–1 | Candidate/display threshold. |
| `FEEDBACK_DIRECT_SIMILARITY_THRESHOLD` | 0.93; >= candidate | Direct-action threshold. |
| `GENERATION_MODEL`, `JUDGE_MODEL` | Flash-Lite defaults | Independent roles and configured fallback chains. |

Improve safely by defining the product contract, adding a regression test for that contract, changing the graph/wrapper, then updating the examples in this guide.
