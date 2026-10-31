# How DocuMind Answers Questions

## LangGraph flow, examples, errors, and improvement notes

**Project:** DocuMind — Technical Documentation Assistant  
**Repository:** DS-Intern Assignment  
**Prepared:** 2026-10-31  
**Purpose:** A practical guide for understanding, debugging, and improving the question-answering pipeline.

> This guide describes the current code. A question can be answered by the feedback-memory shortcut before it reaches LangGraph. FastAPI starts the request, and the frontend uses SSE to receive progress and answer text while the request runs.

## Contents

1. Executive summary and architecture
2. Request lifecycle outside and inside the graph
3. Graph flow, steps, and branch rules
4. RAGState field reference
5. Worked traces and edge cases
6. Feedback memory and cached-answer rules
7. Streaming protocol and frontend behavior
8. Model fallback, confidence, and safe responses
9. Debugging playbook
10. Review findings and improvement roadmap
11. Test results and source map

## Quick terms

- **RAG:** Search the documentation first, then ask a model to write from the search results.
- **Chunk:** A smaller piece of an indexed document. The assistant retrieves and cites chunks rather than whole files.
- **Embedding:** A list of numbers representing the meaning of text. Similar questions usually have embeddings that point in similar directions.
- **Vector ID:** The FAISS index's numeric ID for a chunk. Metadata maps that ID back to the document and text.
- **Node:** One step in the graph, implemented as a Python function.
- **Edge:** A connection that says which node runs next. A conditional edge chooses based on the current state.
- **State:** The data carried between nodes for one question, such as the query, retrieved chunks, retries, and draft answer.
- **Judge/evaluator:** The configured model that checks retrieved chunks or scores an answer against evidence.
- **SSE:** A web connection that sends progress and text events from the server as they happen.

## 1. Executive summary and architecture

DocuMind is a retrieval-augmented generation (RAG) assistant. It searches indexed documentation, asks an evaluation model to select useful chunks, asks a generation model to draft from those chunks, then asks the evaluator to compare the draft with the evidence. The graph returns a passed answer only when the minimum score rules are met. It can rewrite the search up to a configured limit when no chunks are useful, and it can regenerate a failed draft once.

A feedback-memory check runs before the graph. It can show similar previous questions, reuse a well-supported upvoted answer, combine several upvoted answers using their current source chunks, or return an insufficient-evidence response after consecutive strong downvotes while the indexed documents have not changed.

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

LangGraph makes the step order and branches explicit. Each node is a Python function that reads the current state and returns changes to it. Progress events sent to the UI are separate from that state; they explain what the backend is doing but do not affect the graph's decisions.

**Source map:** graph and memory logic are in `retrieval/service.py`; state schema in `retrieval/models.py`; model transport/fallback in `retrieval/llm.py`; prompts in `retrieval/prompts.py`; FastAPI and SSE framing in `app.py`; conversation and feedback persistence in `retrieval/store.py`.

## 2. Request lifecycle: wrapper and graph

### 2.1 FastAPI entry point

1. The browser sends `POST /query` with `question`, optional `session_id`, optional `document_ids`/`top_k`, and `stream: true` when it wants SSE.
2. FastAPI validates `QueryRequest`: raw question length is 2–4000 characters. The route trims whitespace and rejects fewer than two remaining characters.
3. `runtime()` lazily creates metadata, ingestion/index service, `AppStore`, and `RAGService`. Health being green proves only that the process responded; it does not prove FAISS, the embedding model, or Gemini is ready.
4. If `stream=true` or `Accept` contains `text/event-stream`, the route returns a `StreamingResponse`; otherwise it calls `RAGService.query` and returns JSON.

### 2.2 Pre-graph feedback-memory branch

Both `query()` and `query_stream()` call `_feedback_shortcut()` before building or invoking the graph. A direct cached response therefore skips query rewriting, FAISS retrieval, chunk grading, normal generation, and sometimes answer evaluation. The multiple-answer synthesis route still generates and evaluates.

The matcher compares the question's embedding with saved feedback embeddings and up to 100 recent distinct questions. It uses cosine similarity, a measure of how close two embeddings are. The match threshold is `FEEDBACK_SIMILARITY_THRESHOLD` (default 0.80). Direct feedback actions use the stricter `FEEDBACK_DIRECT_SIMILARITY_THRESHOLD` (default 0.93). Question embeddings are stored with conversation history; older questions are embedded once when first needed. Feedback reads are limited to the newest 500 entries. Only feedback entries with current sources can affect answer reuse or downvote handling.

> **Memory display versus answer shortcut:** A prior question may be shown even when it has no rating. An unvoted history record never supplies an answer and never triggers downvote suppression; the normal RAG graph still runs.

### 2.3 Graph invocation and persistence

When no shortcut returns an answer, `_initial_state()` creates the request state. Unless explicit `chat_history` is supplied, history comes from `AppStore.history(session_id, max_history_turns)`. The graph executes through `graph.invoke()`. In streaming mode it runs in a worker thread and publishes progress/token/result events into a queue. `_format_response()` converts evaluation fields to the API shape and saves the exchange when `session_id` exists.

## 3. Graph flow, steps, and branch rules

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

Implementation: `RAGService._query_analysis()`. It uses `QUERY_REWRITE_PROMPT` with the generation model and asks for JSON. Inputs are the original question, the current version of the question, the retry number, and recent chat history. It expects `standalone_query` and `rewritten_query`.

| Case | State update | Routing consequence |
|---|---|---|
| Valid JSON and nonblank fields | Set standalone and rewritten query; clear error | Continue to retrieval. |
| First rewrite failure or malformed response | Use current standalone/original for both | Continue with grounded retrieval. |
| Retry rewrite failure | Retain the prior standalone wording | Search may repeat; retry budget remains bounded. |

Example: history says “FastAPI dependencies use Depends.” The user asks “How do I add one to a route?” The model can replace “one” with “a dependency” and make a clearer search phrase. It does not answer at this stage.

### 3.3 Node 2 — `retrieval`

Implementation: `RAGService._retrieval()`. It turns `rewritten_query` into an embedding (using the standalone question if needed), searches FAISS, and maps each returned vector ID back to its chunk and document. A result contains the document and version, section, chunk ID/hash, vector ID, text, and similarity score.

- No filter: ask FAISS for `top_k` neighbors.
- With `document_ids`: search up to the indexed chunk count, filter to allowed document IDs, then take `top_k`. This is exact but can cost more than metadata-aware search.
- An FAISS ID without metadata is skipped.
- Embedding/FAISS failures set `state.error`; the graph routes to `safe_response` after grading.

Example: FAISS returns vectors 11–15. A filter allows only the documents containing vectors 12 and 14, so only those two reach grading.

Edge cases: query analysis checks whether any chunks exist in the selected document scope before calling the rewrite model. If none exist, it skips rewriting and retrieval retries, then the graph returns insufficient evidence. If chunks exist but FAISS returns no relevant results, the normal limited retry path still applies.

### 3.4 Node 3 — `document_grading`

Implementation: `RAGService._document_grading()`. This is the first judge decision. Each chunk is sent with chunk ID, document, section, and text. The judge must return one boolean relevance decision per supplied chunk.

| Situation | State update | Next edge |
|---|---|---|
| No chunks | Empty grades; `should_retry = retry_count < max`; increment retry when used | Rewrite or insufficient. |
| Valid grades; some relevant | Set graded and relevant documents; `should_retry=false` | Generate. |
| Valid grades; none relevant | Empty relevant docs; retry if budget remains | Rewrite or insufficient. |
| Missing grade, wrong type, invalid JSON, model outage | `error=judge_unavailable` | Safe response. |

The evaluator is told to reject a chunk that is only about the same topic but does not help answer the question. Generation sees only `relevant_documents`. Edge case: grades are matched by `chunk_id`; duplicate chunk IDs could apply a grade to the wrong chunk, so IDs need to be unique across the index.

### 3.5 Router after grading

Implementation: `_after_grading()`. Order matters: error first, then relevant documents, then retry flag, then insufficient evidence. Thus an embedding/index/judge error cannot enter generation. `MAX_RETRIEVAL_RETRIES=2` means at most three retrieval/grading passes: the initial pass plus two retries.

### 3.6 Node 4 — `generation`

Implementation: `RAGService._generation()`. It builds source metadata from relevant chunks and sends their document, section, chunk ID, and text to `GENERATION_PROMPT`. The prompt says to use only the excerpts, state evidence gaps, and omit inline `[S1]` tags. Source metadata is returned separately from prose.

When `generation_retry_count` is nonzero, prior evaluator feedback is added to the correction prompt. In streaming mode `_stream_model_answer()` reads model chunks into a private buffer. It does not send answer text to the browser yet. On provider fallback or regeneration it can emit `reset_draft`; no unapproved answer text has been shown.

Edge cases: blank model output becomes a generation error; provider failure routes to safe response. The user sees progress while this node runs, but answer tokens are held until the evaluation node passes.

```text
Evidence: fastapi_dependencies.md / Dependency Injection / chunk_001
         “FastAPI uses Depends() to declare dependencies.”
Draft:   “Use Depends() in the path operation signature to declare a dependency.”
The source list is built from the actual evidence chunk; answer text needs no source label.
```

### 3.7 Node 5 — `answer_evaluation`

Implementation: `RAGService._answer_evaluation()`. It gives the evaluator the original question, the resolved standalone question, the draft, and relevant evidence. The resolved wording helps score follow-ups like “what about async?” while the original wording preserves the user's intent. The JSON response must contain integer scores for faithfulness, relevance, correctness, plus unsupported claims and feedback.

`overall = round(0.45 × faithfulness + 0.30 × relevance + 0.25 × correctness)`.

A pass requires `overall >= MINIMUM_OVERALL_SCORE` and **each** dimension `>= MINIMUM_DIMENSION_SCORE`. Defaults: 75 overall and 70 per dimension. A high overall cannot compensate for one dimension below its floor.

| Dimension | Question answered |
|---|---|
| Faithfulness | Are claims supported by supplied chunks? |
| Relevance | Does the answer address the user’s original question? |
| Correctness | Is it consistent with the evidence? |

The evaluator checks that scores are integers from 0 to 100 and that unsupported claims are strings. A non-empty `unsupported_claims` list now forces a failure even if the numeric scores pass. The list is added to correction feedback. If the answer fails and no regeneration has been used, the graph tries one correction; otherwise it returns the safe response.

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
| `answer`, `sources` | Draft and its source metadata. | Draft stays on the server; only a passed answer is sent to the browser. |
| `evaluation_scores`, `confidence_score` | Judge dimensions, aggregate, status. | Overall is null if judge unavailable/insufficient. |
| `retry_count` | Retrieval rewrites already used. | Initial retrieval is not counted; configured max 0–2. |
| `generation_retry_count` | Corrective generation attempts. | At most one. |
| `fallback_used` | Whether retrieval had to retry. | Does **not** mean Gemini switched models. |
| `error`, `response_status` | Operational error and response status. | Router checks error before success. |
| `correction_feedback` | Judge feedback for regeneration. | Populated by answer evaluation. |
| `should_retry`, `should_regenerate` | Router flags. | Set by the node owning the decision. |
| `stream_output`, `progress_callback` | Streaming mode and progress-event callback. | The callback does not change graph state. |

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

The query is embedded for memory matching. The matcher checks up to 500 newest feedback rows in the same document scope and compares against stored embeddings for up to 100 recent distinct questions. Old history rows without embeddings are filled in the first time they are used. It emits `memory_check`, then `memory_match` or `memory_none`. An unvoted match is shown but does not shortcut; normal FAISS search still runs. A supported, current answer with enough strong upvotes may be reused. Multiple upvoted variants are synthesized from their currently cited chunks and judged again.

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
| Empty index or filter has no indexed chunks | Stop before query rewrite and FAISS search. | Insufficient evidence without model retries. |
| Judge omits grade | Treat as judge failure. | Safe response. |
| No relevant chunks | Retry until budget exhausted. | Insufficient evidence. |
| Blank generation output | Treat as generation error. | Safe response. |
| Invalid score type/range | Reject evaluation output. | Safe response. |
| One dimension under threshold | Per-dimension gate fails even if overall high. | Regenerate once, else safe. |
| Unsupported list non-empty but scores pass | Treat evaluation as failed and include claims in correction feedback. | Regenerate once; otherwise use safe response. |
| Client disconnect during SSE | A cancellation event stops work between graph steps and closes an active model stream. | A blocking non-stream model request may run until its timeout; the graph then stops before its next step. |
| No session ID | Exchange is not persisted; feedback verification cannot locate it. | No persistent experience memory for that request. |

## 6. Feedback memory and cached-answer rules

| Setting | Default | Purpose |
|---|---:|---|
| `FEEDBACK_SIMILARITY_THRESHOLD` | 0.80 | Candidate detection and similar-question display. |
| `FEEDBACK_DIRECT_SIMILARITY_THRESHOLD` | 0.93 | Higher-confidence direct feedback actions. |

Cosine similarity is not a probability. Lower thresholds find more paraphrases but increase false matches.

### Positive feedback flow

1. Feedback is accepted only for an exchange the server can verify in that session.
2. The server allows one vote for each session/question/answer combination. A repeated submission updates the rating instead of adding another vote; existing duplicate rows are reduced to the newest rating during migration.
3. Server computes the question embedding and stores answer, confidence, server-owned sources, version/hash, corpus fingerprint, scope, and rating.
4. At query time, only passed upvotes with non-empty current sources are candidates.
5. At least three strong direct votes for one answer/source signature can reuse that answer.
6. If three or more votes span distinct answer groups, or matches are broader semantic matches, a lightweight model synthesizes against current chunks and the judge verifies.

Votes are preference signals, not evidence. Current documentation remains the synthesis evidence and the evaluator is the final gate.

### Negative feedback flow

The matcher scans newest-first entries. It counts strong (>=0.93) matching downvotes only when the saved document-set fingerprint matches the current indexed documents. Three consecutive qualifying downvotes return fixed insufficient-evidence text; a qualifying upvote resets the counter. Lower-similarity candidates do not affect this action.

Product tradeoff: a downvote may mean poor phrasing, not absent evidence. Suppressing an answer may mislead if the docs can answer it. Consider forcing normal retrieval/evaluation on downvotes rather than returning insufficient evidence.

### Limitations

- History display includes at most 100 newest distinct stored questions.
- Feedback matching checks at most 500 newest rows.
- Old history rows without embeddings are embedded and saved the first time they are compared; new exchanges save their question embedding.
- One vote is counted per session, normalized question, and answer hash. There is no user identity, so separate sessions can still produce separate votes.
- Matching is document-scope-sensitive: feedback saved under a different `document_ids` scope is not eligible.
- Version/hash checks prevent stale cache use; popularity never overrides source freshness.

## 7. Streaming protocol and frontend behavior

```text
POST /query {stream:true}
Content-Type: text/event-stream
event: stage        {stage, detail, optional matches[]}
event: reset_draft                 # internal stream fallback/regeneration reset
event: token         {text}        # emitted only after answer is approved
event: result        {response}    # authoritative final JSON response
event: error         {message}
```

`query_stream()` starts a daemon worker thread. The worker publishes to a queue and the endpoint yields SSE frames. `streamQuestion()` parses frames, updates the current step, appends text, clears text on reset, and completes only after `result`. The progress `<details>` disclosure is closed by default. It shows current step; expansion shows prior steps and similar questions. `App.jsx` clears progress on success/error, and `ChatMessage` only renders it while the message is streaming.

Example event sequence: `memory_check → memory_match → query_analysis → retrieval → source_review → generation → evaluation → approved → token… → result`. If evaluation fails, no answer text has been sent; the graph retries once or returns the safe response. The final answer still arrives in multiple token events, but only after it passes.

> **Streaming contract:** progress arrives while processing. Answer tokens arrive only after evaluation approves the complete answer. The final `result` event remains authoritative and contains the same answer.

## 8. Model fallback, confidence, and safe responses

Generation and judge models are configured independently. `GeminiClient` deduplicates model names and tries the next configured model only for rate limits and temporary service outages. If both roles use the same primary model, their fallback lists are merged in generation-first, judge-second order. Permanent request failures do not trigger fallback. JSON is parsed and node-specific fields are validated.

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

These values are evaluator scores and a weighted aggregate, not calibrated probabilities. Unverified and insufficient responses return null score values rather than inventing confidence.

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
| No answer text until late in the request | stage events, evaluation, approved, token events | Expected: answer text is buffered until the evaluator approves it. |
| Null confidence | terminal status and evaluation error | Expected on insufficient evidence or judge outage. |

Useful log events: `query_pipeline_started`, `http_request_started`, `graph_node_started/completed`, `gemini_request_started/completed`, `gemini_model_fallback`, and `feedback_shortcut_applied`. Correlate with `request_id`. Current logging avoids prompts, document text, and keys.

## 10. Review fixes and remaining limits

The findings below were fixed without changing the normal retrieval-generation-evaluation order. The feedback-memory path stays before the graph, and the API response shape stays the same.

| Earlier finding | Change made | Test coverage |
|---|---|---|
| Answer text was sent before evaluation passed. | Generation text stays in a server buffer. After the graph approves the complete answer, the server sends it in token events and then sends the final result. Failed drafts are not sent as answer text. | test_streaming_sends_answer_only_after_evaluation_approves_it; test_stream_fallback_resets_partial_model_output_without_emitting_draft |
| Repeated feedback could count several times for the same answer in one session. | A unique key uses session, normalized question, and answer hash. Repeating a vote updates the existing row; startup keeps the newest duplicate from old data. | test_feedback_submission_updates_existing_vote_instead_of_counting_twice; test_store_migration_keeps_newest_of_legacy_duplicate_votes |
| A judge could list unsupported claims and still pass on numeric scores. | Any non-empty unsupported_claims forces failure and is included in correction feedback. | test_unsupported_claim_forces_regeneration_even_with_high_scores |
| Follow-up evaluation saw only the short original wording. | Evaluation now receives both the original and resolved standalone questions. | Follow-up test checks that both question forms reach the evaluator. |
| Shared primary model names could overwrite one fallback list. | Fallback lists are merged in stable order with duplicates removed. | Test exercises merged chains through GeminiClient.generate() with a fake transport. |
| History questions were re-embedded on every query and feedback reads were unbounded. | Store history embeddings, backfill older rows once, compare the 100 newest distinct questions, and cap feedback reads at 500 rows. | Memory test checks stored vectors are reused; bounded-read test checks the limit. |
| SSE worker ignored client disconnects. | A cancellation event stops work between graph steps and closes an active generation stream. | test_sse_disconnect_stops_before_retrieval_after_current_model_call |
| Empty index spent model calls on query rewrites. | Check the selected document scope before query rewriting. If there are no indexed chunks, return insufficient evidence without Gemini or FAISS calls. | test_empty_index_skips_rewrite_and_retrieval_retries |
| README threshold and source examples did not match the application. | Updated streaming behavior, both similarity thresholds, the source example, and the feedback request example. | Compared documented values and fields with runtime configuration and response data. |
| Cache freshness and streaming branches lacked coverage. | Added checks for changed chunk hashes, fallback resets, approved token ordering over both the service stream and FastAPI route, vote updates, and model fallback selection. | test_feedback_answer_cache_is_rejected_when_source_chunk_changes; test_fastapi_sse_streams_only_approved_answer_text; related tests above. |

**Remaining limits:** a blocking, non-stream model request cannot be interrupted mid-request after the browser disconnects. It finishes or times out, then the cancellation event stops the next graph step. There is no authenticated user identity; one session can only vote once per answer, but a person can create another session. Gemini was not called live in tests; fallback behavior was checked with a fake transport to avoid network and quota dependence.

**Where to start in the code:** answer buffering and cancellation are in retrieval/service.py (_stream_model_answer, _generation, query_stream) and frontend state in frontend/src/App.jsx; vote uniqueness and saved vectors are in retrieval/store.py; evaluator rules are in retrieval/service.py (_answer_evaluation) and retrieval/prompts.py; fallback merging is in RAGService.__init__; empty-index handling is in _query_analysis and _retrieval; request examples are in README.md.

## 11. Test results and source map

### Verification performed

Command: `.venv\Scripts\python.exe -m pytest -q`  
Result: **25 passed**. Tests cover graph branches, feedback memory, vote updates and legacy migration, source freshness, unsupported-claim handling, follow-up evaluation, empty-index behavior, approved token delivery through the service and HTTP SSE route, stream cancellation, and fallback chains. No live Gemini integration test was run; tests use scripted models or a fake provider transport.

Frontend verification: `npm run build` completed successfully with Vite.

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
