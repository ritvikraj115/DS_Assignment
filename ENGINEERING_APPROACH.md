# Engineering Approach: DocuMind

## The problem I chose to solve

A documentation assistant is only useful when its answer can be traced to current evidence. That creates a few connected engineering problems: documents need to keep their structure when indexed, updates must not leave obsolete text searchable, follow-up questions need their conversational context, and a fluent answer still needs to be checked before it is presented as reliable.

I organized DocuMind around those failure points. Ingestion, retrieval, generation, evaluation, feedback, and the interface have clear boundaries, but they share document and source identifiers so that a decision made in one stage can be checked by the next. The goal was to make the system understandable and recoverable at assignment scale, rather than hide behavior behind a large framework.

## How a question moves through the system

FastAPI validates the request and calls reusable ingestion or RAG services. The query service uses a LangGraph state to pass the original question, its resolved form, retrieved evidence, evaluation results, and retry counts between nodes. Its main path is:

1. **Resolve context.** A follow-up such as “what about async?” is interpreted with the conversation history, producing a standalone question for search. The original wording is retained too, so the response and evaluation still reflect what the user asked.
2. **Check experience memory.** Semantically similar questions are grouped under a canonical question. A matching positive answer can be proposed for reuse only if its confidence is high enough and its cited chunks are still current. A model reframes that material for the new wording and the answer judge checks it. If it is unsuitable, the graph continues to document retrieval. Negative feedback can suppress an answer only under the configured similarity, score, confidence, and corpus-freshness conditions.
3. **Retrieve and grade evidence.** FAISS supplies candidate chunks. The document judge decides whether they support the question. If evidence is inadequate, the graph can rewrite the search question and retry a bounded number of times; it then returns an insufficient-evidence response instead of searching indefinitely.
4. **Generate and evaluate.** The generation model drafts from the retrieved text. A structured evaluator checks relevance, support, and correctness against both the original and resolved question. Unsupported claims veto a pass. One correction attempt is allowed; if the response still does not pass, a safe response is returned.
5. **Present approved output.** Progress events show the current stage and any retrieval retry. Answer text is streamed only after it passes evaluation, so the interface does not expose a draft that may later be rejected. The final answer includes confidence and source references; progress detail stays collapsed after completion.

Pydantic models validate structured model results at the boundary. The model client can try configured lighter fallback models for retryable provider errors or invalid structured responses. This keeps provider instability from silently becoming trusted output; if evaluation remains unavailable, the service uses its safe behavior rather than treating an unchecked draft as approved.

## Keeping the knowledge base current

Ingestion is split into parsing, preprocessing, structure-aware chunking, embedding, metadata storage, and FAISS indexing. The parsers handle Markdown, plain text, HTML, URLs, and common office formats. Headings and section paths are retained; tables become readable Markdown; code blocks and tables are treated as atomic blocks when possible. Prose is grouped and split near paragraph, sentence, or line boundaries.

Each source gets a stable document ID and each document version a SHA-256 content hash. Re-uploading identical content returns an already-indexed result without embedding it again. For changed content, chunk hashes identify unchanged sections. Their chunk IDs, vectors, and FAISS IDs are retained; only new or changed chunks are embedded. Removed chunks are omitted from the replacement index.

The index update is staged before it becomes active. Metadata is committed with the proposed document version, then the staged FAISS index is promoted. If that sequence fails, the service restores prior metadata and rebuilds the prior index. This recovery path matters because a metadata row that points to a missing vector, or a vector that points to obsolete metadata, can produce incorrect citations. When a chunk is replaced or removed, conversation and feedback records citing it are invalidated as well, so an old answer cannot gain a shortcut back into the response path.

SQLite is the no-configuration metadata store and MongoDB can be selected for document and chunk metadata with `MONGODB_URI`. FAISS stays local and persistent. This keeps the default setup easy to run while allowing the document metadata store to be swapped without changing parsing or query logic.

## Chunking and embedding choices

The default chunk target is 1,000 characters with 150 characters of overlap. Section-aware grouping keeps a chunk interpretable when retrieved on its own, while overlap reduces the chance that a fact at a boundary loses its context. Code and tables stay together when possible because splitting a function or a row relationship can make the resulting passage misleading. Large atomic blocks may exceed the target rather than be broken arbitrarily.

`all-MiniLM-L6-v2` creates local, normalized sentence embeddings; FAISS searches those vectors by similarity. This avoids sending the corpus to an embedding API and gives the project a repeatable local baseline. Boundaries are based on document structure and text boundaries, not an embedding-based semantic segmentation model. That is a deliberate simplicity trade-off: it is explainable and inexpensive, though a character target is only an approximation of the model's token limit and may not be optimal for every document collection.

## Feedback as evidence, not an override

Feedback has value only if similar questions share it and the answer remains valid. The current positive grouping threshold is `0.72`; the stricter negative suppression threshold is `0.80`. A vote changes the score for an answer variant under its canonical question, and repeated submission for the same session/question/answer updates that vote rather than inflating the count. Positive answer reuse requires at least one positive vote and the configured confidence floor. The model still has to adapt the answer to the new question, and the judge still has to approve it. If it cannot, regular retrieval runs.

This separation is intentional: feedback identifies useful prior answers, but it does not prove that the same answer fits a new question. Negative feedback similarly signals caution, but only low-confidence answers with sufficiently negative aggregate feedback and unchanged source context can trigger the insufficient-evidence shortcut. When cited chunks change, feedback memory tied to those chunks is removed. Deleting a chat also clears its conversation and feedback records, giving the user a direct reset path.

## Decisions that make this version distinct

- **Freshness reaches beyond the index.** Updating a document also invalidates answer memory that depends on changed chunks. This connects ingestion correctness to feedback correctness.
- **A positive vote is a candidate signal.** Similarity and votes can save repeated work, but current citations and answer evaluation remain in the loop.
- **Retries are bounded and visible.** Users can see whether the system is resolving context, checking memory, retrieving again, or evaluating. Internal progress is separate from the approved answer stream.
- **Failures have explicit outcomes.** Empty or unsupported evidence leads to an insufficient-evidence response; invalid structured output can use configured fallbacks; index promotion failures restore the prior state.
- **The components are replaceable.** Parsing, chunking, embeddings, metadata, indexing, and graph orchestration are separated so a better model or storage choice can be introduced without rewriting the application routes.

These are practical reliability choices rather than claims that the current thresholds or models are universally optimal. The code and configuration keep those decisions visible and adjustable.

## Assumptions, limits, and next steps

I assumed documents are primarily text-based and documentation URLs are publicly reachable. Scanned PDFs need OCR before their content can be indexed. Embeddings run locally, while selected evidence and prompts are sent to the configured generation and judge provider. Session IDs group conversation state; they are not authentication or access control. The local FAISS setup is intended for an assignment-sized corpus, not concurrent multi-node production traffic.

With more time, I would build a labeled question set and measure retrieval recall, answer support, and the quality of feedback shortcuts before tuning thresholds. I would add OCR and authentication, then test simultaneous ingestion and index recovery under process interruption. I would also evaluate token-aware chunk sizing and broader document corpora, and add operational metrics for fallback use, retrieval retries, and judge disagreement. These steps would turn the current reasoned baseline into decisions supported by measured usage.
