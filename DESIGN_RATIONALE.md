# DocuMind: Design Rationale

I designed DocuMind around one practical goal: answers should be traceable to the technical documents a user has provided. That is why ingestion, search, generation, and answer checking are separate parts of the app. Each part can be understood and improved without turning the whole system into one large request handler.

## How the workflow fits together

Ingestion turns files and documentation URLs into structured text blocks. The parser keeps useful context such as headings, lists, code, and tables. A stable document ID and content hash let the service skip unchanged documents. When a document changes, it reuses embeddings for unchanged chunks and builds a replacement FAISS index before promoting it. SQLite is the default metadata store; MongoDB can be selected through configuration.

For a question, the retrieval graph first resolves conversational references so a follow-up such as `what about async?` can be searched and evaluated with its subject understood. It retrieves candidate chunks, checks their relevance, drafts an answer from the evidence, and evaluates that answer. A failed draft gets one correction attempt; if it still cannot be supported, the app returns a safe response. Streaming begins only after the answer passes evaluation, so an unverified draft is not shown as if it were final.

## Chunking and embeddings

Chunking is structure-aware and boundary-aware. Blocks under the same heading are grouped; prose is split at sentence or newline boundaries when it grows beyond the configured size. The default target is 1,000 characters with 150 characters of overlap. Code fences and tables are kept together when possible, even if that means a chunk exceeds the target. Each chunk retains its section path for citations.

This is a lightweight semantic strategy: it preserves document structure and natural text boundaries, but it does not use an embedding model to decide where every chunk should split. That keeps ingestion simple and predictable. The default `all-MiniLM-L6-v2` Sentence Transformer runs locally, and normalized vectors are stored in FAISS for similarity search. Both the embedding model and chunk settings can be configured. A future semantic-boundary option should be evaluated against retrieval quality before replacing this behavior.

## Choices and assumptions

I assumed technical documentation is mostly selectable text and that public URLs can be fetched directly. Scanned PDFs need OCR before ingestion. The local embedding model avoids sending document text to a hosted embedding service; answer generation and judging use configured Gemini models, with fallbacks for transient failures. Metadata persistence and the FAISS index are local by default, while MongoDB is optional.

I kept the workflow modest for an assignment-sized app: a small number of graph stages, one answer correction, configurable thresholds, and a bounded feedback history. Evaluation scores are useful checks, but they are model judgments rather than a guarantee that every answer is correct.

## What makes this version distinct

The additions focus on how people actually use a documentation assistant. Follow-up questions can use recent conversation context. Users can give a direct thumbs-up or thumbs-down without writing a comment. Similar prior questions can be surfaced, and strongly supported answers may be reused only while their source chunks are still current. The interface shows the current processing stage, provides confidence and source details, and streams only an approved answer. These choices make the system easier to inspect without exposing internal draft text.

## What I would improve next

With more time, I would measure retrieval and answer quality on a labeled set of real questions, then tune chunk sizes, overlap, and thresholds from those results. I would add authentication before treating feedback as user-specific, improve support for scanned documents with OCR, and make concurrent ingestion safer. I would also add clearer operational metrics for model fallbacks and index rebuilds. Those improvements build on the current modular boundaries instead of requiring a new architecture.
