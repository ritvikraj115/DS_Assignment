# Engineering Approach: DocuMind

I approached DocuMind as a reliability problem: a useful documentation assistant must find the right evidence, keep its index current, and make it clear when an answer has not been verified. I organized the implementation around those responsibilities so that parsing, indexing, retrieval, generation, and feedback can be understood and changed independently.

## Design decisions

### Keep ingestion and answering separate

The ingestion service parses documents, preserves their structure, chunks them, creates embeddings, and updates storage. The query service uses LangGraph to coordinate search, relevance checks, answer generation, and evaluation. FastAPI connects these reusable services to the UI. This separation keeps request handling small and gives each part a clear place for errors, logging, and tests.

### Make updates safe and repeatable

Each file or URL receives a stable document ID and a SHA-256 content hash. Unchanged documents are skipped. When content changes, the ingestion service compares chunk hashes, reuses embeddings and FAISS IDs for unchanged chunks, and prepares a replacement index before promoting it. Metadata can restore the index after a failed promotion or restart. If chunks are replaced or removed, the app also deletes conversation and feedback records that cite those chunk IDs. These choices prevent stale answers from being reused and avoid redoing embedding work unnecessarily.

SQLite and a local FAISS index are the defaults for a small, low-setup project. MongoDB can store document metadata when configured; conversation and feedback memory remain in SQLite.

### Use a bounded, evidence-first query flow

Before searching, the app resolves follow-up questions against session history and groups semantic feedback under one canonical question. A high-confidence positive answer is only a candidate: the generation model receives the positive answer variants and their current cited chunks, then the judge verifies its proposed response. If either step declines or fails, the app continues through normal LangGraph retrieval. Repeated negative feedback suppresses an answer only when its net score is below -1 and the downvoted answers were low confidence. Retrieval retries are shown as progress steps, followed by insufficient evidence when no suitable sources are found.

Generated answers are evaluated against both the original question and its resolved standalone form. A failed first evaluation gets one correction attempt. If the answer still fails, the app returns a safe response. Pydantic schemas validate structured model outputs, and approved answer text is streamed character by character only after evaluation passes. Progress stays available in a collapsed summary after the answer is complete.

### Preserve the useful shape of technical documents

Chunks follow headings and section paths; prose is split near sentence or line boundaries. The default target is 1,000 characters with 150 characters of overlap. Fenced code and Markdown tables stay together when possible, even if a chunk exceeds the target, because splitting them can lose meaning.

The default `all-MiniLM-L6-v2` Sentence Transformer creates normalized embeddings locally, and FAISS ranks chunks by similarity. The chunker is structure- and boundary-aware; it does not use embeddings to find semantic split points. That keeps the strategy lightweight and explainable. The tradeoff is that character limits do not match token limits exactly, and larger technical blocks can exceed the target.

## Feedback and follow-up questions

Feedback is attached to the resolved question while the original prompt remains in conversation history. Questions are deduplicated at the positive similarity threshold (`0.72`) regardless of vote type; positive answer reuse uses `0.72` and negative suppression uses `0.80`. Votes add `+1` or `-1`, and distinct answers for a canonical question are kept as response variants. A positive response is considered only when its confidence meets the configured minimum (75 by default) and its cited chunks are still current. The generation model receives the positively rated answer variants and current cited text, and the judge must approve its proposed response before it is shown. If either model declines or verification fails, the request continues through normal retrieval. A net score below `-1` suppresses future answers only when the downvoted answers have low confidence and the corpus is unchanged. Keeping answer and source references together lets the app aggregate similar responses without trusting a stale answer.

## Assumptions and next steps

I assumed the corpus is mostly selectable text and that documentation URLs are public and reachable. Scanned PDFs need OCR before ingestion. Embeddings run locally; selected evidence is sent to the configured generation and judge models. Session IDs support conversation grouping but are not user authentication.

With more time, I would benchmark retrieval and answer quality on labeled questions, then tune chunking and thresholds from those results. I would add authentication, OCR support, and stronger concurrency controls for simultaneous ingestion. I would also measure fallback frequency and index recovery behavior in a larger corpus before replacing the simple FAISS setup.
