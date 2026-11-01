QUERY_REWRITE_PROMPT = """Rewrite the user's question as a standalone technical documentation question.
Use conversation history only to resolve references such as 'it' or 'that'. Do not answer.
If this is a retrieval retry, produce a meaningfully different search query while preserving intent.
Return JSON only: {{\"standalone_query\": string, \"rewritten_query\": string}}.

Original question: {original_query}
Current standalone question: {standalone_query}
Retry number: {retry_count}
Conversation history: {chat_history}"""

GRADE_PROMPT = """Judge whether each document chunk provides useful evidence for answering the question.
Return JSON only with {{\"grades\":[{{\"chunk_id\": string, \"relevant\": boolean, \"reason\": string}}]}}.
Grade each supplied chunk exactly once. Be conservative; topical similarity alone is not enough.

Question: {question}
Chunks: {documents}"""

FEEDBACK_ANSWER_PROMPT = """Decide whether the current question can be answered from the supplied current documentation excerpts.
The previous answers were explicitly upvoted for a semantically similar canonical question. Treat them as useful
context and preferred wording, but not as independent evidence. Use only facts supported by the excerpts below.
If the excerpts support a direct answer to the current question, return can_answer=true and write a concise answer.
If they do not, return can_answer=false and an empty answer so the application can run normal retrieval.
Return JSON only: {{"can_answer": boolean, "answer": string, "reason": string}}.

Current question: {question}
Canonical feedback question: {canonical_question}
Upvoted answer variants: {answers}
Current cited documentation excerpts: {documents}"""

GENERATION_PROMPT = """Answer the user's technical question using only the supplied documentation excerpts.
If evidence is incomplete, state the limitation. Do not add outside facts.
Do not add bracketed source markers such as [S1] or [S2] to the answer. Source details are returned separately.
{correction}
{feedback_context}

Question: {question}
Documentation excerpts:
{documents}"""

EVALUATION_PROMPT = """Evaluate the draft answer against the provided evidence and both question forms. Use the resolved standalone question to understand short follow-ups, while checking that the answer still addresses the user's original intent.
Return JSON only with this shape:
{{\"faithfulness\": {{\"score\": integer 0-100, \"reason\": string}},
 \"relevance\": {{\"score\": integer 0-100, \"reason\": string}},
 \"correctness\": {{\"score\": integer 0-100, \"reason\": string}},
 \"unsupported_claims\": [string], \"feedback\": string}}.
Faithfulness means claims are supported by evidence. Relevance means it answers the question directly.
Correctness means it is consistent with the evidence. Identify concrete unsupported claims.

Original user question: {question}
Resolved standalone question: {standalone_query}
Draft answer: {answer}
Evidence: {documents}"""
