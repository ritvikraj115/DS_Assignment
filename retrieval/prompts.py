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

GENERATION_PROMPT = """Answer the user's technical question using only the supplied documentation excerpts.
If evidence is incomplete, state the limitation. Do not add outside facts.
Add citations inline using only the exact source labels [S1], [S2], etc. provided below.
Do not invent document names, sections, chunk IDs, or citations. Do not include a source label unless it supports the adjacent claim.
{correction}

Question: {question}
Documentation excerpts:
{documents}"""

EVALUATION_PROMPT = """Evaluate the draft answer only against the provided evidence and original question.
Return JSON only with this shape:
{{\"faithfulness\": {{\"score\": integer 0-100, \"reason\": string}},
 \"relevance\": {{\"score\": integer 0-100, \"reason\": string}},
 \"correctness\": {{\"score\": integer 0-100, \"reason\": string}},
 \"unsupported_claims\": [string], \"feedback\": string}}.
Faithfulness means claims are supported by evidence. Relevance means it answers the question directly.
Correctness means it is consistent with the evidence. Identify concrete unsupported claims.

Question: {question}
Draft answer: {answer}
Evidence: {documents}"""
