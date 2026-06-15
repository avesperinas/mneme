"""Answer synthesis (sub-phase 2.3).

Assembles a grounded prompt from the retrieved context and the question, with
explicit instructions to cite sources and to refuse when the context does not
contain the answer. When retrieval yields nothing relevant, the not-found reply
is returned directly without calling the LLM, so the system cannot fabricate an
answer for an out-of-domain question (spec 2.3 GUARDRAIL).
"""

from __future__ import annotations

from dataclasses import dataclass

from mneme.llm.client import LLMClient, Message
from mneme.retrieval.dense import RetrievedChunk, format_context

NOT_FOUND_MESSAGE = "I could not find that in your notes."

SYSTEM_PROMPT = (
    "You answer strictly from the user's personal notes provided as context.\n"
    "Rules:\n"
    "- Use only the context. Do not rely on outside knowledge.\n"
    "- If the context does not contain the answer, reply exactly: "
    f'"{NOT_FOUND_MESSAGE}"\n'
    "- Cite the source notes you used by their path in square brackets.\n"
    "- Be concise and accurate."
)


@dataclass(slots=True)
class Answer:
    text: str
    used: list[RetrievedChunk]


def build_messages(question: str, context: str) -> list[Message]:
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def synthesize_answer(
    llm: LLMClient,
    question: str,
    retrieved: list[RetrievedChunk],
    *,
    min_score: float = 0.0,
) -> Answer:
    relevant = [item for item in retrieved if item.score >= min_score]
    if not relevant:
        return Answer(NOT_FOUND_MESSAGE, [])
    context = format_context(relevant)
    text = await llm.complete(build_messages(question, context))
    return Answer(text, relevant)
