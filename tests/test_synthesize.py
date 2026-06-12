from collections.abc import AsyncIterator

from mneme.rag import NOT_FOUND_MESSAGE, synthesize_answer
from mneme.rag.synthesize import SYSTEM_PROMPT
from mneme.retrieval.dense import RetrievedChunk
from mneme_ingest.models import Chunk


class FakeLLM:
    def __init__(self, reply: str = "Grounded answer [note.md].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **opts) -> str:
        self.calls.append(messages)
        return self.reply

    async def stream(self, messages, **opts) -> AsyncIterator[str]:
        yield self.reply


def _chunk(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="d::0",
            document_id="d",
            rel_path="note.md",
            heading_path=["Note"],
            text=text,
            tags=[],
            ordinal=0,
            token_count=3,
        ),
        score=score,
    )


async def test_out_of_domain_returns_not_found_without_calling_llm():
    llm = FakeLLM()
    answer = await synthesize_answer(llm, "anything?", [])
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.used == []
    assert llm.calls == []  # the model is never asked, so it cannot fabricate


async def test_below_threshold_is_treated_as_not_found():
    llm = FakeLLM()
    weak = [_chunk("loosely related", score=0.1)]
    answer = await synthesize_answer(llm, "q?", weak, min_score=0.5)
    assert answer.text == NOT_FOUND_MESSAGE
    assert llm.calls == []


async def test_grounded_answer_includes_context_and_cite_instruction():
    llm = FakeLLM(reply="Yes, see [note.md].")
    retrieved = [_chunk("the dense searcher uses embeddings", score=0.8)]
    answer = await synthesize_answer(llm, "how does search work?", retrieved)
    assert answer.text == "Yes, see [note.md]."
    assert answer.used == retrieved

    system, user = llm.calls[0]
    assert system["content"] == SYSTEM_PROMPT
    assert NOT_FOUND_MESSAGE in system["content"]
    assert "the dense searcher uses embeddings" in user["content"]
    assert "how does search work?" in user["content"]
