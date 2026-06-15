"""API request/response contracts (spec 3.4)."""

from __future__ import annotations

from pydantic import BaseModel

Mode = str  # "naive" | "hybrid" | "graph" | "agent" (only naive in phase 2)


class QueryRequest(BaseModel):
    question: str
    mode: Mode | None = None
    filters: dict | None = None


class Source(BaseModel):
    rel_path: str
    heading_path: list[str]
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: Mode


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    llm: str
    embed: str
