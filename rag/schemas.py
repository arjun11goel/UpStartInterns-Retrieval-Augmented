"""Typed data objects used by the RAG pipeline."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievedPassage:
    text: str
    source: str
    page: Optional[int]
    similarity: float

@dataclass
class AnswerResult:
    answer: str
    passages: list[RetrievedPassage]
    used_fallback: bool
