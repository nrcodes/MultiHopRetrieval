from typing import Literal
from pydantic import BaseModel, Field

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    doc_id: str
    sent_id: int
    sentence: str
    char_span: tuple[int, int]

class Edge(BaseModel):
    from_node: str
    to_node: str
    predicate: str
    evidence: Triple
    direction: Literal["fwd", "rev"]

class Path(BaseModel):
    edges: list[Edge]
    score: float

class Answer(BaseModel):
    status: Literal["ok", "insufficient_evidence", "ambiguous"]
    query: str
    answer: str | None = None
    confidence: float = 0.0
    hops: int = 0
    chain: list[Edge] = Field(default_factory=list)
    reason: str | None = None