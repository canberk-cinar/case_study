from pydantic import BaseModel


class RagQueryRequest(BaseModel):
    question: str
    transaction_id: int | None = None


class RagSourceChunk(BaseModel):
    document_title: str
    score: float


class RagQueryResponse(BaseModel):
    question: str
    answer: str | None
    note: str | None
    sources: list[RagSourceChunk]
