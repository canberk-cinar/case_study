"""Case 8 — lightweight in-memory domain objects for the RAG service layer.

Deliberately NOT the SQLAlchemy ORM objects from src/database/models/rag.py: a chunk pulled out of
a DB session and handed to vector_search.py / prompt.py needs to keep working after that session
closes (a detached ORM instance can raise on attribute access), and the service layer shouldn't
need to know anything about SQLAlchemy to do retrieval or prompt-building. db_services/rag.py's
Repository functions are the only place ORM objects and these dataclasses meet.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    document_title: str
    document_source: str
    text: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity to the query, in [-1, 1]
