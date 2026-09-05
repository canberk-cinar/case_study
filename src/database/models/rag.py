from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import DateTime

from ..db import Base


class Document(Base):
    """A single knowledge-base document (a fraud/anomaly policy write-up): the unit a user adds
    or removes, source of truth for its chunks. Small, relational, CRUD-shaped data, which is why
    this knowledge base lives in SQLite rather than Parquet (see src/database/db.py's REPO_ROOT-
    anchored sqlite path)."""

    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)  # e.g. "case_06_business_context" or "experimental"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """One retrievable unit: a document is split into one or more chunks (chunking.py), each with
    its own embedding. `embedding` is stored as raw bytes via numpy.ndarray.tobytes() (float32);
    `embedding_dim` and `embedding_model` are kept alongside so a stored embedding can always be
    reconstructed correctly and never silently mixed with vectors from a different model/dimension
    (TF-IDF vectors and Ollama's all-minilm vectors are NOT the same space)."""

    __tablename__ = "rag_document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("rag_documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    embedding_model = Column(String, nullable=True)

    document = relationship("Document", back_populates="chunks")
