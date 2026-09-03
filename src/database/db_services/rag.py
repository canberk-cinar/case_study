"""Repository pattern — CRUD access to the RAG knowledge base, same module-level-function style as
db_services/artifact.py rather than a class wrapper, for consistency with the rest of this
project."""
import numpy as np
from sqlalchemy.orm import Session

from sqlalchemy import delete

from ..models.rag import Document, DocumentChunk


def add_document(db: Session, title: str, source: str, content: str) -> Document:
    document = Document(title=title, source=source, content=content)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def add_chunk(
    db: Session,
    document_id: int,
    chunk_index: int,
    text: str,
    embedding: np.ndarray | None = None,
    embedding_model: str | None = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding.astype(np.float32).tobytes() if embedding is not None else None,
        embedding_dim=int(embedding.shape[0]) if embedding is not None else None,
        embedding_model=embedding_model,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def list_documents(db: Session) -> list[Document]:
    return db.query(Document).order_by(Document.id).all()


def list_chunks(db: Session, embedding_model: str | None = None) -> list[DocumentChunk]:
    query = db.query(DocumentChunk)
    if embedding_model is not None:
        query = query.filter(DocumentChunk.embedding_model == embedding_model)
    return query.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index).all()


def chunk_embedding_matrix(chunks: list[DocumentChunk]) -> np.ndarray:
    """Reassembles the stored embedding BLOBs back into one (n_chunks, dim) float32 matrix, ready
    for vector_search.py's brute-force cosine similarity. All chunks passed in must share the same
    embedding_dim — callers should already have filtered by embedding_model via list_chunks()."""
    dims = {c.embedding_dim for c in chunks}
    if len(dims) > 1:
        raise ValueError(f"chunks have mismatched embedding dimensions: {dims} — filter by embedding_model first")
    return np.stack([np.frombuffer(c.embedding, dtype=np.float32) for c in chunks])


def clear_knowledge_base(db: Session) -> None:
    """Deletes every document and every chunk — used only to make re-ingestion idempotent when the
    notebook is re-run, not a general-purpose destructive operation.

    Both tables are deleted explicitly, in FK-safe order (chunks before documents), rather than
    relying on the ORM relationship's cascade="all, delete-orphan": that cascade only fires for
    session-tracked object deletion (db.delete(obj)), NOT for a bulk Query.delete()/`delete()`
    statement, which bypasses the unit-of-work entirely. Relying on it here left orphaned chunk
    rows behind after re-ingestion, which then silently re-attached to new documents that reused
    the same auto-incremented ids — a real bug caught while verifying this module.
    """
    db.execute(delete(DocumentChunk))
    db.execute(delete(Document))
    db.commit()
