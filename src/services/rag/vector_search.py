"""Case 8: brute-force cosine similarity search over chunk embeddings held in memory. No ANN
index (FAISS/HNSW): at this knowledge base's scale (a handful of documents, a few dozen chunks
at most) an exact O(n) scan is both simpler and, in practice, as fast as building an index would
be; this was decided against premature infrastructure for a dataset this small.
"""
import numpy as np

from src.services.rag.domain.models import Chunk, RetrievedChunk


class VectorSearchIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        if chunks:
            matrix = np.stack([c.embedding for c in chunks])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # guard a theoretically-possible all-zero embedding
            self._normalized = matrix / norms
        else:
            self._normalized = np.empty((0, 0), dtype=np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        scores = self._normalized @ query_norm
        top_indices = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]
