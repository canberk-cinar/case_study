"""Case 8: Strategy pattern: EmbeddingProvider is the common interface, swappable at the
container/config level (`LLM_PROVIDER=openrouter|ollama` in .env, or `embedding_provider=tfidf`
directly on RAGContainer's config) without any caller needing to change.

TfidfEmbeddingProvider is NOT a test mock: it's a genuine, fully local, zero-extra-dependency
(scikit-learn is already a project dependency) embedding strategy, useful for building and
verifying the ENTIRE retrieval pipeline (chunking -> storage -> vector search -> ranking) with no
network dependency at all. Its one real constraint, inherent to TF-IDF rather than a shortcut
taken here: the vectorizer must be fit on the knowledge base's own documents before it can embed a
query: embed_documents() does that fit, embed_query() reuses it. Its vector space is also
specific to the fitted vocabulary, so TF-IDF and Ollama/OpenRouter embeddings are never comparable
and must never be mixed in one vector_search.py index (models.py's `embedding_model` tag on every
stored chunk exists specifically to prevent that).

OllamaEmbeddingProvider calls Ollama's local REST API directly via httpx (already a project
dependency): no new dependency needed, no `ollama` PyPI package required.

OpenRouterEmbeddingProvider calls OpenRouter's OpenAI-compatible `/embeddings` endpoint the same
httpx-only way. It's the default provider (config.py's `LLM_PROVIDER`): this machine's RAM
constraint (Case 9) rules out running Ollama locally, and OpenRouter's free embedding tier (NVIDIA
Nemotron 3 Embed 1B, chosen over LiquidAI's LFM2.5-Embedding-350M for retrieval quality: see
notebooks/case_08_rag_pipeline.ipynb's multi-concept-query finding) covers the gap. TF-IDF remains
available as a fully local fallback via `embedding_provider: tfidf` on RAGContainer's config.
"""
from abc import ABC, abstractmethod

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tag stored alongside every embedding (models.rag.DocumentChunk.embedding_model) so
        vectors from different providers/vocabularies are never silently compared."""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        ...


class TfidfEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    @property
    def name(self) -> str:
        return "tfidf"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddingProvider.embed_documents() must run first (fits the vocabulary)")
        return self._vectorizer.transform([text]).toarray().astype(np.float32)[0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "all-minilm", base_url: str = OLLAMA_DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def _embed_one(self, text: str) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_DEFAULT_BASE_URL):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"openrouter:{self.model}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda row: row["index"])
        return np.array([row["embedding"] for row in data], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def ollama_is_running(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> bool:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
