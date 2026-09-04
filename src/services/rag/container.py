"""Case 8 — dependency_injector Container: wires provider selection behind config, mirroring Case
7's RuleEngineContainer pattern. Two independent Selectors:
  - `config.embedding_provider` ("tfidf"|"ollama"|"openrouter") — defaults to "openrouter" now
    that OpenRouter's `/embeddings` endpoint (confirmed OpenAI-compatible) is available with a
    genuinely free, retrieval-quality-focused model (NVIDIA Nemotron 3 Embed 1B). TF-IDF stays
    available as the fully local, zero-network fallback (e.g. for offline notebook runs).
  - `config.llm_provider` ("ollama"|"openrouter") — defaults to "openrouter" now that the case
    study team approved it as this machine's local-Ollama stand-in (Case 9's RAM constraint);
    swapping back to "ollama" once it's installed is a config-only change, no code change.

A caller only ever asks the container for `rag_pipeline`, never constructs a provider directly.
"""
from dependency_injector import containers, providers

from src.config import settings
from src.services.rag.embeddings import OllamaEmbeddingProvider, OpenRouterEmbeddingProvider, TfidfEmbeddingProvider
from src.services.rag.llm import OllamaLLMProvider, OpenRouterLLMProvider
from src.services.rag.pipeline import RAGPipeline

DEFAULT_CONFIG = {
    "embedding_provider": "openrouter",  # "tfidf" | "ollama" | "openrouter"
    "ollama_embedding_model": "all-minilm",
    "openrouter_embedding_model": settings.EMBEDDING_MODEL,
    "llm_provider": "openrouter",  # "ollama" | "openrouter"
    "ollama_llm_model": "smollm2:360m",
    "openrouter_model": settings.LLM_MODEL,
    "openrouter_api_key": settings.LLM_API_KEY,
    "openrouter_base_url": settings.LLM_BASE_URL,
    "top_k": 3,
}


class RAGContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    embedding_provider = providers.Selector(
        config.embedding_provider,
        tfidf=providers.Singleton(TfidfEmbeddingProvider),
        ollama=providers.Singleton(OllamaEmbeddingProvider, model=config.ollama_embedding_model),
        openrouter=providers.Singleton(
            OpenRouterEmbeddingProvider,
            model=config.openrouter_embedding_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        ),
    )

    llm_provider = providers.Selector(
        config.llm_provider,
        ollama=providers.Singleton(OllamaLLMProvider, model=config.ollama_llm_model),
        openrouter=providers.Singleton(
            OpenRouterLLMProvider,
            model=config.openrouter_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        ),
    )

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        top_k=config.top_k,
    )
