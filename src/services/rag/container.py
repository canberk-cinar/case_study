"""Case 8: dependency_injector Container: wires provider selection behind config, mirroring Case
7's RuleEngineContainer pattern. Two independent Selectors, both driven off ONE .env switch
(`settings.LLM_PROVIDER`, "openrouter"|"ollama") so swapping the whole RAG pipeline between cloud
and local is a single-line .env change: no code change:
  - `config.embedding_provider` ("tfidf"|"ollama"|"openrouter"): `LLM_PROVIDER` only ever feeds
    it "ollama" or "openrouter"; `tfidf` remains selectable manually (e.g. `container.config.
    embedding_provider.from_value("tfidf")` in a notebook) as a fully local, zero-network fallback,
    but isn't part of the single-switch swap since there's no TF-IDF *chat* model to pair it with.
  - `config.llm_provider` ("ollama"|"openrouter"): directly mirrors `LLM_PROVIDER`.

A caller only ever asks the container for `rag_pipeline`, never constructs a provider directly.
"""
from dependency_injector import containers, providers

from src.config import settings
from src.services.rag.embeddings import OllamaEmbeddingProvider, OpenRouterEmbeddingProvider, TfidfEmbeddingProvider
from src.services.rag.llm import OllamaLLMProvider, OpenRouterLLMProvider
from src.services.rag.pipeline import RAGPipeline

DEFAULT_CONFIG = {
    "embedding_provider": settings.LLM_PROVIDER,  # "tfidf" | "ollama" | "openrouter"
    "ollama_embedding_model": "all-minilm",
    "openrouter_embedding_model": settings.EMBEDDING_MODEL,
    "llm_provider": settings.LLM_PROVIDER,  # "ollama" | "openrouter"
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
