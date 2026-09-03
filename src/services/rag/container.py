"""Case 8 — dependency_injector Container: wires provider selection behind config, mirroring Case
7's RuleEngineContainer pattern. `config.embedding_provider` ("tfidf"|"ollama") selects between
TfidfEmbeddingProvider and OllamaEmbeddingProvider via a Selector provider — this is the concrete
mechanism behind the EMBEDDING_PROVIDER=tfidf|ollama swap the plan calls for; a caller only ever
asks the container for `rag_pipeline`, never constructs a provider directly.
"""
from dependency_injector import containers, providers

from src.services.rag.embeddings import OllamaEmbeddingProvider, TfidfEmbeddingProvider
from src.services.rag.llm import OllamaLLMProvider
from src.services.rag.pipeline import RAGPipeline

DEFAULT_CONFIG = {
    "embedding_provider": "tfidf",  # "tfidf" | "ollama" — tfidf until Ollama is installed
    "ollama_embedding_model": "all-minilm",
    "ollama_llm_model": "smollm2:360m",
    "top_k": 3,
}


class RAGContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    embedding_provider = providers.Selector(
        config.embedding_provider,
        tfidf=providers.Singleton(TfidfEmbeddingProvider),
        ollama=providers.Singleton(OllamaEmbeddingProvider, model=config.ollama_embedding_model),
    )

    llm_provider = providers.Singleton(OllamaLLMProvider, model=config.ollama_llm_model)

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        top_k=config.top_k,
    )
