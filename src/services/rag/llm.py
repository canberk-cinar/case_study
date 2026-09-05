"""Case 8: Strategy pattern, mirroring embeddings.py: LLMProvider is the swappable generation
interface, selected behind config (container.py's `llm_provider` Selector): no caller needs to
know or care which one is active.

OllamaLLMProvider calls Ollama's local `/api/generate` REST endpoint directly via httpx.
OpenRouterLLMProvider calls any OpenAI-compatible `/chat/completions` endpoint the same way: same
httpx-only style, no `langchain`/`openai` SDK dependency added just for this. OpenRouter's free
tier is used as a stand-in for local Ollama on this machine's limited RAM (Case 9's RAM
constraint): the provider Strategy this file already had is exactly what made that a config
change instead of a rewrite.
"""
from abc import ABC, abstractmethod

import httpx

from src.services.rag.embeddings import OLLAMA_DEFAULT_BASE_URL

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str = "smollm2:360m", base_url: str = OLLAMA_DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class OpenRouterLLMProvider(LLMProvider):
    """Any OpenAI-compatible `/chat/completions` endpoint, not just OpenRouter specifically: the
    same shape would work unchanged against Ollama's own OpenAI-compatible endpoint or another
    OpenAI-compatible provider, by base_url/api_key/model alone."""

    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_DEFAULT_BASE_URL):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @property
    def name(self) -> str:
        return f"openrouter:{self.model}"

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
