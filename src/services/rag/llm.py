"""Case 8 — Strategy pattern, mirroring embeddings.py: LLMProvider is the swappable generation
interface. OllamaLLMProvider is the one real implementation — calls Ollama's local
`/api/generate` REST endpoint directly via httpx, no new dependency. An OpenRouter-backed
LLMProvider could implement this same ABC later (documented extension point only — no API key
available, and running fully local is this case's whole point, so it isn't built here).
"""
from abc import ABC, abstractmethod

import httpx

from src.services.rag.embeddings import OLLAMA_DEFAULT_BASE_URL


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
