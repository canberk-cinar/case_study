"""Case 9 — one place every agent gets its chat model from, same shape as the user's own reference
project (dashboard/backend/src/agents/shared/llm.py): ChatOpenAI pointed at a configurable
base_url, not a provider-specific client. Defaults to Ollama's OpenAI-compatible local endpoint
(the brief's local-first requirement for this case); repointing to OpenRouter or another
OpenAI-compatible endpoint later is a settings/env change only, no code change — directly useful
if that becomes the answer to the pending question about local hardware constraints.
"""
from langchain_openai import ChatOpenAI

from src.config import settings


def get_llm(temperature: float = 0.2, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
