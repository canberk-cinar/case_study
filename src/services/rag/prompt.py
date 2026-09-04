"""Case 8 — Builder pattern: PromptBuilder assembles the final LLM prompt from optional parts
(system instructions, retrieved context chunks, the question) added independently and in any
order, then finalized with .build() — this IS "LLM context injection": the retrieved chunks'
actual text gets woven into the prompt the model will see, with explicit source numbering so a
generated answer can point back to which policy document supported it.

The default system instructions are a static asset (static/system_prompt.json), not a hardcoded
Python string — this is RAG's own general-purpose prompt (any caller of RAGPipeline uses it, not
just Case 9's policy_explanation agent), so it lives here rather than under any one agent's folder.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.services.rag.domain.models import RetrievedChunk

_STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_SYSTEM_INSTRUCTIONS = json.loads((_STATIC_DIR / "system_prompt.json").read_text())["system_instructions"]


@dataclass
class PromptBuilder:
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS
    context_chunks: list[RetrievedChunk] = field(default_factory=list)
    question: str | None = None

    def with_system_instructions(self, text: str) -> "PromptBuilder":
        self.system_instructions = text
        return self

    def with_context(self, retrieved: list[RetrievedChunk]) -> "PromptBuilder":
        self.context_chunks = retrieved
        return self

    def with_question(self, question: str) -> "PromptBuilder":
        self.question = question
        return self

    def build(self) -> str:
        parts = [self.system_instructions]

        if self.context_chunks:
            sources = "\n\n".join(
                f"[{i}] {rc.chunk.document_title} (similarity score: {rc.score:.3f})\n{rc.chunk.text}"
                for i, rc in enumerate(self.context_chunks, start=1)
            )
            parts.append(f"Sources:\n\n{sources}")
        else:
            parts.append("Sources: (no relevant document found)")

        if self.question:
            parts.append(f"Question: {self.question}")

        return "\n\n".join(parts)
