"""Case 8 — Builder pattern: PromptBuilder assembles the final LLM prompt from optional parts
(system instructions, retrieved context chunks, the question) added independently and in any
order, then finalized with .build() — this IS "LLM context injection": the retrieved chunks'
actual text gets woven into the prompt the model will see, with explicit source numbering so a
generated answer can point back to which policy document supported it.
"""
from dataclasses import dataclass, field

from src.services.rag.models import RetrievedChunk

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a policy assistant for a fraud/anomaly detection system. Answer using ONLY the "
    "source texts given below. Do not invent anything not present in the sources; if the sources "
    "don't cover the question, say so explicitly. Cite which source(s) you relied on using "
    "numbers like [1], [2]."
)


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
