"""Case 8 — chunking: splits a document's text into retrievable units.

Kept deliberately simple for this knowledge base's actual shape (short, single-topic policy write-
ups, a few sentences to a couple of paragraphs each — not long multi-topic documents) — a
paragraph-based split, merging short paragraphs up to a target word count so a chunk is neither a
single sentence-fragment nor a whole multi-topic document. A more sophisticated splitter (sentence-
boundary-aware, overlapping windows) would be solving a problem this knowledge base doesn't have.
"""
import re

TARGET_CHUNK_WORDS = 120


def chunk_text(text: str, target_words: int = TARGET_CHUNK_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        if current and current_words + paragraph_words > target_words:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks
