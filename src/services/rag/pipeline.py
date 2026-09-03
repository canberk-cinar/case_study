"""Case 8 — RAGPipeline: Facade over chunking, embedding, storage, vector search, prompt-building,
and generation — three methods hide the whole multi-step flow:
  - ingest(db, documents): chunk -> embed -> persist.
  - answer(db, question): retrieve -> inject context into a prompt -> generate. Degrades
    gracefully (never raises) if the LLM provider is unreachable — returns the retrieval +
    constructed prompt with `answer=None` and an explanatory `note`, so retrieval and context
    injection stay independently verifiable without Ollama running.
  - answer_for_flagged_transaction(db, explanation): the Case 7 bridge — turns a
    RuleEngine.explain()-shaped dict directly into a grounded natural-language question, which is
    this case's actual "make anomaly results more explainable" deliverable, not a generic QA demo.

One constraint worth stating plainly: the SAME RAGPipeline instance (and therefore the same
embedding_provider instance) must be used for both ingest() and answer() in one run — TF-IDF's
vectorizer only knows its vocabulary because embed_documents() fit it during ingest(); a fresh,
unfitted provider can't embed a query. Ollama's provider has no such constraint (it's a fixed
pretrained model), but the API here is intentionally uniform across both.
"""
import httpx
import numpy as np
from sqlalchemy.orm import Session

from src.database.db_services.rag import add_chunk, add_document, chunk_embedding_matrix, clear_knowledge_base, list_chunks
from src.services.rag.chunking import chunk_text
from src.services.rag.embeddings import EmbeddingProvider
from src.services.rag.llm import LLMProvider
from src.services.rag.models import Chunk, RetrievedChunk
from src.services.rag.prompt import PromptBuilder
from src.services.rag.vector_search import VectorSearchIndex


class RAGPipeline:
    def __init__(self, embedding_provider: EmbeddingProvider, llm_provider: LLMProvider, top_k: int = 3):
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.top_k = top_k

    def ingest(self, db: Session, documents: list[tuple[str, str, str]]) -> int:
        """documents: (title, source, content) tuples. Idempotent — clears any previously ingested
        knowledge base first, so re-running a notebook cell doesn't duplicate documents."""
        clear_knowledge_base(db)

        pending: list[tuple[int, list[str]]] = []
        for title, source, content in documents:
            document = add_document(db, title=title, source=source, content=content)
            pending.append((document.id, chunk_text(content)))

        all_texts = [text for _, chunks in pending for text in chunks]
        if not all_texts:
            return 0
        embeddings = self.embedding_provider.embed_documents(all_texts)

        cursor = 0
        for document_id, chunks in pending:
            for chunk_index, text in enumerate(chunks):
                add_chunk(
                    db,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                    embedding=embeddings[cursor],
                    embedding_model=self.embedding_provider.name,
                )
                cursor += 1
        return len(all_texts)

    def _load_index(self, db: Session) -> VectorSearchIndex:
        orm_chunks = list_chunks(db, embedding_model=self.embedding_provider.name)
        if not orm_chunks:
            return VectorSearchIndex([])
        matrix = chunk_embedding_matrix(orm_chunks)
        chunks = [
            Chunk(
                id=c.id,
                document_id=c.document_id,
                document_title=c.document.title,
                document_source=c.document.source,
                text=c.text,
                embedding=matrix[i],
            )
            for i, c in enumerate(orm_chunks)
        ]
        return VectorSearchIndex(chunks)

    def retrieve(self, db: Session, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        index = self._load_index(db)
        query_embedding = self.embedding_provider.embed_query(query)
        return index.search(query_embedding, top_k=top_k or self.top_k)

    def answer(self, db: Session, question: str, top_k: int | None = None) -> dict:
        retrieved = self.retrieve(db, question, top_k=top_k)
        prompt = PromptBuilder().with_context(retrieved).with_question(question).build()

        try:
            answer_text = self.llm_provider.generate(prompt)
            note = None
        except httpx.HTTPError as exc:
            answer_text = None
            note = f"LLM generation failed ({self.llm_provider.name}): {exc} — showing retrieval + prompt only."

        return {"question": question, "sources": retrieved, "prompt": prompt, "answer": answer_text, "note": note}

    def answer_for_flagged_transaction(self, db: Session, explanation: dict, top_k: int | None = None) -> dict:
        """`explanation`: the dict shape produced by Case 7's RuleEngine.explain() /
        explain_batch() — src/services/rules/engine.py."""
        rule_names = ", ".join(f["rule_name"] for f in explanation["fired_rules"])
        question = (
            f"A transaction was flagged by these rules: {rule_names}. "
            f"Final verdict: {explanation['verdict_severity']} / {explanation['verdict_action']}. "
            "Explain why this transaction was considered risky, based on the relevant policies."
        )
        return self.answer(db, question, top_k=top_k)
