"""Case 10 — POST /rag/query: Case 8's RAGPipeline, free-form question against the policy
knowledge base. Optional `transaction_id` grounds the question in that transaction's rule verdict
instead of a bare question — reuses Case 9's own question-framing prompt
(agents/policy_explanation/static/question_template.json via build_flagged_transaction_question)
rather than duplicating it here. RAGPipeline comes from the DI container — swapping embedding/LLM
provider (container.rag_container.config) needs no route code change.

Re-ingests the (tiny, 8-document) knowledge base on every request rather than caching it across
requests — simpler, and cheap at this scale (matches Case 8/9's own per-call ingest pattern);
worth reconsidering only if the knowledge base grows much larger.
"""
import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.agents.policy_explanation.agent import build_flagged_transaction_question
from src.config import REPO_ROOT
from src.container import ApiContainer
from src.database.db import get_db
from src.schemas.rag import RagQueryRequest, RagQueryResponse, RagSourceChunk
from src.services.rag.pipeline import RAGPipeline
from src.services.rules.data import load_transaction_row
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


def _load_knowledge_base_documents() -> list[tuple[str, str, str]]:
    kb_dir = REPO_ROOT / "data" / "knowledge_base"
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        source = "experimental" if "experimental" in path.stem else "case_06_07"
        documents.append((title, source, text))
    return documents


@router.post("/query", response_model=RagQueryResponse)
@inject
def rag_query(
    body: RagQueryRequest,
    db: Session = Depends(get_db),
    pipeline: RAGPipeline = Depends(Provide[ApiContainer.rag_container.rag_pipeline]),
    engine: RuleEngine = Depends(Provide[ApiContainer.rule_engine_container.rule_engine]),
):
    logger.info("POST /rag/query — question=%r transaction_id=%s", body.question[:80], body.transaction_id)
    pipeline.ingest(db, _load_knowledge_base_documents())

    if body.transaction_id is not None:
        try:
            row = load_transaction_row(body.transaction_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        explanation = engine.explain(row)
        question = build_flagged_transaction_question(explanation)
        result = pipeline.answer(db, question)
    else:
        result = pipeline.answer(db, body.question)

    return RagQueryResponse(
        question=result["question"],
        answer=result["answer"],
        note=result["note"],
        sources=[RagSourceChunk(document_title=rc.chunk.document_title, score=rc.score) for rc in result["sources"]],
    )
