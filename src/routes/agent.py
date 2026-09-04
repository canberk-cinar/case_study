"""Case 10 — GET /agent/{transaction_id}: Case 9's run_agentic_analysis() — the full multi-agent
orchestration (feature engineering -> anomaly scoring -> rule engine -> conditional RAG-grounded
explanation) as one endpoint. RuleEngine/RAGPipeline now come from the DI container, same as
/rules/evaluate, /explain, and /rag/query — this route no longer builds its own RAGContainer
internally, closing the one inconsistency where /agent was the only endpoint not wired through
ApiContainer.
"""
import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from src.container import ApiContainer
from src.schemas.agent import AgentAnalysisResponse
from src.agents.supervisor.agent import run_agentic_analysis
from src.services.rag.pipeline import RAGPipeline
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/{transaction_id}", response_model=AgentAnalysisResponse)
@inject
def agent_analysis(
    transaction_id: int,
    rule_engine: RuleEngine = Depends(Provide[ApiContainer.rule_engine_container.rule_engine]),
    rag_pipeline: RAGPipeline = Depends(Provide[ApiContainer.rag_container.rag_pipeline]),
):
    logger.info("GET /agent/%s", transaction_id)
    try:
        report = run_agentic_analysis(transaction_id, rule_engine=rule_engine, rag_pipeline=rag_pipeline)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return AgentAnalysisResponse(**report)
