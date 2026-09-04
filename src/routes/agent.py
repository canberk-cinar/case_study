"""Case 10 — GET /agent/{transaction_id}: Case 9's run_agentic_analysis() — the full multi-agent
orchestration (feature engineering -> anomaly scoring -> rule engine -> conditional RAG-grounded
explanation) as one endpoint. Deliberately not routed through ApiContainer directly — Case 9's
graph builds its own RAGContainer internally (policy_explanation/agent.py), and threading DI
through LangGraph's node functions would complicate the graph for no real benefit at this scale.
"""
import logging

from fastapi import APIRouter, HTTPException

from src.schemas.agent import AgentAnalysisResponse
from src.services.agents.supervisor.agent import run_agentic_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/{transaction_id}", response_model=AgentAnalysisResponse)
def agent_analysis(transaction_id: int):
    logger.info("GET /agent/%s", transaction_id)
    try:
        report = run_agentic_analysis(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return AgentAnalysisResponse(**report)
