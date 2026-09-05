"""Case 9: Facade: run_agentic_analysis(transaction_id, rule_engine, rag_pipeline) is the single
entry point a caller uses: hides the whole multi-agent graph behind one call, the same role as
the reference project's supervisor/agent.py::run_supervisor.

rule_engine/rag_pipeline default to building a fresh ApiContainer's instances when omitted (e.g. a
notebook calling this directly, outside a running API). Case 10's /agent route always passes its
own DI-container-sourced instances explicitly instead of relying on this default: every request-
path caller then shares the same configured RuleEngine/RAGPipeline as the rest of the API, closing
the one place (this function) that used to build its own container internally.
"""
import logging

from src.agents.graph import build_graph
from src.agents.schemas.state import AgentState
from src.services.rag.pipeline import RAGPipeline
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)


def run_agentic_analysis(
    transaction_id: int,
    rule_engine: RuleEngine | None = None,
    rag_pipeline: RAGPipeline | None = None,
) -> dict:
    logger.info("run_agentic_analysis: transaction_id=%s", transaction_id)

    if rule_engine is None or rag_pipeline is None:
        from src.container import build_container  # local import avoids a module-load-time cycle

        container = build_container()
        rule_engine = rule_engine or container.rule_engine_container.rule_engine()
        rag_pipeline = rag_pipeline or container.rag_container.rag_pipeline()

    graph = build_graph(rule_engine, rag_pipeline)
    initial_state: AgentState = {
        "transaction_id": transaction_id,
        "features": None,
        "anomaly_scores": None,
        "risk_level": "",
        "rule_verdict": None,
        "policy_explanation": None,
        "report": None,
    }
    result = graph.invoke(initial_state, {"recursion_limit": 10})

    report = {
        "transaction_id": transaction_id,
        "risk_level": result["risk_level"],
        "final_raw_anomaly_score": (
            result["anomaly_scores"]["final_raw_anomaly_score"] if result.get("anomaly_scores") else None
        ),
        "rule_verdict": result.get("rule_verdict"),
        "policy_explanation": result.get("policy_explanation"),
    }
    logger.info("run_agentic_analysis: done, risk_level=%s", report["risk_level"])
    return report
