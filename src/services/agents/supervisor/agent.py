"""Case 9 — Facade: run_agentic_analysis(transaction_id) is the single entry point a caller (this
case's notebook) uses — hides the whole multi-agent graph behind one call, the same role as the
reference project's supervisor/agent.py::run_supervisor.
"""
import logging

from src.services.agents.graph import build_graph
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def run_agentic_analysis(transaction_id: int) -> dict:
    logger.info("run_agentic_analysis — transaction_id=%s", transaction_id)

    graph = build_graph()
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
    logger.info("run_agentic_analysis — done, risk_level=%s", report["risk_level"])
    return report
