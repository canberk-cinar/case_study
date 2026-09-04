"""Case 9 — builds the LangGraph StateGraph: feature_engineering -> anomaly_scoring -> rule_engine
-> [conditional] -> policy_explanation -> END.

Both anomaly_scoring and rule_engine always run — Case 7's own notebook already established that
the statistical anomaly score and the rule engine's verdict catch LARGELY DISJOINT sets of risky
transactions (see case_07_rule_engine.ipynb section 10: "sadece iş kuralları" vs "sadece AI" are
mostly different transactions). An earlier version of this graph gated rule_engine behind a high
anomaly score, which would have skipped the rule engine entirely for exactly the transactions its
own examples fire on (e.g. fraud_r01's CRITICAL example transaction has a below-threshold raw
anomaly score) — caught while verifying this module, fixed by running both unconditionally
(both are cheap and deterministic) and escalating to policy_explanation if EITHER signal is high.

The one conditional edge is a plain Python function on state, not an LLM call — same philosophy
as the reference project (dashboard/backend/src/agents/graph.py): use an LLM only where genuine
ambiguity/NLU is involved. The one place an LLM genuinely earns its keep here is
policy_explanation, turning a structured verdict into a grounded natural-language explanation —
not the routing decision itself.
"""
import logging

from langgraph.graph import END, StateGraph

from src.services.agents.anomaly_scoring.agent import score_transaction
from src.services.agents.feature_engineering.agent import fetch_features
from src.services.agents.policy_explanation.agent import explain_verdict
from src.services.agents.rule_engine.agent import evaluate_rules
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)

ESCALATION_SEVERITIES = {"HIGH", "CRITICAL"}


def _route_after_rule_engine(state: AgentState) -> str:
    severity = state["rule_verdict"].get("verdict_severity") if state.get("rule_verdict") else None
    anomaly_flagged = state["risk_level"] == "elevated"
    rule_flagged = severity in ESCALATION_SEVERITIES
    decision = "policy_explanation" if (anomaly_flagged or rule_flagged) else "end_structured"
    logger.info(
        "route_after_rule_engine — risk_level=%s verdict_severity=%s -> %s",
        state["risk_level"], severity, decision,
    )
    return decision


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("feature_engineering", fetch_features)
    g.add_node("anomaly_scoring", score_transaction)
    g.add_node("rule_engine", evaluate_rules)
    g.add_node("policy_explanation", explain_verdict)

    g.set_entry_point("feature_engineering")
    g.add_edge("feature_engineering", "anomaly_scoring")
    g.add_edge("anomaly_scoring", "rule_engine")
    g.add_conditional_edges("rule_engine", _route_after_rule_engine, {
        "policy_explanation": "policy_explanation",
        "end_structured": END,
    })
    g.add_edge("policy_explanation", END)

    return g.compile()
