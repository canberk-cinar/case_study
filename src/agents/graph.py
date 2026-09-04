"""Case 9 — builds the LangGraph StateGraph: feature_engineering -> anomaly_scoring -> rule_engine
-> [conditional] -> policy_explanation -> END.

build_graph(rule_engine, rag_pipeline) takes both dependencies as parameters and closes over them
in the two node functions that need them — the same closure-injection shape as the user's own
reference project (dashboard/backend/src/agents/graph.py::build_graph(db)). Both come from
ApiContainer (Case 10's DI container) via supervisor/agent.py, so /agent is no longer the one
endpoint that builds its own container internally — every route now shares the same configured
RuleEngine/RAGPipeline instances.

Both anomaly_scoring and rule_engine always run — Case 7's own notebook already established that
the statistical anomaly score and the rule engine's verdict catch LARGELY DISJOINT sets of risky
transactions (see case_07_rule_engine.ipynb section 10: "sadece iş kuralları" vs "sadece AI" are
mostly different transactions). An earlier version of this graph gated rule_engine behind a high
anomaly score, which would have skipped the rule engine entirely for exactly the transactions its
own examples fire on (e.g. fraud_r01's CRITICAL example transaction has a below-threshold raw
anomaly score) — caught while verifying this module, fixed by running both unconditionally
(both are cheap and deterministic) and escalating to policy_explanation if EITHER signal is high.

The one conditional edge is a plain Python function on state, not an LLM call — same philosophy
as the reference project: use an LLM only where genuine ambiguity/NLU is involved. The one place
an LLM genuinely earns its keep here is policy_explanation, turning a structured verdict into a
grounded natural-language explanation — not the routing decision itself.
"""
import logging

from langgraph.graph import END, StateGraph

from src.agents.anomaly_scoring.agent import score_transaction
from src.agents.feature_engineering.agent import fetch_features
from src.agents.policy_explanation.agent import explain_verdict
from src.agents.rule_engine.agent import evaluate_rules
from src.agents.schemas.state import AgentState
from src.services.rag.pipeline import RAGPipeline
from src.services.rules.engine import RuleEngine

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


def build_graph(rule_engine: RuleEngine, rag_pipeline: RAGPipeline):
    def _rule_engine_node(state: AgentState) -> dict:
        return evaluate_rules(state, rule_engine)

    def _policy_explanation_node(state: AgentState) -> dict:
        return explain_verdict(state, rag_pipeline)

    g = StateGraph(AgentState)
    g.add_node("feature_engineering", fetch_features)
    g.add_node("anomaly_scoring", score_transaction)
    g.add_node("rule_engine", _rule_engine_node)
    g.add_node("policy_explanation", _policy_explanation_node)

    g.set_entry_point("feature_engineering")
    g.add_edge("feature_engineering", "anomaly_scoring")
    g.add_edge("anomaly_scoring", "rule_engine")
    g.add_conditional_edges("rule_engine", _route_after_rule_engine, {
        "policy_explanation": "policy_explanation",
        "end_structured": END,
    })
    g.add_edge("policy_explanation", END)

    return g.compile()
