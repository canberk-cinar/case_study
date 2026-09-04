"""Case 9 — shared state that flows through the LangGraph graph, same role as the reference
project's state.py: every node reads what earlier nodes wrote and writes its own piece — this IS
the agent-to-agent communication channel (a Mediator-style architecture, not direct peer-to-peer
messages between agents).
"""
from typing import Optional, TypedDict


class AgentState(TypedDict):
    transaction_id:      int
    features:             Optional[dict]   # feature_engineering's output (temporal/entity/relational, flattened)
    anomaly_scores:        Optional[dict]   # anomaly_scoring's output (the 5 layer scores + final_raw_anomaly_score)
    risk_level:              str            # "low" | "elevated" — set by anomaly_scoring, drives the first routing decision
    rule_verdict:              Optional[dict]  # rule_engine's output — shape of RuleEngine.explain()
    policy_explanation:          Optional[dict]  # policy_explanation's output — shape of RAGPipeline.answer(...)
    report:                        Optional[dict]  # supervisor's final aggregated report
