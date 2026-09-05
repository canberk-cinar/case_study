"""Case 9: rule_engine agent node: wraps Case 7's RuleEngine (Adapter pattern), fully
deterministic. Row loading (raw fields + Case 3 features + Case 5's final_raw_anomaly_score) is
shared with Case 10's /rules/evaluate and /explain routes via rules/data.py: not duplicated here.

Takes the RuleEngine as a parameter rather than constructing one inline: graph.py's closures
inject it from the same DI container Case 10's routes use (ApiContainer.rule_engine_container),
so every consumer of RuleEngine in the whole project (routes and this agent alike) shares one
configured instance instead of each reloading fraud_rules.yaml independently.
"""
import logging

from src.agents.schemas.state import AgentState
from src.services.rules.data import load_transaction_row
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)


def evaluate_rules(state: AgentState, engine: RuleEngine) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("evaluate_rules: transaction_id=%s", transaction_id)

    row = load_transaction_row(transaction_id)
    explanation = engine.explain(row)

    logger.info(
        "evaluate_rules: verdict_severity=%s verdict_action=%s",
        explanation["verdict_severity"], explanation["verdict_action"],
    )
    return {"rule_verdict": explanation}
