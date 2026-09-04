"""Case 9 — rule_engine agent node: wraps Case 7's RuleEngine (Adapter pattern), fully
deterministic. Row assembly (raw fields + Case 3 features + Case 5's final_raw_anomaly_score) is
shared with Case 10's /rules/evaluate and /explain routes via rules/data.py — not duplicated here.
"""
import logging

from src.config import REPO_ROOT
from src.services.agents.state import AgentState
from src.services.rules.data import load_transaction_row
from src.services.rules.engine import RuleEngine
from src.services.rules.loader import RuleLoader
from src.services.rules.resolution import build_default_resolution_chain

logger = logging.getLogger(__name__)

RULES_PATH = REPO_ROOT / "src" / "services" / "rules" / "definitions" / "fraud_rules.yaml"


def evaluate_rules(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("evaluate_rules — transaction_id=%s", transaction_id)

    row = load_transaction_row(transaction_id)

    rules = RuleLoader().load(RULES_PATH)
    engine = RuleEngine(rules, build_default_resolution_chain())
    explanation = engine.explain(row)

    logger.info(
        "evaluate_rules — verdict_severity=%s verdict_action=%s",
        explanation["verdict_severity"], explanation["verdict_action"],
    )
    return {"rule_verdict": explanation}
