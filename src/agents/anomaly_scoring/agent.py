"""Case 9: anomaly_scoring agent node: wraps Case 4/5 (Adapter pattern), fully deterministic.
Looks up this transaction's precomputed final_raw_anomaly_score and sets `risk_level`: the input
to graph.py's routing decision.

Reads from the scored_transactions table (src/database/models/scoring.py) rather than
recomputing the full anomaly pipeline per call: see src/pipelines/build_scoring_artifact.py.
RISK_THRESHOLD lives in aggregation.py (the module that produces the score), reused here and by
fraud_r07 (fraud_rules.yaml) and the /score route, so there is exactly one "elevated" threshold in
the whole project.
"""
import logging

from src.agents.schemas.state import AgentState
from src.database.db_services import close, new_session
from src.database.db_services.scoring import get_scored_transaction
from src.services.anomaly.aggregation import RISK_THRESHOLD

logger = logging.getLogger(__name__)


def score_transaction(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("score_transaction: transaction_id=%s", transaction_id)

    db = new_session()
    try:
        row = get_scored_transaction(db, transaction_id)
        if row is None:
            raise ValueError(
                f"transaction_id={transaction_id} not found in scored_transactions: "
                "run `python -m src.pipelines.build_scoring_artifact` if the table is empty/stale"
            )
        score = float(row.final_raw_anomaly_score)
    finally:
        close(db)

    risk_level = "elevated" if score >= RISK_THRESHOLD else "low"
    logger.info("score_transaction: final_raw_anomaly_score=%.4f risk_level=%s", score, risk_level)

    return {
        "anomaly_scores": {"TransactionID": transaction_id, "final_raw_anomaly_score": score},
        "risk_level": risk_level,
    }
