"""Case 9 — anomaly_scoring agent node: wraps Case 4/5 (Adapter pattern), fully deterministic.
Runs the full statistical anomaly-scoring pipeline and extracts this transaction's
final_raw_anomaly_score, also setting `risk_level` — the input to graph.py's first routing
decision.

RISK_THRESHOLD reuses the exact constant fraud_r07 (fraud_rules.yaml) already uses — the dataset's
own top-1% cutoff on final_raw_anomaly_score — rather than inventing a second, possibly
inconsistent threshold.
"""
import logging

from src.config import settings
from src.services.agents.state import AgentState
from src.services.anomaly.aggregation import compute_final_raw_anomaly_score
from src.services.anomaly.combined import PRIMARY_SCORE_COLUMNS, compute_all_anomaly_scores
from src.services.anomaly.normalization import normalize_scores

logger = logging.getLogger(__name__)

RISK_THRESHOLD = 0.9410418735356755  # same constant as fraud_r07's top-1% threshold


def score_transaction(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("score_transaction — transaction_id=%s", transaction_id)

    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    all_scores = compute_all_anomaly_scores(parquet_path)
    normalized = normalize_scores(all_scores, PRIMARY_SCORE_COLUMNS)
    final_raw = compute_final_raw_anomaly_score(normalized, PRIMARY_SCORE_COLUMNS)

    row = final_raw.loc[final_raw["TransactionID"] == transaction_id]
    if row.empty:
        raise ValueError(f"transaction_id={transaction_id} not found in merged_transactions.parquet")

    score = float(row.iloc[0]["final_raw_anomaly_score"])
    risk_level = "elevated" if score >= RISK_THRESHOLD else "low"
    logger.info("score_transaction — final_raw_anomaly_score=%.4f risk_level=%s", score, risk_level)

    return {"anomaly_scores": row.iloc[0].to_dict(), "risk_level": risk_level}
