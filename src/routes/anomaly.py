"""Case 10 — GET /score/{transaction_id}: Case 5's final_raw_anomaly_score for one transaction.
Fastest endpoint — no rule engine, no LLM."""
import logging

from fastapi import APIRouter, HTTPException

from src.config import settings
from src.schemas.anomaly import ScoreResponse
from src.services.agents.anomaly_scoring.agent import RISK_THRESHOLD
from src.services.anomaly.aggregation import compute_final_raw_anomaly_score
from src.services.anomaly.combined import PRIMARY_SCORE_COLUMNS, compute_all_anomaly_scores
from src.services.anomaly.normalization import normalize_scores

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["anomaly"])


@router.get("/{transaction_id}", response_model=ScoreResponse)
def get_score(transaction_id: int):
    logger.info("GET /score/%s", transaction_id)
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    all_scores = compute_all_anomaly_scores(parquet_path)
    normalized = normalize_scores(all_scores, PRIMARY_SCORE_COLUMNS)
    final_raw = compute_final_raw_anomaly_score(normalized, PRIMARY_SCORE_COLUMNS)

    row = final_raw.loc[final_raw["TransactionID"] == transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"transaction_id={transaction_id} not found")

    score = float(row.iloc[0]["final_raw_anomaly_score"])
    risk_level = "elevated" if score >= RISK_THRESHOLD else "low"
    return ScoreResponse(transaction_id=transaction_id, final_raw_anomaly_score=score, risk_level=risk_level)
