"""Case 10: GET /score/{transaction_id}: Case 5's precomputed final_raw_anomaly_score for one
transaction. Fastest endpoint: no rule engine, no LLM, and (since the scoring artifact refactor)
no per-request recomputation either: a single indexed lookup in scored_transactions.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database.db import get_db
from src.database.db_services.scoring import get_scored_transaction
from src.schemas.anomaly import ScoreResponse
from src.services.anomaly.aggregation import RISK_THRESHOLD

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["anomaly"])


@router.get("/{transaction_id}", response_model=ScoreResponse)
def get_score(transaction_id: int, db: Session = Depends(get_db)):
    logger.info("GET /score/%s", transaction_id)
    row = get_scored_transaction(db, transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"transaction_id={transaction_id} not found")

    score = float(row.final_raw_anomaly_score)
    risk_level = "elevated" if score >= RISK_THRESHOLD else "low"
    return ScoreResponse(transaction_id=transaction_id, final_raw_anomaly_score=score, risk_level=risk_level)
