"""Case 7/10 — loads the one row of data RuleEngine's conditions need (raw transaction fields +
Case 3 features + Case 5's final_raw_anomaly_score) for a single transaction_id.

Reads from the precomputed `scored_transactions` table (src/database/models/scoring.py), not
recomputed from merged_transactions.parquet on every call — a single point lookup by TransactionID
used to trigger a full recomputation of every feature and anomaly layer (~27s); SQLite's primary-
key index makes this a few milliseconds instead. Rebuild the table with
`python -m src.pipelines.build_scoring_artifact` whenever scoring/feature logic changes.
"""
import pandas as pd

from src.database.db_services import close, new_session
from src.database.db_services.scoring import get_scored_transaction
from src.database.models.scoring import ScoredTransaction

_COLUMNS = [c.name for c in ScoredTransaction.__table__.columns]


def load_transaction_row(transaction_id: int) -> pd.Series:
    db = new_session()
    try:
        row = get_scored_transaction(db, transaction_id)
        if row is None:
            raise ValueError(
                f"transaction_id={transaction_id} not found in scored_transactions — "
                "run `python -m src.pipelines.build_scoring_artifact` if the table is empty/stale"
            )
        return pd.Series({col: getattr(row, col) for col in _COLUMNS})
    finally:
        close(db)
