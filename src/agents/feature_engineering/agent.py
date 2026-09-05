"""Case 9: feature_engineering agent node: wraps Case 3's feature-building functions (Adapter
pattern), fully deterministic: no LLM involved. Reads this transaction's precomputed temporal/
entity/relational features from the scored_transactions table (src/database/models/scoring.py)
rather than recomputing Case 3's functions over the full dataset per call: see
src/pipelines/build_scoring_artifact.py. `state["features"]` is informational output only (no
downstream node reads it back), so this reads the columns fraud_rules.yaml's conditions need, not
Case 3's full feature set.

Reading from a shared store rather than recomputing is a deliberate boundary choice, not an
oversight: each agent still only exchanges serializable state, the same boundary a distributed
multi-agent deployment would actually have: it's just backed by an indexed table instead of a
full recomputation now.
"""
import logging

from src.agents.schemas.state import AgentState
from src.database.db_services import close, new_session
from src.database.db_services.scoring import get_scored_transaction

logger = logging.getLogger(__name__)

_FEATURE_COLUMNS = [
    "hour_of_day", "is_low_volume_hour", "is_weekend_proxy",
    "user_transaction_count_so_far", "user_amount_zscore", "user_seconds_since_last_transaction",
    "is_new_addr1_for_card", "is_new_device_for_card",
]


def fetch_features(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("fetch_features: transaction_id=%s", transaction_id)

    db = new_session()
    try:
        row = get_scored_transaction(db, transaction_id)
        if row is None:
            raise ValueError(
                f"transaction_id={transaction_id} not found in scored_transactions: "
                "run `python -m src.pipelines.build_scoring_artifact` if the table is empty/stale"
            )
        features = {col: getattr(row, col) for col in _FEATURE_COLUMNS}
    finally:
        close(db)

    return {"features": features}
