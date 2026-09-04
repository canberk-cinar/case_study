"""Case 9 — feature_engineering agent node: wraps Case 3's feature-building functions (Adapter
pattern), fully deterministic — no LLM involved. Computes the dataset's temporal/entity/relational
features with the same vectorized functions Cases 3-8 already use, then returns just the requested
transaction's row, flattened into a plain dict for the shared AgentState.

Recomputing over the full dataset per agent call (rather than trying to share live DataFrames
across agent boundaries) is deliberate, not an oversight — it's what keeps each agent a genuinely
independent unit that only exchanges serializable state, the same boundary a distributed multi-
agent deployment would actually have.
"""
import logging

from src.config import settings
from src.services.agents.state import AgentState
from src.services.features.entity import build_entity_features
from src.services.features.relational import build_relational_features
from src.services.features.temporal import build_temporal_features

logger = logging.getLogger(__name__)


def fetch_features(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("fetch_features — transaction_id=%s", transaction_id)

    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    temporal = build_temporal_features(parquet_path)
    entity = build_entity_features(parquet_path)
    relational = build_relational_features(parquet_path)

    merged = temporal.merge(entity, on="TransactionID").merge(relational, on="TransactionID")
    row = merged.loc[merged["TransactionID"] == transaction_id]
    if row.empty:
        raise ValueError(f"transaction_id={transaction_id} not found in merged_transactions.parquet")

    return {"features": row.iloc[0].to_dict()}
