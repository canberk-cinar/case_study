"""Case 9 — rule_engine agent node: wraps Case 7's RuleEngine (Adapter pattern), fully
deterministic. Reassembles the columns fraud_rules.yaml's conditions need (raw transaction fields
+ Case 3 features + Case 5's final_raw_anomaly_score) and calls RuleEngine.explain() for this one
transaction — the exact same explainability output Case 7's notebook already demonstrated.
"""
import logging

import pyarrow.parquet as pq

from src.config import REPO_ROOT, settings
from src.services.agents.state import AgentState
from src.services.anomaly.aggregation import compute_final_raw_anomaly_score
from src.services.anomaly.combined import PRIMARY_SCORE_COLUMNS, compute_all_anomaly_scores
from src.services.anomaly.normalization import normalize_scores
from src.services.features.entity import build_entity_features
from src.services.features.relational import build_relational_features
from src.services.features.temporal import build_temporal_features
from src.services.rules.engine import RuleEngine
from src.services.rules.loader import RuleLoader
from src.services.rules.resolution import build_default_resolution_chain

logger = logging.getLogger(__name__)

RULES_PATH = REPO_ROOT / "src" / "services" / "rules" / "definitions" / "fraud_rules.yaml"


def evaluate_rules(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    logger.info("evaluate_rules — transaction_id=%s", transaction_id)

    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    raw = pq.ParquetFile(parquet_path).read(
        columns=["TransactionID", "TransactionAmt", "addr2", "DeviceInfo", "dist1"]
    ).to_pandas()
    temporal = build_temporal_features(parquet_path)
    entity = build_entity_features(parquet_path)
    relational = build_relational_features(parquet_path)
    all_scores = compute_all_anomaly_scores(parquet_path)
    normalized = normalize_scores(all_scores, PRIMARY_SCORE_COLUMNS)
    final_raw = compute_final_raw_anomaly_score(normalized, PRIMARY_SCORE_COLUMNS)

    df = raw.merge(temporal, on="TransactionID") \
        .merge(entity, on="TransactionID") \
        .merge(relational, on="TransactionID") \
        .merge(final_raw, on="TransactionID")
    row = df.loc[df["TransactionID"] == transaction_id]
    if row.empty:
        raise ValueError(f"transaction_id={transaction_id} not found in merged_transactions.parquet")

    rules = RuleLoader().load(RULES_PATH)
    engine = RuleEngine(rules, build_default_resolution_chain())
    explanation = engine.explain(row.iloc[0])

    logger.info(
        "evaluate_rules — verdict_severity=%s verdict_action=%s",
        explanation["verdict_severity"], explanation["verdict_action"],
    )
    return {"rule_verdict": explanation}
