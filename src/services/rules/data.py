"""Case 7/10 — assembles the one row of data RuleEngine's conditions need (raw transaction fields
+ Case 3 features + Case 5's final_raw_anomaly_score) for a single transaction_id. Factored out of
Case 9's rule_engine agent so the API layer (Case 10) can reuse the exact same assembly instead of
duplicating it.
"""
import pandas as pd
import pyarrow.parquet as pq

from src.config import settings
from src.services.anomaly.aggregation import compute_final_raw_anomaly_score
from src.services.anomaly.combined import PRIMARY_SCORE_COLUMNS, compute_all_anomaly_scores
from src.services.anomaly.normalization import normalize_scores
from src.services.features.entity import build_entity_features
from src.services.features.relational import build_relational_features
from src.services.features.temporal import build_temporal_features


def load_transaction_row(transaction_id: int) -> pd.Series:
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

    return row.iloc[0]
