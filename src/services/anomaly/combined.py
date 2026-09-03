"""Case 4 — assembling all four anomaly layers into one table, scores kept explicitly separate.

The case brief's own instruction: "farklı anomali skorlarını ayrı ayrı üretiniz" (produce the
different anomaly scores separately) — not one master score. Collapsing four independent
perspectives into a single number would defeat the point of building them independently in the
first place (the case brief's own "tek modele bağlı kalmadan" principle) — a downstream consumer
(a human reviewer, a rule engine, a supervised model trained later) should see all four views and
decide for itself how to weigh them, rather than this pipeline making that call.

This module does no new scoring — it's pure assembly: each function call below is a thin call
into the layer's own module (column_anomaly.py, multivariate_anomaly.py, entity_anomaly.py,
temporal_anomaly.py), merged on TransactionID into one wide table. No score here has ever looked
at isFraud, and combining them here doesn't change that.
"""
from pathlib import Path

import pandas as pd

from src.services.anomaly.column_anomaly import compute_column_anomaly_scores
from src.services.anomaly.entity_anomaly import compute_entity_anomaly_scores
from src.services.anomaly.multivariate_anomaly import compute_multivariate_anomaly_scores
from src.services.anomaly.temporal_anomaly import compute_temporal_anomaly_scores

# The five primary scores a downstream consumer most likely wants at a glance — column has one,
# multivariate has two (Mahalanobis and Isolation Forest are two independent methods, not
# collapsed into one), entity and temporal have one each. Everything else each layer returns
# (top_column, history_depth, block_count, ...) is supporting/explainability detail, still present
# in the full table but not "primary."
PRIMARY_SCORE_COLUMNS = [
    "column_anomaly_mean_abs_zscore",
    "multivariate_mahalanobis_distance",
    "multivariate_isolation_forest_score",
    "entity_anomaly_score",
    "temporal_anomaly_score",
]


def compute_all_anomaly_scores(parquet_path: Path) -> pd.DataFrame:
    column = compute_column_anomaly_scores(parquet_path)
    multivariate = compute_multivariate_anomaly_scores(parquet_path)
    entity = compute_entity_anomaly_scores(parquet_path)
    temporal = compute_temporal_anomaly_scores(parquet_path)

    df = column.merge(multivariate, on="TransactionID")
    df = df.merge(entity, on="TransactionID")
    df = df.merge(temporal, on="TransactionID")
    return df
