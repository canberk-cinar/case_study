"""Case 10 follow-up: builds the precomputed scoring artifact (scored_transactions table) that
the API and rule engine read from instead of recomputing every feature and anomaly layer on every
request. Run whenever scoring/feature logic changes:

    python -m src.pipelines.build_scoring_artifact

Reuses the exact same assembly Case 7's rule engine and Case 10's routes already relied on
(previously duplicated inline in services/rules/data.py::load_transaction_row): this script is
the one place that runs it over the FULL dataset once and persists the result; the per-request
read path (data.py) becomes a single indexed lookup.
"""
import logging

import typer

from src.config import settings
from src.database.db import run_migrations
from src.database.db_services import close, new_session
from src.database.db_services.scoring import replace_scored_transactions
from src.services.anomaly.aggregation import compute_final_raw_anomaly_score
from src.services.anomaly.combined import PRIMARY_SCORE_COLUMNS, compute_all_anomaly_scores
from src.services.anomaly.normalization import normalize_scores
from src.services.features.entity import build_entity_features
from src.services.features.relational import build_relational_features
from src.services.features.temporal import build_temporal_features

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()

SCORING_COLUMNS = [
    "TransactionID",
    "TransactionAmt", "addr2", "DeviceInfo", "dist1",
    "hour_of_day", "is_low_volume_hour", "is_weekend_proxy",
    "user_transaction_count_so_far", "user_amount_zscore", "user_seconds_since_last_transaction",
    "is_new_addr1_for_card", "is_new_device_for_card",
    "final_raw_anomaly_score",
]


@app.command()
def build():
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    logger.info("reading raw columns from %s", parquet_path)
    raw = pq.ParquetFile(parquet_path).read(
        columns=["TransactionID", "TransactionAmt", "addr2", "DeviceInfo", "dist1"]
    ).to_pandas()

    logger.info("building Case 3 features (temporal/entity/relational)")
    temporal = build_temporal_features(parquet_path)
    entity = build_entity_features(parquet_path)
    relational = build_relational_features(parquet_path)

    logger.info("scoring all 4 anomaly layers + aggregation (this is the slow part, ~seconds)")
    all_scores = compute_all_anomaly_scores(parquet_path)
    normalized = normalize_scores(all_scores, PRIMARY_SCORE_COLUMNS)
    final_raw = compute_final_raw_anomaly_score(normalized, PRIMARY_SCORE_COLUMNS)

    df = raw.merge(temporal, on="TransactionID") \
        .merge(entity, on="TransactionID") \
        .merge(relational, on="TransactionID") \
        .merge(final_raw, on="TransactionID")
    df = df[SCORING_COLUMNS]

    logger.info("writing %d rows to scored_transactions", len(df))
    run_migrations()
    db = new_session()
    try:
        n = replace_scored_transactions(db, df)
    finally:
        close(db)

    logger.info("done: %d rows in scored_transactions", n)


if __name__ == "__main__":
    app()
