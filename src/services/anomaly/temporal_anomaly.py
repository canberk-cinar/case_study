"""Case 4 — temporal anomaly detection: does the amount look unusual for this hour of day, and
does the LOCAL activity level around this transaction's time look like a burst.

Two components, both computed without ever looking at isFraud:
  - amount component: reused directly from Case 3's context.py (amount_zscore_within_hour) — how
    far this transaction's amount sits from the causal (so-far) average for its hour_of_day. Not
    re-derived; this layer just adopts it as one of its two signals. temporal_anomaly_hour_history
    reports how many prior transactions that hour_of_day group had already seen — the same
    "shallow history makes a causal statistic unstable" pattern column_anomaly.py's scored_count
    and entity_anomaly.py's history_depth already surfaced shows up here too (verified: the
    highest amount-component values in this dataset came from a transaction's hour_of_day group
    having only 2 prior observations, both nearly identical, collapsing prior_std toward zero).
  - burst component: a genuinely new signal here. Transactions are binned into real (day,
    hour_of_day) blocks (182 days x 24 hours = 4,368 blocks, ~182 observations per hour_of_day —
    enough for a stable mean/std, no degeneracy risk the way the sparse numeric columns had).
    Each block's transaction count is scored against the mean/std of counts sharing the same
    hour_of_day, not the whole dataset's — a burst at 3am is compared to typical 3am volume
    (Case 1 found volume varies ~15x across hours), not typical 6pm volume. Only the positive
    direction is scored (np.clip at 0): an unusually QUIET hour isn't the same kind of signal a
    fraud system cares about the way a sudden burst is, so a lull doesn't count as "anomalous"
    here. Every transaction in the same (day, hour) block shares the same burst score — this
    layer flags suspicious *time windows*, not what's distinctive about one transaction within a
    burst; that distinction is what the other three layers are for.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.services.features.context import build_context_features
from src.services.features.temporal import SECONDS_PER_DAY, SECONDS_PER_HOUR


def compute_burst_scores(parquet_path: Path) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=["TransactionID", "TransactionDT"]).to_pandas()
    df["day_of_period"] = df["TransactionDT"] // SECONDS_PER_DAY
    df["hour_of_day"] = (df["TransactionDT"] // SECONDS_PER_HOUR) % 24

    block_counts = df.groupby(["day_of_period", "hour_of_day"]).size().rename("block_count").reset_index()
    hour_stats = block_counts.groupby("hour_of_day")["block_count"].agg(hour_mean="mean", hour_std="std")
    block_counts = block_counts.merge(hour_stats, on="hour_of_day")

    with np.errstate(invalid="ignore", divide="ignore"):
        raw_zscore = (block_counts["block_count"] - block_counts["hour_mean"]) / block_counts["hour_std"]
    block_counts["burst_zscore"] = np.clip(raw_zscore.where(block_counts["hour_std"] > 0, 0.0), a_min=0, a_max=None)

    merged = df.merge(
        block_counts[["day_of_period", "hour_of_day", "block_count", "burst_zscore"]],
        on=["day_of_period", "hour_of_day"],
    )
    return merged[["TransactionID", "block_count", "burst_zscore"]]


def _hour_history_depth(parquet_path: Path) -> pd.DataFrame:
    """How many prior transactions (causal, ordered by TransactionDT) this row's hour_of_day
    group had already seen — the same transparency signal entity_anomaly.py reports as
    history_depth, computed here for the hour segment instead of the card1 entity."""
    df = pq.ParquetFile(parquet_path).read(columns=["TransactionID", "TransactionDT"]).to_pandas()
    df["hour_of_day"] = (df["TransactionDT"] // SECONDS_PER_HOUR) % 24
    df = df.sort_values(["hour_of_day", "TransactionDT"])
    df["hour_history_depth"] = df.groupby("hour_of_day").cumcount()
    return df[["TransactionID", "hour_history_depth"]]


def compute_temporal_anomaly_scores(parquet_path: Path) -> pd.DataFrame:
    context = build_context_features(parquet_path)
    burst = compute_burst_scores(parquet_path)
    history = _hour_history_depth(parquet_path)

    df = context[["TransactionID", "amount_zscore_within_hour"]].merge(burst, on="TransactionID")
    df = df.merge(history, on="TransactionID")

    amount_component = df["amount_zscore_within_hour"].abs().fillna(0.0)
    burst_component = df["burst_zscore"]

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "temporal_anomaly_score": amount_component + burst_component,
        "temporal_anomaly_amount_component": amount_component,
        "temporal_anomaly_burst_component": burst_component,
        "temporal_anomaly_block_count": df["block_count"],
        "temporal_anomaly_hour_history": df["hour_history_depth"],
    })
