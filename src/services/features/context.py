"""Case 3 — context features: how a transaction compares to the distribution of its own context
segment (product type, hour of day) — not the entity's personal history (entity.py), and not the
relationship between entity and context values (relational.py), but where the transaction sits
relative to everyone else transacting in the same segment at that point in time.

Same causal ("_so_far") discipline as entity.py/relational.py: a segment's running mean/std only
reflects transactions strictly before the current one, ordered by TransactionDT. Reuses the exact
"expanding, then shift by one row" pattern from entity.py, grouped by a context segment instead of
an entity key — no new derivation, just a different groupby column.

Two segments, chosen for being genuinely interpretable rather than an exhaustive scan of every
possible grouping column:
  - ProductCD: the transaction's product category (5 values) — Case 1 found this column's
    identity coverage itself varies drastically by ProductCD, so amount behavior likely does too.
  - hour_of_day: reuses temporal.py's derivation — Case 1 found a strong fraud-rate/hour
    relationship, so "is this amount unusual for this hour" is a natural next question.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.services.features.temporal import SECONDS_PER_HOUR

SOURCE_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]


def _causal_segment_stats(
    df: pd.DataFrame, segment_col: str, value_col: str, time_col: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (prior_mean, prior_std, zscore) for `value_col` within `segment_col` groups,
    ordered by `time_col` — the same expanding-then-shift mechanism entity.py uses for card1,
    applied here to a context segment instead of an entity."""
    ordered = df.sort_values([segment_col, time_col])
    grouped = ordered.groupby(segment_col)[value_col]

    cumsum_incl = grouped.cumsum()
    cumcount_incl = grouped.cumcount() + 1
    prior_count = cumcount_incl - 1
    prior_sum = cumsum_incl - ordered[value_col]
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, prior_sum / prior_count, np.nan)

    expanding_std_incl = grouped.expanding().std().reset_index(level=0, drop=True)
    prior_std = expanding_std_incl.groupby(ordered[segment_col]).shift(1).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(
            (prior_count.to_numpy() > 0) & (prior_std > 0),
            (ordered[value_col].to_numpy() - prior_mean) / prior_std,
            np.nan,
        )

    idx = ordered.index
    return (
        pd.Series(prior_mean, index=idx).reindex(df.index),
        pd.Series(prior_std, index=idx).reindex(df.index),
        pd.Series(zscore, index=idx).reindex(df.index),
    )


def build_context_features(parquet_path: Path) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=SOURCE_COLUMNS).to_pandas()
    df["hour_of_day"] = (df["TransactionDT"] // SECONDS_PER_HOUR) % 24

    productcd_mean, _, productcd_zscore = _causal_segment_stats(
        df, "ProductCD", "TransactionAmt", "TransactionDT"
    )
    hour_mean, _, hour_zscore = _causal_segment_stats(
        df, "hour_of_day", "TransactionAmt", "TransactionDT"
    )

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "productcd_avg_amount_so_far": productcd_mean,
        "amount_zscore_within_productcd": productcd_zscore,
        "hour_avg_amount_so_far": hour_mean,
        "amount_zscore_within_hour": hour_zscore,
    })
