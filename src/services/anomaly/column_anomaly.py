"""Case 4: column-level (univariate) anomaly detection.

For each numeric column, scores every row by how far its value sits from that column's typical
value: a modified (MAD-based) z-score, robust to the outliers/skew Case 1 found dominate this
dataset (91 of 96 numeric columns were flagged as heavily right-skewed). Log1p is applied first
wherever distributions.py already recommended it, so this reuses Case 1's already-validated
per-column judgment rather than re-deriving it with a second set of rules.

This is the "column anomaly" layer of Case 4's multi-layer architecture: it looks at each column
in isolation, unlike the multivariate layer (columns considered together) or the entity/temporal
layers (a row's own history). Column-batched, same reasoning as every other analyzer/feature
module in this project: the full per-column z-score matrix (96 x 590k rows, ~450MB) is never
held in memory at once; only four running per-row accumulators are, each ~5MB.

Combined into one score per row via four signals:
  - column_anomaly_mean_abs_zscore: average anomalousness across all scored columns for this row
   : the layer's primary score.
  - column_anomaly_scored_count: how many columns actually had a value for this row (most rows
    are missing most of the 96 numeric columns: Case 1 found an average of ~196 missing columns
    per row out of 434). This is reported explicitly because it matters for interpreting the
    score: a row with very few valid columns and one extreme value gets a much noisier
    mean_abs_zscore than a row averaging the same extreme value across many columns: the score
    is not directly comparable across rows with very different scored_count.
  - column_anomaly_extreme_column_count: how many columns individually cross
    MAD_OUTLIER_THRESHOLD (reused from quality.py: same definition of "extreme" used there).
  - column_anomaly_top_column / column_anomaly_max_abs_zscore: the single column driving the
    row's score hardest: the explainability anchor ("flagged mainly because of V266").
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.services.analyzers.column_types import classify_columns, profile_columns
from src.services.analyzers.distributions import NUMERIC_SEMANTIC_TYPES, compute_numeric_distributions
from src.services.analyzers.quality import MAD_CONSISTENCY_CONSTANT, MAD_OUTLIER_THRESHOLD

logger = logging.getLogger(__name__)


def compute_column_anomaly_scores(parquet_path: Path, batch_size: int = 25) -> pd.DataFrame:
    profile = profile_columns(parquet_path)
    classified = classify_columns(profile)
    numeric_cols = classified[classified["semantic_type"].isin(NUMERIC_SEMANTIC_TYPES)]["column"].tolist()

    dist = compute_numeric_distributions(parquet_path, numeric_cols)
    log_transform_cols = set(dist.loc[dist["log_transform_recommended"], "column"])

    pf = pq.ParquetFile(parquet_path)
    ids = pf.read(columns=["TransactionID"]).to_pandas()["TransactionID"]
    num_rows = len(ids)

    sum_abs_zscore = np.zeros(num_rows, dtype=np.float64)
    scored_count = np.zeros(num_rows, dtype=np.int32)
    extreme_column_count = np.zeros(num_rows, dtype=np.int32)
    max_abs_zscore = np.zeros(num_rows, dtype=np.float64)
    max_abs_zscore_column = np.full(num_rows, "", dtype=object)

    for i in range(0, len(numeric_cols), batch_size):
        batch_cols = numeric_cols[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        for col in batch_cols:
            values = df[col].to_numpy(dtype=np.float64)
            if col in log_transform_cols:
                values = np.log1p(np.clip(values, a_min=0, a_max=None))

            valid = ~np.isnan(values)
            if valid.sum() == 0:
                continue
            median = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values[valid] - median))
            if mad == 0:
                continue

            abs_z = np.full(num_rows, np.nan)
            abs_z[valid] = np.abs(MAD_CONSISTENCY_CONSTANT * (values[valid] - median) / mad)

            row_valid = ~np.isnan(abs_z)
            sum_abs_zscore[row_valid] += abs_z[row_valid]
            scored_count[row_valid] += 1
            extreme_column_count[row_valid] += (abs_z[row_valid] > MAD_OUTLIER_THRESHOLD).astype(int)

            beats_max = row_valid & (abs_z > max_abs_zscore)
            max_abs_zscore[beats_max] = abs_z[beats_max]
            max_abs_zscore_column[beats_max] = col

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs_zscore = np.where(scored_count > 0, sum_abs_zscore / scored_count, np.nan)

    return pd.DataFrame({
        "TransactionID": ids,
        "column_anomaly_mean_abs_zscore": mean_abs_zscore,
        "column_anomaly_scored_count": scored_count,
        "column_anomaly_extreme_column_count": extreme_column_count,
        "column_anomaly_max_abs_zscore": max_abs_zscore,
        "column_anomaly_top_column": max_abs_zscore_column,
    })
