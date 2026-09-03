"""Case 4 — multivariate anomaly detection: how anomalous is a transaction across TransactionAmt,
C9, and C13 considered JOINTLY, not one column at a time (that's column_anomaly.py's job).

Feature scope note: column_anomaly.py found only 8 of 96 numeric columns are non-degenerate
(MAD != 0). Of those, only TransactionAmt, C9, and C13 are also fully populated (0% missing) —
the rest (id_02, D8, D9, id_21, id_25) are 76-99% missing, and requiring all 8 simultaneously
non-null leaves just 0.5% of rows (2,971 out of 590,540). Widening the missingness threshold to
even 30% adds nothing — the next non-degenerate column after these three jumps straight to 76%
missing; there is no gentle middle ground in this dataset. So this layer deliberately works with
three columns, not eight: thin, but every row can actually be scored, which a multivariate method
needs (unlike column_anomaly.py's independent per-column batching, a joint distance needs all its
dimensions valid at once for a given row).

Two methods, computed on the identical 3-column feature matrix (log1p applied wherever
distributions.py already recommended it — no new judgment call here), same "compare, don't just
pick one" instinct as quality.py's IQR vs MAD:
  - Mahalanobis distance (pure numpy): distance from the multivariate center, accounting for
    correlation between the three features. Assumes a roughly elliptical joint distribution.
    Directly explainable — each feature's own contribution to the squared distance is computable.
  - Isolation Forest (scikit-learn): tree-based, captures non-linear multivariate structure
    Mahalanobis can't. Not decomposable per-row the way Mahalanobis is; this module reports the
    feature with the largest univariate (MAD) deviation as an approximate explanation — not
    Isolation Forest's literal split logic, just a practical, cheap stand-in.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

from src.services.analyzers.distributions import compute_numeric_distributions

MULTIVARIATE_COLUMNS = ["TransactionAmt", "C9", "C13"]

ISOLATION_FOREST_N_ESTIMATORS = 200
ISOLATION_FOREST_RANDOM_STATE = 42
ISOLATION_FOREST_CONTAMINATION = "auto"


def _prepare_features(parquet_path: Path, columns: list[str] = MULTIVARIATE_COLUMNS) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=["TransactionID"] + columns).to_pandas()
    dist = compute_numeric_distributions(parquet_path, columns)
    log_cols = set(dist.loc[dist["log_transform_recommended"], "column"])
    for col in columns:
        if col in log_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def compute_mahalanobis_scores(parquet_path: Path, columns: list[str] = MULTIVARIATE_COLUMNS) -> pd.DataFrame:
    df = _prepare_features(parquet_path, columns)
    X = df[columns].to_numpy(dtype=np.float64)

    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)  # pseudo-inverse — safe even if cov is near-singular

    diff = X - mean
    contribution = (diff @ inv_cov) * diff  # per-dimension contribution to the squared distance
    squared_distance = contribution.sum(axis=1)
    distance = np.sqrt(np.clip(squared_distance, a_min=0, a_max=None))

    top_contributor = np.array(columns)[np.abs(contribution).argmax(axis=1)]

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "multivariate_mahalanobis_distance": distance,
        "multivariate_mahalanobis_top_contributor": top_contributor,
    })


def compute_isolation_forest_scores(parquet_path: Path, columns: list[str] = MULTIVARIATE_COLUMNS) -> pd.DataFrame:
    df = _prepare_features(parquet_path, columns)
    X = df[columns].to_numpy(dtype=np.float64)

    model = IsolationForest(
        n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=ISOLATION_FOREST_RANDOM_STATE,
    )
    model.fit(X)
    # decision_function: higher = more normal. Flip sign so higher = more anomalous, matching the
    # Mahalanobis distance convention above.
    anomaly_score = -model.decision_function(X)

    # Approximate per-row explanation: each feature's own MAD-based deviation, reused from
    # column_anomaly.py's mechanism — not Isolation Forest's actual split logic, just a cheap,
    # practical stand-in for "which single feature looks most unusual on its own here."
    per_col_z = np.zeros_like(X)
    for j, col in enumerate(columns):
        values = X[:, j]
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        per_col_z[:, j] = np.abs(values - median) / mad if mad > 0 else 0.0
    top_contributor = np.array(columns)[per_col_z.argmax(axis=1)]

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "multivariate_isolation_forest_score": anomaly_score,
        "multivariate_isolation_forest_top_contributor": top_contributor,
    })


def compute_multivariate_anomaly_scores(parquet_path: Path, columns: list[str] = MULTIVARIATE_COLUMNS) -> pd.DataFrame:
    mahalanobis = compute_mahalanobis_scores(parquet_path, columns)
    isolation_forest = compute_isolation_forest_scores(parquet_path, columns)
    return mahalanobis.merge(isolation_forest, on="TransactionID")
