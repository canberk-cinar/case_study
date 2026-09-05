"""Case 1: distribution analysis over the merged Parquet file.

Three distinct shapes of "distribution", handled separately because they answer different
questions and use different math:
  - numeric_continuous / monetary / count columns: percentiles, skew/kurtosis, zero/negative
    ratio, a log-transform recommendation for heavily right-skewed columns.
  - categorical-like columns (categorical, binary_flag, high_cardinality_text, and
    numeric_encoded_categorical: the latter is numeric-typed but flagged in column_types.py as
    not-safe-to-average, so it belongs here, not in the numeric summary): top-N values, Shannon
    entropy, normalized entropy, imbalance ratio.
  - TransactionDT: not really a numeric column at all here: it's an offset the competition
    documents as seconds from a reference point, so the useful "distribution" is derived
    day/hour volume and how TransactionAmt (and, descriptively only, isFraud) shifts across them.

Uses classify_columns() from column_types.py to route each column to the right treatment instead
of re-deriving cardinality/dtype rules here. Column-batched reads, same reasoning as the other
analyzers in this package.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import settings
from src.services.analyzers.column_types import classify_columns, profile_columns

logger = logging.getLogger(__name__)

NUMERIC_SEMANTIC_TYPES = {"numeric_continuous", "monetary", "count"}
CATEGORICAL_SEMANTIC_TYPES = {"categorical", "binary_flag", "high_cardinality_text", "numeric_encoded_categorical"}

# A right-skewed numeric column past this point is flagged as a log-transform candidate: 2.0 is
# a common "highly skewed" cutoff (vs. ~0.5-1.0 for "moderately skewed"); log1p needs min >= 0.
LOG_TRANSFORM_SKEW_THRESHOLD = 2.0

# How many of a categorical column's most frequent values to report by name.
TOP_N_CATEGORICAL_VALUES = 10

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600


def compute_numeric_distributions(parquet_path: Path, columns: list[str], batch_size: int = 25) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)
    rows = []

    for i in range(0, len(columns), batch_size):
        batch_cols = columns[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        for col in batch_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            skewness = float(series.skew())
            zero_ratio = float((series == 0).mean())
            negative_ratio = float((series < 0).mean())
            log_transform_recommended = bool(skewness > LOG_TRANSFORM_SKEW_THRESHOLD and negative_ratio == 0.0)

            rows.append({
                "column": col,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p1": float(series.quantile(0.01)),
                "p5": float(series.quantile(0.05)),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "skewness": skewness,
                "kurtosis": float(series.kurt()),
                "zero_ratio": zero_ratio,
                "negative_ratio": negative_ratio,
                "log_transform_recommended": log_transform_recommended,
            })

    return pd.DataFrame(rows).sort_values("skewness", ascending=False).reset_index(drop=True)


def _shannon_entropy(value_counts: pd.Series) -> float:
    probs = value_counts / value_counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def compute_categorical_distributions(parquet_path: Path, columns: list[str], batch_size: int = 25) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)
    rows = []

    for i in range(0, len(columns), batch_size):
        batch_cols = columns[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        for col in batch_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            value_counts = series.value_counts()
            nunique = len(value_counts)
            entropy = _shannon_entropy(value_counts)
            max_entropy = np.log2(nunique) if nunique > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            imbalance_ratio = float(value_counts.iloc[0] / value_counts.sum())

            top_values = [
                {"value": str(v), "count": int(c), "pct": round(c / len(series) * 100, 2)}
                for v, c in value_counts.head(TOP_N_CATEGORICAL_VALUES).items()
            ]

            rows.append({
                "column": col,
                "nunique": nunique,
                "entropy_bits": entropy,
                "normalized_entropy": normalized_entropy,
                "imbalance_ratio": imbalance_ratio,
                "top_values": top_values,
            })

    return pd.DataFrame(rows).sort_values("normalized_entropy").reset_index(drop=True)


def analyze_time_distribution(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TransactionDT is seconds elapsed from an undocumented reference point (not a real
    timestamp): day/hour are derived by simple integer division, the standard approach for this
    dataset. Returns (daily, hourly) volume/amount/fraud-rate tables. isFraud is used here purely
    descriptively (how the label happens to distribute over time): never to drive a typing or
    cleaning decision elsewhere in this pipeline."""
    df = pq.ParquetFile(parquet_path).read(columns=["TransactionDT", "TransactionAmt", "isFraud"]).to_pandas()
    df["day"] = df["TransactionDT"] // SECONDS_PER_DAY
    df["hour_of_day"] = (df["TransactionDT"] // SECONDS_PER_HOUR) % 24

    daily = df.groupby("day").agg(
        transaction_count=("TransactionDT", "count"),
        mean_amount=("TransactionAmt", "mean"),
        median_amount=("TransactionAmt", "median"),
        fraud_rate_pct=("isFraud", lambda s: s.mean() * 100),
    ).reset_index()

    hourly = df.groupby("hour_of_day").agg(
        transaction_count=("TransactionDT", "count"),
        mean_amount=("TransactionAmt", "mean"),
        median_amount=("TransactionAmt", "median"),
        fraud_rate_pct=("isFraud", lambda s: s.mean() * 100),
    ).reset_index()

    return daily, hourly


def run() -> dict:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    profile = profile_columns(parquet_path)
    classified = classify_columns(profile)

    numeric_cols = classified[classified["semantic_type"].isin(NUMERIC_SEMANTIC_TYPES)]["column"].tolist()
    categorical_cols = classified[classified["semantic_type"].isin(CATEGORICAL_SEMANTIC_TYPES)]["column"].tolist()

    numeric_dist = compute_numeric_distributions(parquet_path, numeric_cols)
    categorical_dist = compute_categorical_distributions(parquet_path, categorical_cols)
    daily, hourly = analyze_time_distribution(parquet_path)

    print(f"=== Numeric distributions ({len(numeric_dist)} columns): top 10 most skewed ===")
    print(numeric_dist.head(10)[["column", "skewness", "kurtosis", "zero_ratio", "log_transform_recommended"]].to_string(index=False))

    print(f"\n=== Categorical distributions ({len(categorical_dist)} columns): top 10 most imbalanced (lowest normalized entropy) ===")
    print(categorical_dist.head(10)[["column", "nunique", "normalized_entropy", "imbalance_ratio"]].to_string(index=False))

    print("\n=== Daily volume/amount/fraud rate (first 5 days) ===")
    print(daily.head().to_string(index=False))

    print("\n=== Hourly volume/amount/fraud rate ===")
    print(hourly.to_string(index=False))

    return {
        "numeric": numeric_dist,
        "categorical": categorical_dist,
        "daily": daily,
        "hourly": hourly,
    }
