"""Case 1: data quality analysis over the merged Parquet file.

Covers: duplicate rows, duplicate TransactionIDs, duplicate (value-identical) column pairs,
business-rule consistency checks, IQR/MAD-based outlier ratios, and text-value consistency
(whitespace/case collisions): closed out with a weighted quality scorecard.

Reuses column_types.py's per-column profile (dtype, nunique, min/max, ...) instead of
recomputing it: the same statistics answer two different questions (typing vs. quality), no
reason to scan the file twice for them. Everything else stays column-batched or reads only a
narrow, targeted set of columns, for the same memory reasons as missingness.py/column_types.py.
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import settings
from src.services.analyzers.column_types import classify_columns, profile_columns

logger = logging.getLogger(__name__)

COUNT_COLUMN_PATTERN = re.compile(r"^C\d+$")

# Modified z-score constant (0.6745 makes MAD comparable to standard deviation under normality)
# and the conventional outlier threshold: Iglewicz & Hoaglin's rule of thumb, not tuned per column.
MAD_CONSISTENCY_CONSTANT = 0.6745
MAD_OUTLIER_THRESHOLD = 3.5
IQR_OUTLIER_MULTIPLIER = 1.5

# Scorecard weights: duplicate TransactionIDs would mean the merge itself is broken (most
# severe); duplicate rows and business-rule violations are real data problems; outliers are the
# least severe since a fraud dataset is expected to have genuine extreme values.
QUALITY_WEIGHTS = {
    "duplicate_rows": 25,
    "duplicate_transaction_ids": 30,
    "value_consistency": 25,
    "text_consistency": 10,
    "outliers": 10,
}


def check_duplicate_rows(parquet_path: Path, batch_size: int = 25) -> dict:
    """Full-row duplicate detection without ever holding all 434 columns at once: each column
    batch is hashed per-row (pandas' hash_pandas_object) and combined across batches with XOR,
    giving a single 64-bit fingerprint per row. Collision risk at ~590k rows / 64-bit hashes is
    negligible (well below the birthday bound), so this is treated as exact rather than
    approximate."""
    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names
    num_rows = pf.metadata.num_rows

    combined_hash = np.zeros(num_rows, dtype=np.uint64)
    for i in range(0, len(columns), batch_size):
        batch_cols = columns[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        batch_hash = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
        combined_hash ^= batch_hash

    hashes = pd.Series(combined_hash)
    dup_mask = hashes.duplicated(keep=False)

    return {
        "duplicate_row_count": int(dup_mask.sum()),
        "duplicate_row_pct": float(dup_mask.mean() * 100),
        "duplicate_group_count": int(hashes[dup_mask].nunique()),
    }


def check_duplicate_transaction_ids(parquet_path: Path) -> dict:
    ids = pq.ParquetFile(parquet_path).read(columns=["TransactionID"]).to_pandas()["TransactionID"]
    return {
        "row_count": len(ids),
        "unique_count": int(ids.nunique()),
        "duplicate_count": int(len(ids) - ids.nunique()),
    }


def find_duplicate_columns(profile: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    """Value-identical column pairs. Cheap fingerprint first (dtype, non-null count, nunique,
    min, max: already computed by column_types.profile_columns), verified only within groups
    that share a fingerprint, reading just those columns: avoids comparing all 434*433/2 pairs
    directly. Expected to surface real hits inside the V-column families found in the
    missing-pattern analysis."""
    fingerprint_cols = ["physical_dtype", "non_null_count", "nunique", "min", "max"]
    grouped = profile.groupby(fingerprint_cols, dropna=False)["column"].apply(list)
    candidate_groups = [cols for cols in grouped if len(cols) >= 2]

    pf = pq.ParquetFile(parquet_path)
    results = []
    for group in candidate_groups:
        df = pf.read(columns=group).to_pandas()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if df[a].equals(df[b]):
                    results.append({"column_a": a, "column_b": b})

    return pd.DataFrame(results, columns=["column_a", "column_b"])


def check_value_consistency(parquet_path: Path) -> pd.DataFrame:
    """A handful of IEEE-CIS-specific business-rule checks: transaction amounts must be
    positive, count features (C1..C14) must be non-negative, and TransactionDT (seconds elapsed
    from a reference point) should not decrease row over row."""
    pf = pq.ParquetFile(parquet_path)
    num_rows = pf.metadata.num_rows
    findings = []

    amt = pf.read(columns=["TransactionAmt"]).to_pandas()["TransactionAmt"]
    non_positive = int((amt <= 0).sum())
    findings.append({
        "check": "TransactionAmt <= 0",
        "violation_count": non_positive,
        "violation_pct": non_positive / num_rows * 100,
    })

    count_cols = [c for c in pf.schema_arrow.names if COUNT_COLUMN_PATTERN.match(c)]
    df_counts = pf.read(columns=count_cols).to_pandas()
    for col in count_cols:
        neg = int((df_counts[col] < 0).sum())
        findings.append({
            "check": f"{col} < 0",
            "violation_count": neg,
            "violation_pct": neg / num_rows * 100,
        })

    dt = pf.read(columns=["TransactionDT"]).to_pandas()["TransactionDT"]
    decreasing = int((dt.diff().dropna() < 0).sum())
    findings.append({
        "check": "TransactionDT decreases row over row",
        "violation_count": decreasing,
        "violation_pct": decreasing / num_rows * 100,
    })

    return pd.DataFrame(findings)


def compute_outlier_ratios(parquet_path: Path, numeric_columns: list[str], batch_size: int = 25) -> pd.DataFrame:
    """IQR (Tukey's fences) and MAD-based (modified z-score) outlier ratios, column-batched.
    Reported side by side because they disagree in informative ways: IQR is sensitive to skew
    (flags a lot on right-skewed columns like TransactionAmt), MAD is more robust to skew but
    degenerates (mad == 0) on columns dominated by a single repeated value."""
    pf = pq.ParquetFile(parquet_path)
    rows = []

    for i in range(0, len(numeric_columns), batch_size):
        batch_cols = numeric_columns[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        for col in batch_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - IQR_OUTLIER_MULTIPLIER * iqr, q3 + IQR_OUTLIER_MULTIPLIER * iqr
            iqr_outlier_ratio = float(((series < lower) | (series > upper)).mean())

            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0:
                mad_outlier_ratio = None
            else:
                modified_z = MAD_CONSISTENCY_CONSTANT * (series - median).abs() / mad
                mad_outlier_ratio = float((modified_z > MAD_OUTLIER_THRESHOLD).mean())

            rows.append({
                "column": col,
                "iqr_outlier_ratio": iqr_outlier_ratio,
                "mad_outlier_ratio": mad_outlier_ratio,
            })

    return pd.DataFrame(rows).sort_values("iqr_outlier_ratio", ascending=False).reset_index(drop=True)


def check_text_consistency(parquet_path: Path, text_columns: list[str]) -> pd.DataFrame:
    """Flags text columns where distinct values collapse under case/whitespace normalization:
    e.g. "gmail.com" vs "gmail.com " being counted as different categories would silently
    fragment a categorical encoding downstream."""
    if not text_columns:
        return pd.DataFrame(columns=["column", "raw_nunique", "normalized_nunique", "collisions"])

    df = pq.ParquetFile(parquet_path).read(columns=text_columns).to_pandas()
    rows = []
    for col in text_columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        normalized = series.str.strip().str.lower()
        raw_nunique = series.nunique()
        normalized_nunique = normalized.nunique()
        collisions = raw_nunique - normalized_nunique
        if collisions > 0:
            rows.append({
                "column": col,
                "raw_nunique": raw_nunique,
                "normalized_nunique": normalized_nunique,
                "collisions": collisions,
            })

    return pd.DataFrame(rows)


def build_quality_scorecard(
    duplicate_rows: dict,
    duplicate_ids: dict,
    consistency_findings: pd.DataFrame,
    text_findings: pd.DataFrame,
    outlier_ratios: pd.DataFrame,
    num_rows: int,
) -> dict:
    dup_row_score = 100 * (1 - min(duplicate_rows["duplicate_row_pct"] / 100, 1))
    dup_id_score = 100 * (1 - min(duplicate_ids["duplicate_count"] / num_rows, 1))

    consistency_violation_pct = consistency_findings["violation_pct"].sum() if not consistency_findings.empty else 0.0
    consistency_score = 100 * (1 - min(consistency_violation_pct / 100, 1))

    text_collision_total = int(text_findings["collisions"].sum()) if not text_findings.empty else 0
    text_score = 100.0 if text_collision_total == 0 else max(0.0, 100 - text_collision_total)

    mean_outlier_ratio = outlier_ratios["iqr_outlier_ratio"].mean() if not outlier_ratios.empty else 0.0
    outlier_score = 100 * (1 - min(mean_outlier_ratio, 1))

    scores = {
        "duplicate_rows": dup_row_score,
        "duplicate_transaction_ids": dup_id_score,
        "value_consistency": consistency_score,
        "text_consistency": text_score,
        "outliers": outlier_score,
    }
    total_weight = sum(QUALITY_WEIGHTS.values())
    overall = sum(scores[k] * QUALITY_WEIGHTS[k] for k in scores) / total_weight

    return {f"{k}_score": round(v, 2) for k, v in scores.items()} | {"overall_score": round(overall, 2)}


def run() -> dict:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    num_rows = pq.ParquetFile(parquet_path).metadata.num_rows

    profile = profile_columns(parquet_path)
    classified = classify_columns(profile)

    dup_rows = check_duplicate_rows(parquet_path)
    dup_ids = check_duplicate_transaction_ids(parquet_path)
    dup_cols = find_duplicate_columns(profile, parquet_path)
    consistency = check_value_consistency(parquet_path)

    numeric_cols = classified[classified["semantic_type"].isin(["numeric_continuous", "monetary"])]["column"].tolist()
    outliers = compute_outlier_ratios(parquet_path, numeric_cols)

    text_cols = classified[classified["semantic_type"].isin(["categorical", "high_cardinality_text"])]["column"].tolist()
    text_consistency = check_text_consistency(parquet_path, text_cols)

    scorecard = build_quality_scorecard(dup_rows, dup_ids, consistency, text_consistency, outliers, num_rows)

    print("=== Duplicate rows ===")
    print(dup_rows)
    print("\n=== Duplicate TransactionID ===")
    print(dup_ids)
    print(f"\n=== Duplicate column pairs ({len(dup_cols)} pairs) ===")
    print(dup_cols.to_string(index=False))
    print("\n=== Consistency checks ===")
    print(consistency.to_string(index=False))
    print("\n=== Outlier ratio: top 10 columns ===")
    print(outliers.head(10).to_string(index=False))
    print(f"\n=== Text inconsistencies ({len(text_consistency)} columns affected) ===")
    print(text_consistency.to_string(index=False))
    print("\n=== Quality scorecard ===")
    for key, value in scorecard.items():
        print(f"{key}: {value}")

    return {
        "duplicate_rows": dup_rows,
        "duplicate_ids": dup_ids,
        "duplicate_columns": dup_cols,
        "consistency": consistency,
        "outliers": outliers,
        "text_consistency": text_consistency,
        "scorecard": scorecard,
    }
