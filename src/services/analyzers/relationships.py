"""Case 2 — column relationship discovery.

Three relationship types, each needing different math and a different column scope:

  - numeric-numeric: Pearson (linear) + Spearman (monotonic, rank-based — catches relationships
    Pearson misses on the heavily right-skewed columns Case 1 found) correlation across all 96
    numeric columns at once. Unlike the other analyzers in this package, correlation genuinely
    needs the full column set loaded together (it's a pairwise, not per-column, statistic) — read
    as float32 rather than float64 to roughly halve the ~280MB footprint, checked against this
    machine's RAM budget before running.
  - categorical-categorical: Cramér's V (contingency-table association strength, computed here
    without a scipy dependency — chi-square from the observed/expected counts directly).
  - numeric-categorical: eta-squared (how much of a numeric column's variance sits between a
    categorical column's groups vs. within them).

Both categorical-involving checks are restricted to a fixed, documented set of interpretable
low/moderate-cardinality columns (RELATIONSHIP_CATEGORICAL_COLUMNS) rather than all ~300
categorical-like columns — same reasoning as rare_combinations.py: scanning all pairs among
V-family columns would be slow, and would mostly surface expected in-family associations rather
than genuine cross-domain relationships.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import settings

# Only surface a numeric-numeric pair if at least one of Pearson/Spearman clears this.
NUMERIC_CORRELATION_THRESHOLD = 0.7

RELATIONSHIP_CATEGORICAL_COLUMNS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

CRAMERS_V_THRESHOLD = 0.3
ETA_SQUARED_THRESHOLD = 0.06  # Cohen's conventional "medium effect size" cutoff


def compute_numeric_correlations(
    parquet_path: Path, numeric_columns: list[str], threshold: float = NUMERIC_CORRELATION_THRESHOLD
) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=numeric_columns).to_pandas().astype("float32")

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    rows = []
    cols = pearson.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p, s = pearson.iloc[i, j], spearman.iloc[i, j]
            if pd.notna(p) and pd.notna(s) and (abs(p) >= threshold or abs(s) >= threshold):
                rows.append({"column_a": cols[i], "column_b": cols[j], "pearson": float(p), "spearman": float(s)})

    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def cramers_v(contingency: pd.DataFrame) -> float:
    observed = contingency.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return 0.0
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0))
    r, k = observed.shape
    denom = min(r - 1, k - 1)
    return float(np.sqrt((chi2 / n) / denom)) if denom > 0 else 0.0


def compute_categorical_associations(
    parquet_path: Path, columns: list[str] = RELATIONSHIP_CATEGORICAL_COLUMNS, threshold: float = CRAMERS_V_THRESHOLD
) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=columns).to_pandas()

    rows = []
    for a, b in itertools.combinations(columns, 2):
        contingency = pd.crosstab(df[a], df[b])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        v = cramers_v(contingency)
        rows.append({"column_a": a, "column_b": b, "cramers_v": v})

    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    result["is_strong"] = result["cramers_v"] >= threshold
    return result


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    if df["group"].nunique() < 2:
        return 0.0
    grand_mean = df["value"].mean()
    ss_total = ((df["value"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = df.groupby("group", observed=True)["value"].apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)


def compute_numeric_categorical_relationships(
    parquet_path: Path,
    numeric_columns: list[str],
    categorical_columns: list[str] = RELATIONSHIP_CATEGORICAL_COLUMNS,
    threshold: float = ETA_SQUARED_THRESHOLD,
) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=numeric_columns + categorical_columns).to_pandas()

    rows = []
    for num_col in numeric_columns:
        for cat_col in categorical_columns:
            eta2 = eta_squared(df[num_col], df[cat_col])
            rows.append({"numeric_column": num_col, "categorical_column": cat_col, "eta_squared": eta2})

    result = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    result["is_strong"] = result["eta_squared"] >= threshold
    return result
