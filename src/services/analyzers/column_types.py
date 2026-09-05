"""Case 1: automatic column typing over the merged Parquet file.

Two layers, kept explicitly separate rather than collapsed into one guess:

  1. Statistical base classification: physical dtype, cardinality, integer-valuedness, dominant-
     value ratio. Defensible on any dataset, no prior knowledge of this schema required. This is
     what catches the case naive profiling tools usually miss: an int-typed column with few
     distinct values relative to row count (card1, addr1, ...) is a categorical code, not a
     quantity: averaging it would be meaningless even though pandas reports it as numeric.
  2. Domain-hint refinement: a small, explicitly separate table of exact-name and name-pattern
     overrides (isFraud -> target, TransactionDT -> timestamp_offset, C1..C14 -> count,
     TransactionAmt -> monetary). These come from knowing the IEEE-CIS schema, not from column
     statistics, and are documented as a distinct pass so the boundary between "inferred from data"
     and "asserted from domain knowledge" stays visible.

Reads in column batches, not the full 434-column frame at once: same reasoning as
missingness.py: peak memory stays bounded to one batch width regardless of the file's column
count, which matters on this machine's constrained RAM budget.
"""
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import settings

logger = logging.getLogger(__name__)

# --- statistical thresholds --------------------------------------------------------------------
IDENTIFIER_UNIQUE_RATIO = 0.98          # >=98% of non-null values are distinct
NEAR_CONSTANT_DOMINANCE_RATIO = 0.99    # one value covers >=99% of non-null rows
ENCODED_CATEGORICAL_MAX_UNIQUE_RATIO = 0.05  # integer-valued numeric, but few distinct vs. rows
HIGH_CARDINALITY_TEXT_MIN_NUNIQUE = 1000
HIGH_CARDINALITY_TEXT_MIN_UNIQUE_RATIO = 0.5

# --- domain hints (IEEE-CIS schema knowledge, not statistically derived) --------------------------
TARGET_COLUMN = "isFraud"
IDENTIFIER_COLUMNS = {"TransactionID"}
TIMESTAMP_OFFSET_COLUMNS = {"TransactionDT"}  # seconds elapsed from a reference point, not a real timestamp
MONETARY_COLUMN_NAMES = {"TransactionAmt"}
COUNT_COLUMN_PATTERN = re.compile(r"^C\d+$")


def profile_columns(parquet_path: Path, batch_size: int = 25) -> pd.DataFrame:
    """One pass, column-batched: physical dtype, non-null count, distinct count, unique ratio,
    dominant-value ratio, and (for numeric columns) min/max/integer-valuedness."""
    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names
    rows = []

    for i in range(0, len(columns), batch_size):
        batch_cols = columns[i : i + batch_size]
        df = pf.read(columns=batch_cols).to_pandas()
        for col in batch_cols:
            series = df[col]
            non_null = series.dropna()
            non_null_count = len(non_null)
            nunique = int(series.nunique(dropna=True))
            dtype_kind = series.dtype.kind  # 'i','f' numeric, 'O' object, 'b' bool

            min_val = max_val = None
            is_integer_valued = None
            if dtype_kind in "if" and non_null_count > 0:
                min_val = float(non_null.min())
                max_val = float(non_null.max())
                is_integer_valued = bool((non_null % 1 == 0).all())

            dominant_ratio = None
            if non_null_count > 0:
                dominant_ratio = float(non_null.value_counts(normalize=True).iloc[0])

            rows.append({
                "column": col,
                "physical_dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "nunique": nunique,
                "unique_ratio": nunique / non_null_count if non_null_count else 0.0,
                "dominant_value_ratio": dominant_ratio,
                "is_integer_valued": is_integer_valued,
                "min": min_val,
                "max": max_val,
            })

    return pd.DataFrame(rows)


def classify_statistical(row: pd.Series) -> str:
    """Base semantic type from statistics alone: no column-name knowledge."""
    dtype = row["physical_dtype"]
    is_numeric = dtype.startswith(("int", "float"))
    nunique = row["nunique"]
    unique_ratio = row["unique_ratio"]

    if nunique <= 1:
        return "constant"
    if row["dominant_value_ratio"] is not None and row["dominant_value_ratio"] >= NEAR_CONSTANT_DOMINANCE_RATIO:
        return "near_constant"
    if unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
        return "identifier"
    if nunique <= 2:
        return "binary_flag"

    if not is_numeric:
        if nunique >= HIGH_CARDINALITY_TEXT_MIN_NUNIQUE or unique_ratio >= HIGH_CARDINALITY_TEXT_MIN_UNIQUE_RATIO:
            return "high_cardinality_text"
        return "categorical"

    if unique_ratio <= ENCODED_CATEGORICAL_MAX_UNIQUE_RATIO and bool(row["is_integer_valued"]):
        return "numeric_encoded_categorical"
    return "numeric_continuous"


def apply_domain_hints(row: pd.Series, statistical_type: str) -> str:
    name = row["column"]
    if name == TARGET_COLUMN:
        return "target"
    if name in IDENTIFIER_COLUMNS:
        return "identifier"
    if name in TIMESTAMP_OFFSET_COLUMNS:
        return "timestamp_offset"
    if statistical_type == "numeric_continuous" and name in MONETARY_COLUMN_NAMES:
        return "monetary"
    if statistical_type in ("numeric_continuous", "numeric_encoded_categorical") and COUNT_COLUMN_PATTERN.match(name):
        return "count"
    return statistical_type


def classify_columns(profile: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    result["statistical_type"] = result.apply(classify_statistical, axis=1)
    result["semantic_type"] = result.apply(
        lambda row: apply_domain_hints(row, row["statistical_type"]), axis=1
    )
    return result


def run() -> pd.DataFrame:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    profile = profile_columns(parquet_path)
    classified = classify_columns(profile)

    print("=== Semantic type distribution ===")
    print(classified["semantic_type"].value_counts().to_string())

    print("\n=== Columns where a domain hint overrode the statistical guess ===")
    changed = classified[classified["statistical_type"] != classified["semantic_type"]]
    print(changed[["column", "statistical_type", "semantic_type"]].to_string(index=False))

    return classified
