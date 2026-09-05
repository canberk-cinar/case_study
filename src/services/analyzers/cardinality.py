"""Case 1: high-cardinality analysis over the merged Parquet file.

Reuses classify_columns() from column_types.py instead of rescanning the file: cardinality
tiering and encoding recommendations are just a further read on stats already computed there.

Same two-layer approach as column_types.py: a general statistical rule (tier + quasi-identifier
flag from nunique/unique_ratio) gives an encoding recommendation for every column, then a small,
explicit domain-hint list (ENTITY_KEY_COLUMN_HINTS) overrides it for columns known: from the
IEEE-CIS schema, not from statistics: to be anonymized entity identifiers (card/address codes,
device fingerprint, email domain) rather than disposable high-cardinality noise. This is the
key distinction: a high-cardinality column is not automatically a "bad" column to drop: some of
them are exactly the aggregation keys a later anomaly-detection stage will want to group by.
"""
import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.services.analyzers.column_types import classify_columns, profile_columns

logger = logging.getLogger(__name__)

# Cardinality tiers, by raw distinct-value count.
CARDINALITY_TIER_BOUNDS = [
    (2, "binary"),
    (10, "low"),
    (50, "medium"),
    (1000, "high"),
]  # anything above the last bound is "very_high"

# A column where >90% of non-null values are distinct is a quasi-identifier: encoding it the
# normal way (one-hot, frequency) would just memorize row identity, not generalize.
QUASI_IDENTIFIER_UNIQUE_RATIO = 0.9

# Domain knowledge (IEEE-CIS schema), not statistically derived: these are anonymized entity
# identifiers a later anomaly-detection stage will likely want as aggregation keys, not features
# to one-hot/frequency-encode or drop for being high-cardinality.
ENTITY_KEY_COLUMN_HINTS = {
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "DeviceInfo", "DeviceType",
    "P_emaildomain", "R_emaildomain",
}

# Semantic types that were already flagged in column_types.py as "not safe to average": the
# ones a cardinality/encoding recommendation is actually about.
CATEGORICAL_LIKE_TYPES = {"categorical", "binary_flag", "high_cardinality_text", "numeric_encoded_categorical"}
CONTINUOUS_TYPES = {"numeric_continuous", "monetary", "count"}
NOT_APPLICABLE_TYPES = {"identifier", "target", "timestamp_offset"}


def cardinality_tier(nunique: int) -> str:
    for bound, label in CARDINALITY_TIER_BOUNDS:
        if nunique <= bound:
            return label
    return "very_high"


def recommend_encoding(row: pd.Series) -> str:
    name = row["column"]
    semantic_type = row["semantic_type"]
    tier = row["cardinality_tier"]
    is_quasi_id = row["is_quasi_identifier"]

    if semantic_type in NOT_APPLICABLE_TYPES:
        return "n/a: handled separately (identifier/target/timestamp)"

    if name in ENTITY_KEY_COLUMN_HINTS:
        return "preserve as entity key"

    if semantic_type in CONTINUOUS_TYPES:
        if is_quasi_id:
            return "needs review: continuous but near-unique, possible leakage/noise"
        return "n/a: continuous numeric, no encoding needed"

    # remaining: categorical-like columns not already caught by the entity-key hint list
    if is_quasi_id:
        return "drop: near-unique per row, won't generalize"
    if tier in ("binary", "low", "medium"):
        return "one-hot"
    if tier == "high":
        return "frequency encoding"
    return "hashing or entity-key candidate: needs manual review"  # very_high, not in hint list


def compute_cardinality(classified: pd.DataFrame) -> pd.DataFrame:
    result = classified.copy()
    result["cardinality_tier"] = result["nunique"].apply(cardinality_tier)
    result["is_quasi_identifier"] = result["unique_ratio"] > QUASI_IDENTIFIER_UNIQUE_RATIO
    result["encoding_recommendation"] = result.apply(recommend_encoding, axis=1)
    return result[[
        "column", "semantic_type", "nunique", "unique_ratio",
        "cardinality_tier", "is_quasi_identifier", "encoding_recommendation",
    ]]


def run() -> pd.DataFrame:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    profile = profile_columns(parquet_path)
    classified = classify_columns(profile)
    cardinality = compute_cardinality(classified)

    print("=== Cardinality tier distribution ===")
    print(cardinality["cardinality_tier"].value_counts().to_string())

    print("\n=== Encoding recommendation distribution ===")
    print(cardinality["encoding_recommendation"].value_counts().to_string())

    print("\n=== Columns preserved as entity keys ===")
    entity_keys = cardinality[cardinality["encoding_recommendation"] == "preserve as entity key"]
    print(entity_keys.to_string(index=False))

    print("\n=== Quasi-identifiers (unique_ratio > 0.9) ===")
    print(cardinality[cardinality["is_quasi_identifier"]].to_string(index=False))

    return cardinality
