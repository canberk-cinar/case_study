"""Case 2: entity behavior pattern analysis.

Case 1's cardinality analysis flagged card1/addr1/DeviceInfo/... as entity keys to preserve for
aggregation rather than encode away. This is that aggregation put to use, applied to per-card1
behavior: transaction count, how many distinct regions/devices it operates from, how variable its
spending is, how fast consecutive transactions come in. These are exactly the signals a later
anomaly-detection stage would build velocity/diversity features from.

card1 is the primary entity: the highest-cardinality, most literal "customer" proxy among the
ten preserved entity keys. addr1 and DeviceInfo are folded in as per-entity diversity dimensions
rather than run as separate entities: a single region or device code isn't itself something with
"behavior" the way a recurring card is.

isFraud is included purely descriptively (per-entity fraud rate): same rule as the rest of this
pipeline: it never decides which entities get flagged, only how the result is narrated.

Fully vectorized (groupby().agg() + groupby().diff()), not a per-group Python callback: card1
alone has ~13.5k distinct values, so a row-by-row .apply() over that many groups would be slow.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import settings

ENTITY_COLUMN = "card1"
BEHAVIOR_SOURCE_COLUMNS = [ENTITY_COLUMN, "addr1", "DeviceInfo", "TransactionAmt", "TransactionDT", "isFraud"]

# A single-transaction entity has no "pattern" to speak of: one data point, nothing to compare.
MIN_TRANSACTIONS_FOR_PROFILE = 2

HIGH_ADDR_DIVERSITY_THRESHOLD = 3   # transacting from 3+ distinct regions is unusual for one card
HIGH_AMOUNT_CV_THRESHOLD = 1.5      # coefficient of variation: spending swings wildly
FAST_VELOCITY_SECONDS = 60          # consecutive transactions under a minute apart


def build_entity_profiles(parquet_path: Path, entity_column: str = ENTITY_COLUMN) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=BEHAVIOR_SOURCE_COLUMNS).to_pandas()
    df = df.dropna(subset=[entity_column]).sort_values([entity_column, "TransactionDT"])

    grouped = df.groupby(entity_column)
    df["gap_seconds"] = grouped["TransactionDT"].diff()  # NaN at each entity's first row, by construction

    profiles = df.groupby(entity_column).agg(
        transaction_count=("TransactionDT", "size"),
        distinct_addr1=("addr1", "nunique"),
        distinct_device_info=("DeviceInfo", "nunique"),
        amount_mean=("TransactionAmt", "mean"),
        amount_std=("TransactionAmt", "std"),
        min_gap_seconds=("gap_seconds", "min"),
        median_gap_seconds=("gap_seconds", "median"),
        fraud_rate_pct=("isFraud", "mean"),
    ).reset_index().rename(columns={entity_column: "entity"})

    profiles["amount_cv"] = profiles["amount_std"] / profiles["amount_mean"]
    profiles["fraud_rate_pct"] *= 100
    return profiles[profiles["transaction_count"] >= MIN_TRANSACTIONS_FOR_PROFILE].reset_index(drop=True)


def flag_unusual_entities(profiles: pd.DataFrame) -> pd.DataFrame:
    result = profiles.copy()
    result["high_addr_diversity"] = result["distinct_addr1"] >= HIGH_ADDR_DIVERSITY_THRESHOLD
    result["high_amount_variability"] = result["amount_cv"].fillna(0) >= HIGH_AMOUNT_CV_THRESHOLD
    result["fast_velocity"] = result["min_gap_seconds"].fillna(float("inf")) <= FAST_VELOCITY_SECONDS
    result["unusual_flag_count"] = result[
        ["high_addr_diversity", "high_amount_variability", "fast_velocity"]
    ].sum(axis=1)
    return result


def summarize_entity_profiles(profiles: pd.DataFrame) -> dict:
    return {
        "profiled_entities": len(profiles),
        "median_transactions_per_entity": float(profiles["transaction_count"].median()),
        "max_transactions_per_entity": int(profiles["transaction_count"].max()),
        "entities_with_multiple_addr1": int((profiles["distinct_addr1"] > 1).sum()),
        "entities_with_multiple_devices": int((profiles["distinct_device_info"] > 1).sum()),
    }
