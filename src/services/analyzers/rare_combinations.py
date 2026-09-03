"""Case 2 — rare categorical combination analysis.

Finds column-value combinations across a small set of interpretable categorical columns that
occur far less often than their marginal frequencies would suggest — the joint distribution can
be rare even when every individual value is common on its own, and that's exactly the pattern a
per-column profile (Case 1) can't see.

Scoped to a fixed, deliberately small set of columns (COMBINATION_COLUMNS) rather than all ~300
categorical-like columns: the combination space grows multiplicatively (this set's
5x4x4x2 = 160 possible combos is already interpretable; adding one more column of cardinality
~60, like an email domain, would blow that up to ~9,600 combos, most single-digit-occurrence and
not meaningfully "rare" so much as sparse by construction).
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import settings

COMBINATION_COLUMNS = ["ProductCD", "card4", "card6", "DeviceType"]

# A combination is flagged rare if it clears either floor — count alone would flag legitimate
# small segments as "rare" purely because ~590k rows makes any ratio look tiny, so both a small
# absolute count and a tiny ratio are required together with an OR (either one is suspicious
# enough to report) rather than needing both (which would under-flag).
RARE_COMBINATION_MAX_COUNT = 10
RARE_COMBINATION_MAX_RATIO = 0.0001  # 0.01% of rows


def compute_combination_frequencies(parquet_path: Path, columns: list[str] = COMBINATION_COLUMNS) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=columns).to_pandas()
    total_rows = len(df)
    non_null = df.dropna(subset=columns)

    counts = non_null.groupby(columns, dropna=False).size().reset_index(name="count")
    counts["ratio"] = counts["count"] / total_rows
    counts["is_rare"] = (counts["count"] <= RARE_COMBINATION_MAX_COUNT) | (counts["ratio"] <= RARE_COMBINATION_MAX_RATIO)
    return counts.sort_values("count").reset_index(drop=True)


def find_unobserved_combinations(parquet_path: Path, columns: list[str] = COMBINATION_COLUMNS) -> pd.DataFrame:
    """Combinations that are theoretically possible (cross-product of each column's observed
    values) but never occur in the data — either a structurally impossible pairing or simply
    below this dataset's scale to have hit yet."""
    df = pq.ParquetFile(parquet_path).read(columns=columns).to_pandas().dropna(subset=columns)
    observed = compute_combination_frequencies(parquet_path, columns)

    value_lists = [sorted(df[col].unique().tolist()) for col in columns]
    all_combos = pd.MultiIndex.from_product(value_lists, names=columns).to_frame(index=False)

    merged = all_combos.merge(observed[columns + ["count"]], on=columns, how="left")
    unobserved = merged[merged["count"].isna()].drop(columns="count").reset_index(drop=True)
    return unobserved


def run() -> dict:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"
    freq = compute_combination_frequencies(parquet_path)
    unobserved = find_unobserved_combinations(parquet_path)

    print(f"=== Kombinasyon uzayı: {len(freq)} gözlenen / olası {freq.shape[0] + len(unobserved)} ===")
    print(freq.to_string(index=False))

    print(f"\n=== Nadir kombinasyonlar ({int(freq['is_rare'].sum())} adet) ===")
    print(freq[freq["is_rare"]].to_string(index=False))

    print(f"\n=== Hiç gözlenmemiş kombinasyonlar ({len(unobserved)} adet) ===")
    print(unobserved.to_string(index=False))

    return {"frequencies": freq, "unobserved": unobserved}