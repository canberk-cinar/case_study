"""Case 1 — missing-data analysis over the merged transaction+identity Parquet file.

Reads via PyArrow's columnar Parquet API directly, never materializing the full 434-column,
590k-row frame in pandas — an Adapter over the storage layer, in the same spirit as
services/readers.py's planned DataReader: analysis code gets plain arrays/DataFrames back, but the
underlying access is column-batched (or metadata-only) to keep peak memory bounded regardless of
column count.
"""
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import settings

logger = logging.getLogger(__name__)

# Per-column missing-ratio buckets — separates "clean", "worth imputing", "worth flagging", and
# "candidate to drop" columns at a glance.
MISSING_RATIO_BUCKETS = [
    (0.0, "none"),
    (0.05, "low"),       # <5%
    (0.30, "moderate"),  # 5-30%
    (0.70, "high"),      # 30-70%
    (1.0, "very_high"),  # 70-100%
]

# Known from the Faz A ground-truth check: identity covers 24.4239% of transactions, so ~75.5761%
# of rows have no identity match. A missingness pattern group whose ratio lands near this value is
# a strong candidate for "structural" (join-driven) missingness rather than a genuine data gap.
IDENTITY_NON_COVERAGE_RATIO = 1 - 0.244239
STRUCTURAL_RATIO_TOLERANCE = 0.01


def _bucket(ratio: float) -> str:
    for threshold, label in MISSING_RATIO_BUCKETS:
        if ratio <= threshold:
            return label
    return "very_high"


def compute_missing_ratios(parquet_path: Path) -> pd.DataFrame:
    """Per-column null count/ratio, read straight from Parquet's row-group statistics — no row
    data is loaded, so this stays cheap regardless of the file's width."""
    pf = pq.ParquetFile(parquet_path)
    num_rows = pf.metadata.num_rows
    null_counts = {name: 0 for name in pf.schema_arrow.names}

    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.statistics is not None and col.statistics.null_count is not None:
                null_counts[col.path_in_schema] += col.statistics.null_count

    rows = [
        {
            "column": name,
            "null_count": null_count,
            "row_count": num_rows,
            "missing_ratio": null_count / num_rows,
            "bucket": _bucket(null_count / num_rows),
        }
        for name, null_count in null_counts.items()
    ]
    return pd.DataFrame(rows).sort_values("missing_ratio", ascending=False).reset_index(drop=True)


def analyze_missing_patterns(parquet_path: Path, batch_size: int = 25) -> tuple[pd.DataFrame, np.ndarray]:
    """Single batched pass over all columns (batch_size at a time, not all 434 at once): hashes
    each column's null bitmap to find columns that are null in exactly the same rows — reveals
    structure a flat per-column ratio table can't, e.g. blocks of V-columns always missing
    together. Accumulates a per-row null count along the way so the file is only read once."""
    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names
    num_rows = pf.metadata.num_rows

    row_null_counts = np.zeros(num_rows, dtype=np.int32)
    signatures: dict[str, str] = {}
    null_counts: dict[str, int] = {}

    for i in range(0, len(columns), batch_size):
        batch_cols = columns[i : i + batch_size]
        table = pf.read(columns=batch_cols)
        for name in batch_cols:
            null_mask = table.column(name).is_null().to_numpy(zero_copy_only=False)
            row_null_counts += null_mask
            null_counts[name] = int(null_mask.sum())
            signatures[name] = hashlib.blake2b(
                np.packbits(null_mask).tobytes(), digest_size=16
            ).hexdigest()

    groups: dict[str, list[str]] = {}
    for name, sig in signatures.items():
        groups.setdefault(sig, []).append(name)

    pattern_rows = []
    for sig, cols in groups.items():
        if len(cols) < 2:
            continue  # a "pattern" needs at least 2 columns sharing it to be worth reporting
        ratio = null_counts[cols[0]] / num_rows
        pattern_rows.append({
            "signature": sig[:8],
            "column_count": len(cols),
            "columns": cols,
            "missing_ratio": ratio,
            "likely_structural": abs(ratio - IDENTITY_NON_COVERAGE_RATIO) < STRUCTURAL_RATIO_TOLERANCE,
        })

    patterns_df = (
        pd.DataFrame(pattern_rows)
        .sort_values("column_count", ascending=False)
        .reset_index(drop=True)
    )
    return patterns_df, row_null_counts


def summarize_row_missingness(row_null_counts: np.ndarray) -> dict:
    return {
        "mean_missing_per_row": float(row_null_counts.mean()),
        "median_missing_per_row": float(np.median(row_null_counts)),
        "max_missing_per_row": int(row_null_counts.max()),
        "rows_with_zero_missing": int((row_null_counts == 0).sum()),
        "rows_with_zero_missing_pct": float((row_null_counts == 0).mean() * 100),
    }


def run() -> None:
    parquet_path = settings.processed_data_path / "merged_transactions.parquet"

    ratios = compute_missing_ratios(parquet_path)
    patterns, row_null_counts = analyze_missing_patterns(parquet_path)
    row_summary = summarize_row_missingness(row_null_counts)

    print("=== Kolon başına eksik oranı kovaları ===")
    print(ratios["bucket"].value_counts().to_string())

    print("\n=== En yüksek eksik oranına sahip 15 kolon ===")
    print(ratios.head(15).to_string(index=False))

    print(f"\n=== Eksiklik deseni grupları ({len(patterns)} grup, 2+ kolon paylaşıyor) ===")
    for _, row in patterns.iterrows():
        marker = "YAPISAL (join kaynaklı)" if row["likely_structural"] else "incelenmeli"
        sample = row["columns"][:5]
        more = "..." if row["column_count"] > 5 else ""
        print(f"[{marker}] {row['column_count']} kolon, %{row['missing_ratio']*100:.2f} eksik: {sample}{more}")

    print("\n=== Satır bazında eksiklik özeti ===")
    for key, value in row_summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    from src.config import configure_logging

    configure_logging()
    run()