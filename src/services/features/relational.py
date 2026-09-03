"""Case 3 — relational features: how a transaction's entity (card1) relates to its context
columns (addr1, DeviceInfo) at the time of the transaction.

Same causal ("_so_far") discipline as entity.py: every feature here reflects only what had
already happened strictly before the current transaction, ordered by TransactionDT — nothing
here looks at rows after the one being scored (see entity.py's module docstring for why).

Two directions of relationship, both classic fraud signals:
  - card -> context ("is this address/device new for this card?"): a card suddenly transacting
    from a brand-new region or device is a common account-takeover signal.
  - context -> card ("how many distinct cards has this address/device been shared by?"): an
    address or device shared across many distinct cards is the classic "drop address" /
    device-emulator fraud-ring pattern — a much stronger signal than looking at the card alone.

The running-distinct-count features (addr1_distinct_cards_so_far, ...) are vectorized via a
"first occurrence" trick rather than a per-row running set: sorted by (context, TransactionDT), a
row is the first time its card1 is seen under that context iff
~duplicated(subset=[context, "card1"], keep="first"); a cumulative sum of that flag within each
context group is exactly the running distinct-card count.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ENTITY_COLUMN = "card1"
CONTEXT_COLUMNS = ["addr1", "DeviceInfo"]
SOURCE_COLUMNS = [ENTITY_COLUMN, "TransactionID", "TransactionDT"] + CONTEXT_COLUMNS


def _running_distinct_entity_count(
    df: pd.DataFrame, context_col: str, entity_col: str, time_col: str
) -> pd.Series:
    """For each row, the number of distinct `entity_col` values seen under this row's
    `context_col` value strictly before this row (causal), ordered by `time_col`. NaN where the
    row itself has no context value — there's nothing to relate it to."""
    ordered = df.sort_values([context_col, time_col])
    has_context = ordered[context_col].notna()

    is_first_for_context = ~ordered.duplicated(subset=[context_col, entity_col], keep="first")
    # duplicated() treats NaN == NaN, which would wrongly lump every missing-context row into one
    # group — force those rows' flag to False so they never contribute to a running count.
    is_first_for_context = is_first_for_context.where(has_context, False)

    running_inclusive = is_first_for_context.groupby(ordered[context_col]).cumsum()
    prior = running_inclusive - is_first_for_context.astype(int)
    prior = prior.where(has_context)

    return prior.reindex(df.index)


def build_relational_features(parquet_path: Path) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=SOURCE_COLUMNS).to_pandas()
    df = df.sort_values([ENTITY_COLUMN, "TransactionDT"]).reset_index(drop=True)

    # card -> context: has this card used this addr1 / this device before?
    addr1_pair_cumcount = df.groupby([ENTITY_COLUMN, "addr1"], dropna=False).cumcount()
    is_new_addr1_for_card = (addr1_pair_cumcount == 0) & df["addr1"].notna()

    device_pair_cumcount = df.groupby([ENTITY_COLUMN, "DeviceInfo"], dropna=False).cumcount()
    is_new_device_for_card = (device_pair_cumcount == 0) & df["DeviceInfo"].notna()

    # context -> card: how many distinct cards has this addr1 / this device been shared by so far?
    addr1_distinct_cards_so_far = _running_distinct_entity_count(df, "addr1", ENTITY_COLUMN, "TransactionDT")
    device_distinct_cards_so_far = _running_distinct_entity_count(df, "DeviceInfo", ENTITY_COLUMN, "TransactionDT")

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "card_addr1_pair_count_so_far": addr1_pair_cumcount,
        "is_new_addr1_for_card": is_new_addr1_for_card,
        "is_new_device_for_card": is_new_device_for_card,
        "addr1_distinct_cards_so_far": addr1_distinct_cards_so_far,
        "device_distinct_cards_so_far": device_distinct_cards_so_far,
    })
