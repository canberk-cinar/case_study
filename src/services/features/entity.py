"""Case 3 — entity (card1) features for anomaly-detection models.

Two families of amount aggregation, computed side by side and kept explicitly separate:

  - Safe / causal ("_so_far") — uses only transactions strictly before the current one for the
    same entity, ordered by TransactionDT. This is what a model can actually see at scoring time
    in production, so this is the only family safe to feed into a model.
  - Reference-only ("_global") — the naive aggregate over ALL of an entity's transactions, past
    AND future. This leaks information a production model would never have (a transaction's own
    future isn't known when it's scored). Computed here purely so the notebook can show, with
    real numbers, how far the leaky version drifts from the safe one. LEAKAGE_UNSAFE_COLUMNS
    names exactly which output columns must never be used as model input.

card1 is the entity, consistent with Case 2's entity_behavior.py. isFraud is not touched anywhere
in this module — these are pure transaction-history features, no label involved.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ENTITY_COLUMN = "card1"
SOURCE_COLUMNS = [ENTITY_COLUMN, "TransactionID", "TransactionAmt", "TransactionDT"]

# Output columns that must never be used as model input — computed only for the notebook's
# leakage-comparison demonstration.
LEAKAGE_UNSAFE_COLUMNS = {"user_avg_amount_global"}


def build_entity_features(parquet_path: Path) -> pd.DataFrame:
    df = pq.ParquetFile(parquet_path).read(columns=SOURCE_COLUMNS).to_pandas()
    df = df.sort_values([ENTITY_COLUMN, "TransactionDT"]).reset_index(drop=True)

    grouped_amt = df.groupby(ENTITY_COLUMN)["TransactionAmt"]

    # Inclusive expanding sum/count (includes the current row); subtracting the current row's own
    # contribution gives the "so far" (prior-only) sum/count without a second groupby pass.
    cumsum_incl = grouped_amt.cumsum()
    cumcount_incl = grouped_amt.cumcount() + 1

    prior_count = cumcount_incl - 1
    prior_sum = cumsum_incl - df["TransactionAmt"]
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_avg = np.where(prior_count > 0, prior_sum / prior_count, np.nan)

    # Std can't be derived by simple subtraction — compute inclusive expanding std, then shift by
    # one row within each group to exclude the current transaction.
    expanding_std_incl = grouped_amt.expanding().std().reset_index(level=0, drop=True)
    prior_std = expanding_std_incl.groupby(df[ENTITY_COLUMN]).shift(1).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(
            (prior_count.to_numpy() > 0) & (prior_std > 0),
            (df["TransactionAmt"].to_numpy() - prior_avg) / prior_std,
            np.nan,
        )

    seconds_since_last = df.groupby(ENTITY_COLUMN)["TransactionDT"].diff().to_numpy()

    # Reference-only, leaky global average — see LEAKAGE_UNSAFE_COLUMNS / module docstring.
    global_avg = grouped_amt.transform("mean").to_numpy()

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "user_transaction_count_so_far": prior_count.astype("int64"),
        "user_avg_amount_so_far": prior_avg,
        "user_std_amount_so_far": prior_std,
        "user_amount_zscore": zscore,
        "user_seconds_since_last_transaction": seconds_since_last,
        "user_avg_amount_global": global_avg,  # LEAKAGE_UNSAFE — comparison only, never a model input
    })
