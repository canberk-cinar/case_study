"""Case 4: entity anomaly detection: how anomalous is a transaction relative to the same card1's
OWN established behavior: not the global population (column_anomaly.py) and not a handful of
columns considered jointly (multivariate_anomaly.py), but this one entity's history.

Reuses Case 3's already-built causal ("_so_far") entity/relational features rather than
re-deriving anything: features.entity.build_entity_features gives the amount-deviation signal
(user_amount_zscore), features.relational.build_relational_features gives the behavioral-novelty
signals (is_new_addr1_for_card, is_new_device_for_card). Case 4 turns those feature columns into
one explainable anomaly score for this layer: Case 3 built the inputs, this is what consumes them.

Two components, both computed without ever looking at isFraud:
  - amount component: |user_amount_zscore|: how far this transaction's amount sits from the
    card's own historical average, in the card's own historical spread.
  - novelty component: NEW_ADDR_WEIGHT * is_new_addr1_for_card + NEW_DEVICE_WEIGHT *
    is_new_device_for_card, counted only once the card has SOME established history
    (user_transaction_count_so_far > 0): for a brand-new card, every address/device is trivially
    "new," so novelty is not a meaningful signal on a card's very first transaction. Weights are
    equal (1.0 each) and chosen only to sit on the same rough scale as a typical |z-score|
    (median ~0.43 in this dataset): NOT tuned against isFraud, which stays out of this module
    entirely; see the notebook for a purely descriptive check of how each component relates to
    the label after the fact.

entity_anomaly_history_depth (= user_transaction_count_so_far) is reported explicitly for the same
reason column_anomaly.py reports scored_count: a z-score built on very little prior history is a
noisier estimate than one built on many transactions, and that should stay visible rather than be
hidden inside a single opaque number.
"""
from pathlib import Path

import pandas as pd

from src.services.features.entity import build_entity_features
from src.services.features.relational import build_relational_features

NEW_ADDR_WEIGHT = 1.0
NEW_DEVICE_WEIGHT = 1.0


def compute_entity_anomaly_scores(parquet_path: Path) -> pd.DataFrame:
    entity = build_entity_features(parquet_path)
    relational = build_relational_features(parquet_path)
    df = entity.merge(relational, on="TransactionID")

    has_history = df["user_transaction_count_so_far"] > 0

    amount_component = df["user_amount_zscore"].abs().fillna(0.0)
    novelty_component = (
        NEW_ADDR_WEIGHT * df["is_new_addr1_for_card"].astype(int)
        + NEW_DEVICE_WEIGHT * df["is_new_device_for_card"].astype(int)
    ) * has_history

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "entity_anomaly_score": amount_component + novelty_component,
        "entity_anomaly_history_depth": df["user_transaction_count_so_far"],
        "entity_anomaly_amount_component": amount_component,
        "entity_anomaly_novelty_component": novelty_component,
    })
