from sqlalchemy import Boolean, Column, Float, Integer, String

from ..db import Base


class ScoredTransaction(Base):
    """Precomputed per-transaction scoring artifact — the API's read model.

    Case 1-9 recompute everything from merged_transactions.parquet on every call, which is right
    for notebooks (full-table analytical scans, memory-bounded columnar reads) but wrong for an
    API: serving one transaction took ~27s because a single point lookup triggered a full
    recomputation of every feature and anomaly layer.

    This table stores exactly the columns the rule engine's conditions (fraud_rules.yaml) and the
    API reference, keyed by TransactionID as the primary key — SQLite's B-tree index turns that
    point lookup into microseconds without loading 590k rows. Parquet stays the right home for
    merged_transactions.parquet (scan-heavy, columnar); this is the opposite access pattern
    (single-row lookup), so it gets the opposite store. Rebuild with
    `python -m src.pipelines.build_scoring_artifact` whenever scoring/feature logic changes.
    """

    __tablename__ = "scored_transactions"

    TransactionID = Column(Integer, primary_key=True, index=True)

    # Raw fields fraud_rules.yaml conditions reference
    TransactionAmt = Column(Float, nullable=True)
    addr2 = Column(Float, nullable=True)
    DeviceInfo = Column(String, nullable=True)
    dist1 = Column(Float, nullable=True)

    # Case 3 temporal features
    hour_of_day = Column(Integer, nullable=True)
    is_low_volume_hour = Column(Boolean, nullable=True)
    is_weekend_proxy = Column(Boolean, nullable=True)

    # Case 3 entity features
    user_transaction_count_so_far = Column(Integer, nullable=True)
    user_amount_zscore = Column(Float, nullable=True)
    user_seconds_since_last_transaction = Column(Float, nullable=True)

    # Case 3 relational features
    is_new_addr1_for_card = Column(Boolean, nullable=True)
    is_new_device_for_card = Column(Boolean, nullable=True)

    # Case 5 terminal output
    final_raw_anomaly_score = Column(Float, nullable=False, index=True)