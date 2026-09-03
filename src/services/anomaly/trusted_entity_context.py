"""Case 6, Madde 3 — trusted entity adjustment.

Empirical check first (see notebook): does a card's (`card1`, Case 3's entity identifier) own
transaction history depth (`user_transaction_count_so_far`, causal — only prior transactions
count) predict a LOWER fraud rate, the way "trust builds with tenure" naive intuition assumes?
**No — the opposite.** Fraud rate is lowest for brand-new/thin-history cards (~2.5% for
count_so_far<=20) and RISES with tenure, peaking around 101-1000 prior transactions (~3.9%),
easing only slightly for 1000+ (~3.66%, still above the low-tenure baseline).

Likely explanation, consistent with a finding already surfaced in Case 1/3: `card1` behaves like a
high-volume "bucket" rather than a uniquely-identifying per-customer field at this scale — a
`card1` value with thousands of prior transactions probably represents many different real cards
sharing one anonymized bucket value, not one loyal customer. "More history" here measures how
generic/shared the bucket is, not how trustworthy any individual cardholder is.

Two deliberately contrasted principles, the same two-function pattern as business_context.py /
weekend_context.py:
  - apply_trust_confidence_adjustment (label-free, the operational "benefit of the doubt"
    intuition): established cards (>=TRUST_THRESHOLD prior transactions) get their score damped —
    this is a genuinely different rationale than the confidence dampening in the other two
    modules, which damp the SPARSE/noisy side. Here the DENSE/established side is what's damped,
    because "trust an entity with a long track record" is what "trusted entity adjustment" means
    as a business rule, independent of whether the underlying statistic is noisier or not. Case
    3/4's own "shallow history is noisier" argument would actually suggest damping the *opposite*
    side — flagged here explicitly, not glossed over, precisely because it sets up the tension the
    fraud-calibrated comparison below is meant to expose.
  - apply_trust_fraud_calibrated_adjustment (uses isFraud — explicit, marked exception): calibrates
    against the direction the data actually shows, which is the reverse of the naive intuition —
    established cards get BOOSTED, not damped.

CAVEAT, surfaced rather than hidden: the relationship is not monotonic across the very top of the
tenure distribution. Restricted to the top 10% by tenure (>=3700 prior transactions), the pattern
partly reverses (fraud rate 3.24% vs 3.53% for everyone else — boost<1 there). TRUST_THRESHOLD=100
sits at the boundary of the bucket carrying most of the anomalous rise and is stable across nearby
thresholds (50-200 all show a 1.23-1.36x boost in the same direction), but it is a policy choice,
not the only defensible one — a different threshold could tell a different story.

Reuses business_context.py's EXTREME_SCORE_PERCENTILE / MIN_CONFIDENCE_MULTIPLIER, the same shared
policy as every other context dimension in Case 6.
"""
import numpy as np
import pandas as pd

from src.services.anomaly.business_context import EXTREME_SCORE_PERCENTILE, MIN_CONFIDENCE_MULTIPLIER

TRUST_THRESHOLD = 100  # prior transactions on the same card1


def is_trusted_entity(transaction_count_so_far: pd.Series) -> pd.Series:
    return transaction_count_so_far >= TRUST_THRESHOLD


def apply_trust_confidence_adjustment(scores: pd.DataFrame, entity: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(entity[["TransactionID", "user_transaction_count_so_far"]], on="TransactionID")
    trusted = is_trusted_entity(df["user_transaction_count_so_far"]).to_numpy()

    extreme_threshold = df["final_raw_anomaly_score"].quantile(EXTREME_SCORE_PERCENTILE)
    is_extreme = (df["final_raw_anomaly_score"] >= extreme_threshold).to_numpy()

    multiplier = np.where(trusted & ~is_extreme, MIN_CONFIDENCE_MULTIPLIER, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "trust_adjusted_score_confidence_based": df["final_raw_anomaly_score"] * multiplier,
        "trust_confidence_multiplier": multiplier,
        "is_trusted_entity": trusted,
    })


def apply_trust_fraud_calibrated_adjustment(
    scores: pd.DataFrame, entity: pd.DataFrame, isfraud: pd.DataFrame
) -> pd.DataFrame:
    df = scores.merge(entity[["TransactionID", "user_transaction_count_so_far"]], on="TransactionID")
    df = df.merge(isfraud, on="TransactionID")

    trusted = is_trusted_entity(df["user_transaction_count_so_far"])
    trusted_fraud_rate = df.loc[trusted, "isFraud"].mean()
    new_fraud_rate = df.loc[~trusted, "isFraud"].mean()
    boost_multiplier = trusted_fraud_rate / new_fraud_rate

    multiplier = np.where(trusted.to_numpy(), boost_multiplier, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "trust_adjusted_score_fraud_calibrated": df["final_raw_anomaly_score"] * multiplier,
        "trust_fraud_calibration_multiplier": multiplier,
        "is_trusted_entity": trusted,
    })
