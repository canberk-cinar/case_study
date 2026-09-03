"""Case 6, Item 4 — geographic risk adjustment.

Empirical check first (see notebook): `addr2` (billing region/country code) is ~99.2% a single
value (87 — the dominant/home region in this dataset), making "foreign" (addr2 present and != 87)
a natural, data-grounded proxy for cross-border risk. Unlike every prior Case 6 signal (business
hours, weekend, trusted entity), this one is the STRONGEST measured effect in the whole project —
and, for once, in the direction naive intuition would expect: foreign fraud rate 10.22% vs
domestic 2.40% (4.26x). A third category, addr2 MISSING (11.13% of rows), turns out to carry an
even higher measured fraud rate than "foreign" itself (11.78%, 4.91x) — missing geographic data is
its own, stronger signal, not noise to be folded into "foreign."

Two other candidate geographic signals were checked and explicitly rejected, not silently dropped:
  - `dist1` (a distance feature): weak, non-monotonic across buckets — not used.
  - `P_emaildomain` country-code-style TLDs (uk/de/fr/mx/es/jp): the fraud rate for these is
    actually slightly LOWER than generic domains (3.11% vs 3.61%) — the opposite of what a
    "foreign email = risk" intuition would predict. Not used.

Two deliberately contrasted principles, the same two-function pattern as business_context.py /
weekend_context.py / trusted_entity_context.py:
  - apply_geographic_policy_adjustment (label-free, naive/operational policy): a fixed boost for
    KNOWN foreign transactions only (addr2 present and != 87) — uses
    1 / MIN_CONFIDENCE_MULTIPLIER (business_context.py's existing floor, inverted, rather than a
    newly-invented number). Missing addr2 is left untouched (multiplier=1.0): a genuinely
    label-free policy has no basis to treat "unknown" as "risky" — it can only act on what it
    positively knows. This is a deliberate choice, not an oversight, and it's exactly what exposes
    the gap the fraud-calibrated function below reveals: missing data turns out to be the single
    highest-risk category measured in this whole project, and the naive policy misses it entirely.
  - apply_geographic_fraud_calibrated_adjustment (uses isFraud — the same explicit, marked
    exception as every other *_fraud_calibrated function here): three empirically-measured
    multipliers (domestic=1.0, foreign≈4.26, missing≈4.91), not the two-tier structure the label-
    free function uses.

Neither function uses business_context.py's EXTREME_SCORE_PERCENTILE exemption — that exemption
exists to protect an already-extreme score from being DAMPED down by a low-confidence context; both
functions here BOOST, never damp, matching the pattern already set by every other *_fraud_calibrated
function in this project (none of which use the exemption either — only the damping/confidence-
based functions do).
"""
import numpy as np
import pandas as pd

from src.services.anomaly.business_context import MIN_CONFIDENCE_MULTIPLIER

DOMESTIC_ADDR2 = 87.0
FOREIGN_POLICY_BOOST_MULTIPLIER = 1 / MIN_CONFIDENCE_MULTIPLIER  # 2.0


def classify_geography(addr2: pd.Series) -> pd.Series:
    return np.select(
        [addr2.isna(), addr2 == DOMESTIC_ADDR2],
        ["missing", "domestic"],
        default="foreign",
    )


def apply_geographic_policy_adjustment(scores: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(raw[["TransactionID", "addr2"]], on="TransactionID")
    geography = classify_geography(df["addr2"])

    multiplier = np.where(geography == "foreign", FOREIGN_POLICY_BOOST_MULTIPLIER, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "geographic_adjusted_score_policy": df["final_raw_anomaly_score"] * multiplier,
        "geographic_policy_multiplier": multiplier,
        "geography": geography,
    })


def apply_geographic_fraud_calibrated_adjustment(
    scores: pd.DataFrame, raw: pd.DataFrame, isfraud: pd.DataFrame
) -> pd.DataFrame:
    df = scores.merge(raw[["TransactionID", "addr2"]], on="TransactionID")
    df = df.merge(isfraud, on="TransactionID")
    geography = classify_geography(df["addr2"])

    domestic_rate = df.loc[geography == "domestic", "isFraud"].mean()
    foreign_rate = df.loc[geography == "foreign", "isFraud"].mean()
    missing_rate = df.loc[geography == "missing", "isFraud"].mean()

    foreign_multiplier = foreign_rate / domestic_rate
    missing_multiplier = missing_rate / domestic_rate

    multiplier = np.select(
        [geography == "foreign", geography == "missing"],
        [foreign_multiplier, missing_multiplier],
        default=1.0,
    )

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "geographic_adjusted_score_fraud_calibrated": df["final_raw_anomaly_score"] * multiplier,
        "geographic_fraud_calibration_multiplier": multiplier,
        "geography": geography,
    })
