"""Case 6, Madde 1 — business hours context: reweights Case 5's final_raw_anomaly_score based on
whether a transaction occurred during standard business hours, using two DIFFERENT and
DELIBERATELY CONTRASTED principles rather than picking one:

  - apply_confidence_based_adjustment (label-free): off-hours transactions rest on statistics
    computed from far fewer observations (Case 1 found hourly volume varies ~15x; Case 4
    independently found, in three separate layers, that a causal/segment statistic built on
    shallow history is noisier and more likely to produce spurious extreme scores). A raw score
    that's high mainly because it landed in a low-volume hour is exactly the shape of a false
    positive this step is meant to reduce, so off-hours scores are damped in proportion to how
    little data backs that hour — unless the raw score is already in the extreme tail (top 1%),
    which is treated as genuinely extreme regardless of when it happened.
  - apply_fraud_rate_calibrated_adjustment (uses isFraud — a deliberate, explicitly marked
    exception): Case 1 found the *opposite* real-world pattern in this dataset — fraud rate is
    highest exactly in the low-volume hours the first function dampens. Calibrating the
    adjustment against that measured rate produces the mirror-image correction: off-hours scores
    are boosted, not damped. This is the first place in the whole project where isFraud
    parameterizes a design choice rather than only checking one after the fact — flagged loudly
    here and in the notebook, not quietly slipped in. A real deployment would need this kind of
    label-calibrated rule to go through its own held-out validation before use; that's out of
    scope here.

Both operate on the same BUSINESS_HOURS definition (weekday, 09:00-18:00) and the same
final_raw_anomaly_score input, so the two outputs are directly comparable.
"""
import numpy as np
import pandas as pd

BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18  # exclusive

# Raw scores at or above this percentile are treated as genuinely extreme regardless of hour —
# not damped even off-hours, so a real outlier can't be waved away just because it's 4am.
EXTREME_SCORE_PERCENTILE = 0.99

# Even the lowest-volume hour's confidence multiplier never drops below this — an off-hours
# anomaly is still worth some attention, just less confidently, never zeroed out entirely.
MIN_CONFIDENCE_MULTIPLIER = 0.5


def is_business_hours(hour_of_day: pd.Series, is_weekend: pd.Series) -> pd.Series:
    return (~is_weekend) & (hour_of_day >= BUSINESS_HOUR_START) & (hour_of_day < BUSINESS_HOUR_END)


def apply_confidence_based_adjustment(scores: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(temporal[["TransactionID", "hour_of_day", "is_weekend_proxy"]], on="TransactionID")

    hourly_volume = df["hour_of_day"].value_counts()
    relative_volume = (df["hour_of_day"].map(hourly_volume) / hourly_volume.max()).to_numpy()

    business = is_business_hours(df["hour_of_day"], df["is_weekend_proxy"]).to_numpy()
    confidence_multiplier = np.where(
        business,
        1.0,
        MIN_CONFIDENCE_MULTIPLIER + (1 - MIN_CONFIDENCE_MULTIPLIER) * relative_volume,
    )

    extreme_threshold = df["final_raw_anomaly_score"].quantile(EXTREME_SCORE_PERCENTILE)
    is_extreme = (df["final_raw_anomaly_score"] >= extreme_threshold).to_numpy()
    confidence_multiplier = np.where(is_extreme, 1.0, confidence_multiplier)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "business_adjusted_score_confidence_based": df["final_raw_anomaly_score"] * confidence_multiplier,
        "confidence_multiplier": confidence_multiplier,
        "is_business_hours": business,
    })


def apply_fraud_rate_calibrated_adjustment(
    scores: pd.DataFrame, temporal: pd.DataFrame, isfraud: pd.DataFrame
) -> pd.DataFrame:
    df = scores.merge(temporal[["TransactionID", "hour_of_day", "is_weekend_proxy"]], on="TransactionID")
    df = df.merge(isfraud, on="TransactionID")

    business = is_business_hours(df["hour_of_day"], df["is_weekend_proxy"])
    business_fraud_rate = df.loc[business, "isFraud"].mean()
    off_hours_fraud_rate = df.loc[~business, "isFraud"].mean()
    boost_multiplier = off_hours_fraud_rate / business_fraud_rate

    multiplier = np.where(business.to_numpy(), 1.0, boost_multiplier)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "business_adjusted_score_fraud_calibrated": df["final_raw_anomaly_score"] * multiplier,
        "fraud_calibration_multiplier": multiplier,
        "is_business_hours": business,
    })
