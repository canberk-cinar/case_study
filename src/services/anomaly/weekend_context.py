"""Case 6, Item 2 — weekend adjustment.

Confound check done first (see notebook): is a weekend effect real, or just business_context.py's
hour-of-day effect resurfacing under a different name? Restricted to 09:00-18:00 only — weekday
and weekend transactions in the exact same hour window — weekend fraud rate is still higher
(3.16% vs 2.84%). A genuine, independent effect, not an artifact of the business-hours adjustment,
so it earns its own treatment here rather than being folded into business_context.py.

Same two-principle comparison as business hours (confidence-based, label-free vs fraud-rate-
calibrated, uses isFraud as the same kind of explicit, marked exception) — but the underlying
signal is much weaker here: weekend volume is ~43% of weekday volume (not the ~15x swing
hour_of_day showed), and weekend fraud rate is only modestly higher (3.65% vs 3.43%, a ~6%
relative difference vs. business hours' ~30%). The resulting multipliers are intentionally mild —
inflating them to look as dramatic as the hour-of-day ones would misrepresent what the data
actually shows.

Reuses business_context.py's EXTREME_SCORE_PERCENTILE / MIN_CONFIDENCE_MULTIPLIER rather than
redefining them, so "what counts as extreme" and "how far confidence can drop" stay one shared
policy across every context dimension, not one accidentally-different threshold per file. Operates
on the same final_raw_anomaly_score input as business_context.py (not chained on top of its
output) — each context dimension is applied and compared independently.

A second, finer-grained evaluation is added below the first: is_weekend_proxy alone (used by the
two functions above) treats every weekend hour identically, but a per-hour breakdown shows the
weekend volume itself is not flat — it plateaus at 14:00-01:59 and troughs the rest of the day.
Splitting weekend into a peak window (14:00-01:59) vs the rest shows fraud rate is driven mostly
by that same low-volume/high-volume split, and the "weekend effect" concentrates almost entirely
in weekend's own off-peak hours (+0.80pp there vs +0.09pp within the peak window — see notebook
for the full 2x2 breakdown against weekday's peak/off-peak split). apply_weekend_peak_confidence_adjustment
/ apply_weekend_peak_fraud_calibrated_adjustment implement this finer split, scoped to weekend rows
only (weekday multiplier stays 1.0 — weekday's own hour-based confidence is already
business_context.py's job, not this file's).
"""
import numpy as np
import pandas as pd

from src.services.anomaly.business_context import EXTREME_SCORE_PERCENTILE, MIN_CONFIDENCE_MULTIPLIER

# Weekend's own volume plateau, from the per-hour breakdown: 14:00 through 01:59 (wraps past
# midnight) is a sustained high-volume block; the rest of the day (02:00-13:59) is comparatively
# sparse for weekend transactions specifically. Exclusive end, matching BUSINESS_HOUR_END's
# convention.
WEEKEND_PEAK_HOUR_START = 14
WEEKEND_PEAK_HOUR_END = 2


def apply_weekend_confidence_adjustment(scores: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(temporal[["TransactionID", "is_weekend_proxy"]], on="TransactionID")

    weekday_count = int((~df["is_weekend_proxy"]).sum())
    weekend_count = int(df["is_weekend_proxy"].sum())
    relative_volume = weekend_count / weekday_count  # < 1 — weekend is the sparser side

    confidence_multiplier = MIN_CONFIDENCE_MULTIPLIER + (1 - MIN_CONFIDENCE_MULTIPLIER) * relative_volume

    extreme_threshold = df["final_raw_anomaly_score"].quantile(EXTREME_SCORE_PERCENTILE)
    is_extreme = (df["final_raw_anomaly_score"] >= extreme_threshold).to_numpy()

    multiplier = np.where(df["is_weekend_proxy"].to_numpy() & ~is_extreme, confidence_multiplier, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "weekend_adjusted_score_confidence_based": df["final_raw_anomaly_score"] * multiplier,
        "weekend_confidence_multiplier": multiplier,
        "is_weekend": df["is_weekend_proxy"],
    })


def apply_weekend_fraud_calibrated_adjustment(
    scores: pd.DataFrame, temporal: pd.DataFrame, isfraud: pd.DataFrame
) -> pd.DataFrame:
    df = scores.merge(temporal[["TransactionID", "is_weekend_proxy"]], on="TransactionID")
    df = df.merge(isfraud, on="TransactionID")

    weekday_fraud_rate = df.loc[~df["is_weekend_proxy"], "isFraud"].mean()
    weekend_fraud_rate = df.loc[df["is_weekend_proxy"], "isFraud"].mean()
    boost_multiplier = weekend_fraud_rate / weekday_fraud_rate

    multiplier = np.where(df["is_weekend_proxy"].to_numpy(), boost_multiplier, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "weekend_adjusted_score_fraud_calibrated": df["final_raw_anomaly_score"] * multiplier,
        "weekend_fraud_calibration_multiplier": multiplier,
        "is_weekend": df["is_weekend_proxy"],
    })


def is_weekend_peak_hour(hour_of_day: pd.Series, is_weekend: pd.Series) -> pd.Series:
    in_peak_window = (hour_of_day >= WEEKEND_PEAK_HOUR_START) | (hour_of_day < WEEKEND_PEAK_HOUR_END)
    return is_weekend & in_peak_window


def apply_weekend_peak_confidence_adjustment(scores: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    """Label-free. Weekday rows and weekend-peak rows keep multiplier 1.0 (peak weekend hours are
    weekend's own "normal" case, treated like business hours' in-window case); weekend-off-peak
    rows are damped in proportion to how sparse that slice is relative to weekend-peak — the same
    "less data backs this stat" reasoning as the two adjustments above, just applied to the finer
    split instead of the coarse weekend/weekday one.
    """
    df = scores.merge(temporal[["TransactionID", "hour_of_day", "is_weekend_proxy"]], on="TransactionID")

    peak = is_weekend_peak_hour(df["hour_of_day"], df["is_weekend_proxy"])
    weekend_off_peak = df["is_weekend_proxy"] & ~peak

    peak_count = int(peak.sum())
    off_peak_count = int(weekend_off_peak.sum())
    relative_volume = off_peak_count / peak_count  # < 1 — weekend off-peak is the sparser side

    confidence_multiplier = MIN_CONFIDENCE_MULTIPLIER + (1 - MIN_CONFIDENCE_MULTIPLIER) * relative_volume

    extreme_threshold = df["final_raw_anomaly_score"].quantile(EXTREME_SCORE_PERCENTILE)
    is_extreme = (df["final_raw_anomaly_score"] >= extreme_threshold).to_numpy()

    multiplier = np.where(weekend_off_peak.to_numpy() & ~is_extreme, confidence_multiplier, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "weekend_peak_adjusted_score_confidence_based": df["final_raw_anomaly_score"] * multiplier,
        "weekend_peak_confidence_multiplier": multiplier,
        "is_weekend_off_peak": weekend_off_peak,
    })


def apply_weekend_peak_fraud_calibrated_adjustment(
    scores: pd.DataFrame, temporal: pd.DataFrame, isfraud: pd.DataFrame
) -> pd.DataFrame:
    """Uses isFraud — the same explicit, marked exception as the other *_fraud_calibrated
    functions in this project. Weekend-off-peak's measured fraud rate is calibrated against
    weekend-peak's, producing a boost multiplier for weekend-off-peak rows only; weekday and
    weekend-peak rows stay at 1.0.
    """
    df = scores.merge(temporal[["TransactionID", "hour_of_day", "is_weekend_proxy"]], on="TransactionID")
    df = df.merge(isfraud, on="TransactionID")

    peak = is_weekend_peak_hour(df["hour_of_day"], df["is_weekend_proxy"])
    weekend_off_peak = df["is_weekend_proxy"] & ~peak

    peak_fraud_rate = df.loc[df["is_weekend_proxy"] & peak, "isFraud"].mean()
    off_peak_fraud_rate = df.loc[weekend_off_peak, "isFraud"].mean()
    boost_multiplier = off_peak_fraud_rate / peak_fraud_rate

    multiplier = np.where(weekend_off_peak.to_numpy(), boost_multiplier, 1.0)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "weekend_peak_adjusted_score_fraud_calibrated": df["final_raw_anomaly_score"] * multiplier,
        "weekend_peak_fraud_calibration_multiplier": multiplier,
        "is_weekend_off_peak": weekend_off_peak,
    })
