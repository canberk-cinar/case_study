"""Case 5: weighted aggregation of the five normalized anomaly scores into one final risk score.

Weights are NOT tuned against isFraud: nothing in this pipeline has looked at the label to make a
design decision, and this step doesn't start now. Two weighting schemes are computed and compared
rather than picking one on assertion alone (the same "compare, don't just pick one" instinct as
quality.py's IQR vs MAD and normalization.py's rank vs min-max):

  - Equal weights (1/5 each): the most defensible default for a genuinely unsupervised ensemble,
    since with no labeled ground truth to justify trusting one heuristic more than another,
    treating all five perspectives as equally credible is the honest baseline.
  - Redundancy-adjusted weights: Case 4/5's own Spearman correlation matrix (see combined.py /
    normalization.py's notebook) showed column_anomaly and the two multivariate scores are
    strongly correlated with each other (0.70-0.75): largely re-measuring overlapping
    information, since multivariate's 3-column feature space is a subset of what column_anomaly
    already scores: while entity_anomaly and temporal_anomaly are comparatively independent
    (0.14-0.30 with everything else). A score highly correlated with the others contributes less
    NEW information to an aggregate; weighting it down (and weighting genuinely independent scores
    up) is a standard ensemble-weighting rationale: it uses only the scores' statistical
    relationship to each other, never the label.

Aggregation operates on the rank-normalized columns from normalization.py (not min-max: that
module already showed why raw/min-max scales would let one layer's outliers dominate the sum).
"""
import numpy as np
import pandas as pd

RANK_SUFFIX = "_rank_normalized"

# The dataset's own top-1% cutoff on final_raw_anomaly_score: first established in Case 9's
# anomaly_scoring agent, and reused verbatim by fraud_r07 (fraud_rules.yaml) and the /score route
# so there is exactly one "elevated" threshold in the whole project, defined next to the score it
# thresholds rather than in a consumer of that score.
RISK_THRESHOLD = 0.9410418735356755


def compute_equal_weights(columns: list[str]) -> dict[str, float]:
    return {col: 1.0 / len(columns) for col in columns}


def compute_redundancy_adjusted_weights(normalized: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Weight inversely to how correlated each score is with the other four: a proxy for how
    much unique information it contributes to the aggregate."""
    rank_cols = [f"{col}{RANK_SUFFIX}" for col in columns]
    corr = normalized[rank_cols].corr(method="spearman").abs()
    off_diagonal = corr.where(~np.eye(len(corr), dtype=bool))  # mask self-correlation without mutating in place
    mean_other_corr = off_diagonal.mean(axis=1)

    inverse = 1.0 / (1.0 + mean_other_corr)
    weights = inverse / inverse.sum()
    return dict(zip(columns, weights.to_numpy()))


def apply_weighted_aggregation(normalized: pd.DataFrame, columns: list[str], weights: dict[str, float]) -> pd.Series:
    return sum(weights[col] * normalized[f"{col}{RANK_SUFFIX}"] for col in columns)


def compute_final_risk_scores(normalized: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    equal_weights = compute_equal_weights(columns)
    redundancy_weights = compute_redundancy_adjusted_weights(normalized, columns)

    return pd.DataFrame({
        "TransactionID": normalized["TransactionID"],
        "final_risk_score_equal_weight": apply_weighted_aggregation(normalized, columns, equal_weights),
        "final_risk_score_redundancy_weighted": apply_weighted_aggregation(normalized, columns, redundancy_weights),
    })


def compute_final_raw_anomaly_score(normalized: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The pipeline's single terminal output: one column, one number per transaction, no further
    scaling or bucketing applied (still a continuous "raw" score, not a percentile/tier/alert
    flag: that calibration decision belongs to whatever consumes this score, not to this step).

    Uses redundancy-adjusted weights, not equal weights, as the chosen final scheme: both were
    computed and compared in the notebook (the resulting rankings agreed on 96.7% of the top 1%),
    so the choice between them is low-stakes, but redundancy-adjusted is preferred on principle:
    its weights come only from how correlated the five scores are with each other, never from
    isFraud, and it avoids letting the two correlated column/multivariate signals dominate the sum
    simply because there happen to be more of them.
    """
    redundancy_weights = compute_redundancy_adjusted_weights(normalized, columns)
    return pd.DataFrame({
        "TransactionID": normalized["TransactionID"],
        "final_raw_anomaly_score": apply_weighted_aggregation(normalized, columns, redundancy_weights),
    })
