"""Case 5: normalizing the four anomaly layers' scores onto a common [0,1] scale, so they can
later be weighted and combined into one risk score.

Percentile-rank normalization (each score's rank within its own distribution, scaled to [0,1]) is
used as the primary method rather than min-max scaling: Case 4 documented, in every single layer,
extreme outliers dominating the raw score (column_anomaly: max_abs_zscore over 100; entity_anomaly:
max score 8028 against a median under 1; temporal_anomaly: max score 495). Min-max scaling would
let one such outlier compress the entire rest of that layer's distribution into a sliver near 0,
exactly the sensitivity-to-outliers problem Case 1-4 spent considerable effort working around with
robust (MAD-based) statistics in the first place. Percentile rank sidesteps this entirely: it only
cares about ORDERING, not magnitude, so one absurd outlier can't distort where everything else
lands.

Both methods are computed here, side by side (not just rank alone), so the notebook can show
concretely why rank-based was chosen rather than just asserting it: the same "compare, don't just
pick one" instinct used throughout this project (quality.py's IQR vs MAD, multivariate_anomaly.py's
Mahalanobis vs Isolation Forest).
"""
import pandas as pd


def normalize_scores(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Adds a `{col}_rank_normalized` (percentile rank, [0,1]) and `{col}_minmax_normalized`
    ([0,1] linear scaling) column for each input column."""
    result = scores.copy()
    for col in columns:
        result[f"{col}_rank_normalized"] = scores[col].rank(pct=True)

        min_val, max_val = scores[col].min(), scores[col].max()
        span = max_val - min_val
        result[f"{col}_minmax_normalized"] = (scores[col] - min_val) / span if span > 0 else 0.0

    return result
