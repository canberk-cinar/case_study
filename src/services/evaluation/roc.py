"""Shared descriptive-validation utility, used across Case 4 (each anomaly layer on its own),
Case 5 (combined score against each individual layer), Case 6 (each business-context adjustment
before and after), and Case 7 (rule severity as an ordinal proxy score).

ROC-AUC is a threshold-free discrimination measure: it answers "does this score rank fraud above
non-fraud, across every possible cutoff", without committing to any one bucket boundary. This
project's decile/quantile fraud-rate tables already answer a version of that question, but only at
a handful of fixed cutoffs and with bucket-boundary noise baked in (several of them are explicitly
described as "monotonic but noisy" in the Case 4/6 notebooks). A single AUC number is directly
comparable across scores and free of that noise.

`isFraud` is read here only to compute a descriptive statistic after a score has already been
frozen. It never feeds back into fitting, weighting, or thresholding that score: the same
discipline this project has followed since Case 1, now formalized with the same metric the
IEEE-CIS Kaggle competition itself was scored on.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(scores: pd.Series, labels: pd.Series) -> float:
    """0.5 = no better than random ranking, 1.0 = perfect ranking. NaN scores are dropped first
    (roc_auc_score cannot handle them), matching how the existing decile tables already exclude
    rows a given layer couldn't score."""
    valid = scores.notna()
    return roc_auc_score(labels[valid], scores[valid])


def compute_roc_points(scores: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Full curve (false positive rate, true positive rate, threshold) for plotting, not just the
    single summary AUC number."""
    valid = scores.notna()
    fpr, tpr, thresholds = roc_curve(labels[valid], scores[valid])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def compare_auc(df: pd.DataFrame, score_columns: list[str], label_column: str = "isFraud") -> pd.DataFrame:
    """One AUC per candidate score column, ranked highest first: the direct way to answer "which
    of these scores discriminates fraud best", used in Case 4 (four layers) and Case 5 (combined
    score against each layer it was built from)."""
    rows = [
        {"score_column": column, "auc": compute_auc(df[column], df[label_column])}
        for column in score_columns
    ]
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
