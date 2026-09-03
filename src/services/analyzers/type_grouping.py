"""Case 2 — top-level numeric / categorical / datetime grouping.

column_types.py's semantic_type is deliberately fine-grained (11 classes) because Case 1 needed
per-class treatment (encoding hints, log-transform candidates, etc.). Case 2 asks for a simpler,
top-level split instead. Rather than re-deriving it from raw stats, this rolls the existing
semantic_type up into three buckets — the classification logic lives in one place
(column_types.py), this is just a different lens on the same result.

The one judgment call: numeric_encoded_categorical is int/float-typed but was already flagged in
Case 1 as "not safe to average" (card1, addr1, ...) — it goes to `categorical` here, not
`numeric`, because that's what the label means. constant/near_constant columns don't carry
semantic_type's numeric/categorical distinction (a column that's always the same value has no
"kind" from statistics alone), so they fall back to physical dtype.
"""
import pandas as pd

DATETIME_SEMANTIC_TYPES = {"timestamp_offset"}
EXCLUDED_SEMANTIC_TYPES = {"identifier", "target"}  # not a feature column for this split
CATEGORICAL_SEMANTIC_TYPES = {"categorical", "binary_flag", "high_cardinality_text", "numeric_encoded_categorical"}
NUMERIC_SEMANTIC_TYPES = {"numeric_continuous", "monetary", "count"}


def classify_high_level_type(row: pd.Series) -> str:
    semantic_type = row["semantic_type"]
    if semantic_type in DATETIME_SEMANTIC_TYPES:
        return "datetime"
    if semantic_type in EXCLUDED_SEMANTIC_TYPES:
        return "excluded"
    if semantic_type in CATEGORICAL_SEMANTIC_TYPES:
        return "categorical"
    if semantic_type in NUMERIC_SEMANTIC_TYPES:
        return "numeric"
    # constant / near_constant: no numeric/categorical signal in semantic_type, use physical dtype
    is_numeric_dtype = str(row["physical_dtype"]).startswith(("int", "float"))
    return "numeric" if is_numeric_dtype else "categorical"


def group_columns_by_type(classified: pd.DataFrame) -> pd.DataFrame:
    result = classified.copy()
    result["high_level_type"] = result.apply(classify_high_level_type, axis=1)
    return result