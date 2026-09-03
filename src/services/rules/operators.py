"""Case 7 — Strategy pattern: each comparison operator is an interchangeable, vectorized
(Series-in, Series[bool]-out) algorithm behind one common interface (a callable of
`(field_values: pd.Series, operand) -> pd.Series[bool]`), selected at runtime by the string name
that appears in a rule's YAML/JSON condition (`operator: "gt"`, etc.) — the loader never branches
on operator name itself, it just looks it up in OPERATOR_REGISTRY.

Every function is written to evaluate over a whole DataFrame column at once, consistent with how
the rest of this project (features/, anomaly/) is vectorized pandas rather than row-by-row Python.
"""
from typing import Callable

import pandas as pd

OperatorFn = Callable[[pd.Series, object], pd.Series]


def _gt(values: pd.Series, operand) -> pd.Series:
    return values > operand


def _gte(values: pd.Series, operand) -> pd.Series:
    return values >= operand


def _lt(values: pd.Series, operand) -> pd.Series:
    return values < operand


def _lte(values: pd.Series, operand) -> pd.Series:
    return values <= operand


def _eq(values: pd.Series, operand) -> pd.Series:
    return values == operand


def _ne(values: pd.Series, operand) -> pd.Series:
    return values != operand


def _in(values: pd.Series, operand) -> pd.Series:
    return values.isin(operand)


def _not_in(values: pd.Series, operand) -> pd.Series:
    return ~values.isin(operand)


def _between(values: pd.Series, operand) -> pd.Series:
    low, high = operand
    return (values >= low) & (values <= high)


def _is_null(values: pd.Series, operand) -> pd.Series:
    return values.isna() if operand else values.notna()


OPERATOR_REGISTRY: dict[str, OperatorFn] = {
    "gt": _gt,
    "gte": _gte,
    "lt": _lt,
    "lte": _lte,
    "eq": _eq,
    "ne": _ne,
    "in": _in,
    "not_in": _not_in,
    "between": _between,
    "is_null": _is_null,
}
