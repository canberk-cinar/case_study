"""Adım 7 — Composite pattern: a rule's condition is a tree of Condition nodes that all share one
interface (`evaluate(df) -> pd.Series[bool]`), so a caller never needs to know whether it's holding
one atomic field check or an arbitrarily nested AND/OR/NOT combination of them — it just calls
`.evaluate(df)`. AtomicCondition is the leaf (a single field/operator/value check, delegating the
actual comparison to operators.py's Strategy registry); CompositeCondition is the composite branch
(all_of / any_of / not, each recursing into its children's own `.evaluate`).

Every node evaluates over a whole DataFrame at once (Series[bool] out) — there is deliberately no
separate row-wise code path. A single-transaction explanation (engine.py's `explain`) reuses this
exact same vectorized evaluate() by wrapping one row in a 1-row DataFrame, rather than duplicating
condition logic in a second, row-oriented implementation.
"""
from abc import ABC, abstractmethod

import pandas as pd

from src.services.rules.operators import OPERATOR_REGISTRY


class Condition(ABC):
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def describe(self, row: pd.Series) -> str:
        """Human-readable, single-transaction description of this condition using the actual
        values in `row` — used by engine.py's explain() to interpolate real numbers into a fired
        rule's explanation, not just repeat the abstract rule definition."""
        ...


class AtomicCondition(Condition):
    def __init__(self, field: str, operator: str, value: object):
        if operator not in OPERATOR_REGISTRY:
            raise ValueError(f"unknown operator: {operator!r} (known: {sorted(OPERATOR_REGISTRY)})")
        self.field = field
        self.operator = operator
        self.value = value

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return OPERATOR_REGISTRY[self.operator](df[self.field], self.value)

    def describe(self, row: pd.Series) -> str:
        return f"{self.field}={row[self.field]!r} ({self.operator} {self.value!r})"


class CompositeCondition(Condition):
    """all_of == AND (every child must hold), any_of == OR (at least one child must hold)."""

    def __init__(self, kind: str, children: list[Condition]):
        if kind not in ("all_of", "any_of"):
            raise ValueError(f"unknown composite kind: {kind!r}")
        if not children:
            raise ValueError("CompositeCondition requires at least one child condition")
        self.kind = kind
        self.children = children

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        results = [child.evaluate(df) for child in self.children]
        combined = results[0]
        for r in results[1:]:
            combined = (combined & r) if self.kind == "all_of" else (combined | r)
        return combined

    def describe(self, row: pd.Series) -> str:
        joiner = " AND " if self.kind == "all_of" else " OR "
        return "(" + joiner.join(child.describe(row) for child in self.children) + ")"


class NotCondition(Condition):
    def __init__(self, child: Condition):
        self.child = child

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return ~self.child.evaluate(df)

    def describe(self, row: pd.Series) -> str:
        return f"NOT {self.child.describe(row)}"
