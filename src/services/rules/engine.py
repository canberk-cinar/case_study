"""Case 7: RuleEngine: evaluates every enabled rule against a DataFrame at once (vectorized,
Series[bool] per rule: no row-by-row Python loop for the actual condition checks, consistent with
the rest of this project), then reduces each row's fired-rule set to one verdict.

Two verdicts are computed side by side in evaluate_all, not one: `verdict_rule_id`/`verdict_severity`
(the severity-driven Chain of Responsibility result: resolution.py's SeverityHandler) and
`priority_verdict_rule_id` (resolution.py's resolve_by_priority, severity ignored, lowest priority
number wins outright). Both are vectorized the same way (idxmax over priority-ordered boolean
columns). Keeping both, rather than only the CoR one, is what makes "priority" a real, separately-
measurable mechanism: the notebook reports how often the two verdicts agree or disagree, instead
of priority only ever mattering as an invisible tie-break inside the severity verdict.

Two entry points for explainability, one shared machinery (_build_explanation):
  - explain(row): wraps a single transaction in a 1-row DataFrame and calls evaluate_all on it
    (the exact same Condition.evaluate() / severity logic, no separate row-wise reimplementation),
    then builds a human-readable report, and runs the actual object-oriented Chain of
    Responsibility (resolution.py) over the fired Rule objects, so the CoR pattern itself is
    genuinely exercised (not just its vectorized bulk equivalent).
  - explain_batch(df, result): the actual "explainability output" deliverable: a structured list
    of full explanation records for MANY transactions at once, reusing an already-computed
    evaluate_all() result (no re-evaluation of conditions) rather than calling explain() in a loop.
"""
import pandas as pd

from src.services.rules.domain.models import Rule, Severity
from src.services.rules.resolution import SeverityHandler, resolve_by_priority


class RuleEngine:
    def __init__(self, rules: list[Rule], resolution_chain: SeverityHandler):
        self.rules = sorted((r for r in rules if r.enabled), key=lambda r: r.priority)
        self.resolution_chain = resolution_chain
        self._rules_by_id = {r.id: r for r in self.rules}

    def evaluate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        fired = pd.DataFrame(index=df.index)
        for rule in self.rules:
            fired[rule.id] = rule.condition.evaluate(df).to_numpy()

        any_fired_overall = fired.any(axis=1)
        all_ids_by_priority = [r.id for r in self.rules]  # self.rules already priority-sorted
        priority_verdict_rule_id = fired[all_ids_by_priority].idxmax(axis=1).where(any_fired_overall, other=None)

        severity_any: dict[Severity, pd.Series] = {}
        severity_primary: dict[Severity, pd.Series] = {}
        for severity in Severity:
            ids = [r.id for r in self.rules if r.severity == severity]
            if not ids:
                severity_any[severity] = pd.Series(False, index=df.index)
                severity_primary[severity] = pd.Series(None, index=df.index, dtype=object)
                continue
            any_fired = fired[ids].any(axis=1)
            # idxmax over priority-ordered boolean columns returns the first (lowest-priority-
            # number) column that's True: exactly "highest-precedence fired rule at this
            # severity," vectorized. Masked to NaN where nothing fired at this severity at all.
            primary = fired[ids].idxmax(axis=1).where(any_fired, other=None)
            severity_any[severity] = any_fired
            severity_primary[severity] = primary

        verdict_severity = pd.Series(None, index=df.index, dtype=object)
        verdict_rule_id = pd.Series(None, index=df.index, dtype=object)
        for severity in sorted(Severity, reverse=True):  # CRITICAL -> HIGH -> MEDIUM -> LOW
            still_unresolved = verdict_severity.isna()
            newly_resolved = still_unresolved & severity_any[severity]
            verdict_severity = verdict_severity.mask(newly_resolved, severity.name)
            verdict_rule_id = verdict_rule_id.mask(newly_resolved, severity_primary[severity])

        action_by_rule_id = {r.id: r.action.name for r in self.rules}
        verdict_action = verdict_rule_id.map(action_by_rule_id)
        priority_verdict_severity = priority_verdict_rule_id.map({r.id: r.severity.name for r in self.rules})

        result = fired.copy()
        result["fired_rule_count"] = fired.sum(axis=1)
        result["verdict_severity"] = verdict_severity
        result["verdict_action"] = verdict_action
        result["verdict_rule_id"] = verdict_rule_id
        result["priority_verdict_rule_id"] = priority_verdict_rule_id
        result["priority_verdict_severity"] = priority_verdict_severity
        return result

    def _build_explanation(self, row: pd.Series, fired_rule_ids: list[str]) -> dict:
        fired_rules = [self._rules_by_id[rid] for rid in fired_rule_ids]

        cor_verdict = self.resolution_chain.handle(fired_rules)
        priority_rule = resolve_by_priority(fired_rules)

        fired_explanations = [
            {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "severity": rule.severity.name,
                "action": rule.action.name,
                "priority": rule.priority,
                "condition": rule.condition.describe(row),
                "message": rule.description.format(**row.to_dict()),
            }
            for rule in sorted(fired_rules, key=lambda r: r.priority)
        ]

        return {
            "fired_rules": fired_explanations,
            "verdict_severity": cor_verdict.severity.name if cor_verdict.severity is not None else None,
            "verdict_action": cor_verdict.primary_rule.action.name if cor_verdict.primary_rule is not None else None,
            "verdict_rule_id": cor_verdict.primary_rule.id if cor_verdict.primary_rule is not None else None,
            "priority_verdict_rule_id": priority_rule.id if priority_rule is not None else None,
        }

    def explain(self, row: pd.Series) -> dict:
        row_df = row.to_frame().T
        evaluated = self.evaluate_all(row_df).iloc[0]
        fired_rule_ids = [r.id for r in self.rules if evaluated[r.id]]
        return self._build_explanation(row, fired_rule_ids)

    def explain_batch(self, df: pd.DataFrame, result: pd.DataFrame | None = None) -> list[dict]:
        """The batch explainability output: one full explanation record per row in `df`, reusing
        a precomputed evaluate_all() `result` when given (no redundant condition re-evaluation);
        this is the actual deliverable, not the single-row explain() demos."""
        if result is None:
            result = self.evaluate_all(df)

        id_column = "TransactionID" if "TransactionID" in df.columns else None
        records = []
        for idx in df.index:
            fired_rule_ids = [r.id for r in self.rules if result.at[idx, r.id]]
            record = {"TransactionID": df.at[idx, id_column]} if id_column else {}
            record.update(self._build_explanation(df.loc[idx], fired_rule_ids))
            records.append(record)
        return records
