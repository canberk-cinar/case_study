"""Case 7 — Chain of Responsibility: reduces a ROW's set of already-fired rules down to one final
verdict. This is deliberately the ONLY place in the engine that short-circuits — rule evaluation
itself (engine.py) stays exhaustive over every rule so explainability can report everything that
fired; only the final-verdict step chains through handlers and stops at the first one that
matches, the classic CoR shape.

Each handler asks one question — "is there a fired rule of exactly this severity?" — and either
returns a Verdict (stopping the chain) or passes to the next handler. Order is fixed
CRITICAL -> HIGH -> MEDIUM -> LOW -> none-fired, mirroring Severity's own ordering. When several
fired rules share the winning severity, `priority` (lower = takes precedence) breaks the tie —
keeping "priority" and "severity" doing two visibly different jobs, as the case brief asks for.

`resolve_by_priority` is a second, deliberately simpler resolution strategy, kept alongside the
severity-driven Chain of Responsibility for comparison rather than replacing it — the same
"compare, don't just pick one" instinct used throughout this project (Case 4's Mahalanobis vs
Isolation Forest, Case 5's equal vs redundancy-adjusted weights, Case 6's every context
adjustment). It ignores severity entirely and picks whichever fired rule has the lowest priority
number, full stop — the notebook measures how often this agrees or disagrees with the CoR verdict,
which is what makes "priority" a real, independently-testable mechanism rather than just a tie-
breaker nobody can see the effect of.
"""
from dataclasses import dataclass

from src.services.rules.domain.models import Rule, Severity


@dataclass(frozen=True)
class Verdict:
    severity: Severity | None
    primary_rule: Rule | None
    fired_rules: list[Rule]


class SeverityHandler:
    def __init__(self, severity: Severity, next_handler: "SeverityHandler | None" = None):
        self.severity = severity
        self.next_handler = next_handler

    def handle(self, fired_rules: list[Rule]) -> Verdict:
        at_this_severity = [r for r in fired_rules if r.severity == self.severity]
        if at_this_severity:
            primary = min(at_this_severity, key=lambda r: r.priority)
            return Verdict(severity=self.severity, primary_rule=primary, fired_rules=fired_rules)
        if self.next_handler is not None:
            return self.next_handler.handle(fired_rules)
        return Verdict(severity=None, primary_rule=None, fired_rules=fired_rules)


def build_default_resolution_chain() -> SeverityHandler:
    low = SeverityHandler(Severity.LOW, next_handler=None)
    medium = SeverityHandler(Severity.MEDIUM, next_handler=low)
    high = SeverityHandler(Severity.HIGH, next_handler=medium)
    critical = SeverityHandler(Severity.CRITICAL, next_handler=high)
    return critical


def resolve_by_priority(fired_rules: list[Rule]) -> Rule | None:
    """Alternative, non-CoR resolution: the single fired rule with the lowest priority number,
    severity ignored entirely. Not used by RuleEngine's default verdict — exists so the notebook
    can compare it against the severity-driven Chain of Responsibility verdict."""
    return min(fired_rules, key=lambda r: r.priority) if fired_rules else None
