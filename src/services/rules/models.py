"""Case 7 — data model for a configurable fraud rule, structured explicitly as if/then.

A Rule pairs an `if` (a Condition tree — see conditions.py, Composite pattern: AllOf/AnyOf/Not
nesting of atomic field/operator/value checks) with a `then` consequence: an Action (what the
system should DO — BLOCK/REVIEW/FLAG/ALLOW), a Severity (the axis Chain of Responsibility
resolution, resolution.py, acts on), and a description template. `priority` orders
evaluation/listing and breaks ties between rules of equal severity.

`action` and `severity` are kept as separate fields on purpose: severity is "how dangerous is
this," action is "what do we DO about it" — two rules can share a severity but warrant different
actions (a real fraud-ops system distinguishes "block this transaction" from "just log it for
later review" even at the same risk tier), and collapsing them into one field would hide that.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum

from src.services.rules.conditions import Condition


class Severity(IntEnum):
    """IntEnum so severities compare/sort naturally — CRITICAL is the highest-priority verdict in
    resolution.py's Chain of Responsibility."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Action(Enum):
    """What the system does when a rule's `if` fires — the "then" side of if-then evaluation.
    ALLOW exists for schema completeness/realism (a real rule engine has allow-list rules too)
    even though none of this project's 10 fraud-risk rules use it — every rule here is a risk
    flag, not an allow-list entry, and that's stated rather than silently omitted."""

    BLOCK = "block"
    REVIEW = "review"
    FLAG = "flag"
    ALLOW = "allow"


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    priority: int
    condition: Condition  # the "if"
    action: Action  # the "then" — what to do
    severity: Severity  # the "then" — how dangerous
    description: str  # the "then" — human-readable explanation template
    enabled: bool = True
