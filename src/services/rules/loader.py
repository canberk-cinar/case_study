"""Case 7 — Factory pattern: RuleLoader.load(path) turns a YAML or JSON rules file (dispatched by
file extension) into a list of Rule objects, building each rule's Condition tree recursively via
_build_condition — the one place in this package that knows how to translate the raw dict schema
(all_of/any_of/not/field+operator+value) into Condition instances. Callers never construct
Condition or Rule objects by hand from parsed config; they always go through this factory.

Schema (same shape in YAML and JSON) — explicit if/then, the literal shape of if-then evaluation:
    rules:
      - id: str
        name: str
        priority: int
        enabled: bool                 # optional, defaults to true
        if:
          all_of: [condition, ...]     # or any_of: [...], or not: condition, or a leaf:
          field: str
          operator: gt|gte|lt|lte|eq|ne|in|not_in|between|is_null
          value: <any>
        then:
          action: BLOCK|REVIEW|FLAG|ALLOW
          severity: LOW|MEDIUM|HIGH|CRITICAL
          description: str            # explanation template, e.g. "amount {TransactionAmt} is high"
"""
import json
from pathlib import Path

import yaml

from src.services.rules.conditions import AtomicCondition, Condition, CompositeCondition, NotCondition
from src.services.rules.models import Action, Rule, Severity


def _build_condition(spec: dict) -> Condition:
    if "all_of" in spec:
        return CompositeCondition("all_of", [_build_condition(c) for c in spec["all_of"]])
    if "any_of" in spec:
        return CompositeCondition("any_of", [_build_condition(c) for c in spec["any_of"]])
    if "not" in spec:
        return NotCondition(_build_condition(spec["not"]))
    return AtomicCondition(field=spec["field"], operator=spec["operator"], value=spec.get("value"))


def _build_rule(spec: dict) -> Rule:
    then = spec["then"]
    return Rule(
        id=spec["id"],
        name=spec["name"],
        priority=spec["priority"],
        condition=_build_condition(spec["if"]),
        action=Action[then["action"]],
        severity=Severity[then["severity"]],
        description=then["description"],
        enabled=spec.get("enabled", True),
    )


class RuleLoader:
    def load(self, path: Path | str) -> list[Rule]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif path.suffix == ".json":
            data = json.loads(text)
        else:
            raise ValueError(f"unsupported rules file extension: {path.suffix!r} (expected .yaml/.yml/.json)")

        return [_build_rule(spec) for spec in data["rules"]]
