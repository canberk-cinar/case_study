from pydantic import BaseModel


class FiredRuleSummary(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    action: str
    priority: int


class RuleEvaluationResponse(BaseModel):
    transaction_id: int
    fired_rules: list[FiredRuleSummary]
    verdict_severity: str | None
    verdict_action: str | None
    verdict_rule_id: str | None
