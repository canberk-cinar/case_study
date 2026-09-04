from pydantic import BaseModel


class FiredRuleExplanation(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    action: str
    priority: int
    condition: str
    message: str


class ExplainResponse(BaseModel):
    transaction_id: int
    fired_rules: list[FiredRuleExplanation]
    verdict_severity: str | None
    verdict_action: str | None
    verdict_rule_id: str | None
