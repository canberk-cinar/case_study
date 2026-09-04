from pydantic import BaseModel

from src.schemas.explainability import FiredRuleExplanation


class RuleVerdict(BaseModel):
    fired_rules: list[FiredRuleExplanation]
    verdict_severity: str | None
    verdict_action: str | None
    verdict_rule_id: str | None


class PolicyExplanation(BaseModel):
    question: str
    answer: str | None
    note: str | None
    sources: list[str]


class AgentAnalysisResponse(BaseModel):
    transaction_id: int
    risk_level: str
    final_raw_anomaly_score: float | None
    rule_verdict: RuleVerdict | None
    policy_explanation: PolicyExplanation | None
