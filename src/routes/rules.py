"""Case 10: GET /rules/evaluate/{transaction_id}: Case 7's raw structured rule evaluation (which
rules fired, verdict): no per-rule prose message (see /explain for that richer view). RuleEngine
comes from the DI container (ApiContainer.rule_engine_container.rule_engine), not constructed
inline: swapping the active rules file (container.rule_engine_container.config.rules_path) needs
no route code change.
"""
import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from src.container import ApiContainer
from src.schemas.rules import FiredRuleSummary, RuleEvaluationResponse
from src.services.rules.data import load_transaction_row
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("/evaluate/{transaction_id}", response_model=RuleEvaluationResponse)
@inject
def evaluate_rules_endpoint(
    transaction_id: int,
    engine: RuleEngine = Depends(Provide[ApiContainer.rule_engine_container.rule_engine]),
):
    logger.info("GET /rules/evaluate/%s", transaction_id)
    try:
        row = load_transaction_row(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    evaluated = engine.evaluate_all(row.to_frame().T).iloc[0]
    fired = [
        FiredRuleSummary(
            rule_id=r.id, rule_name=r.name, severity=r.severity.name,
            action=r.action.name, priority=r.priority,
        )
        for r in engine.rules if evaluated[r.id]
    ]

    return RuleEvaluationResponse(
        transaction_id=transaction_id,
        fired_rules=fired,
        verdict_severity=evaluated["verdict_severity"],
        verdict_action=evaluated["verdict_action"],
        verdict_rule_id=evaluated["verdict_rule_id"],
    )
