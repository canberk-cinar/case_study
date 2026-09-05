"""Case 10: GET /explain/{transaction_id}: Case 7's RuleEngine.explain(): the same evaluation as
/rules/evaluate, but with per-rule human-readable messages/conditions interpolated with this
transaction's real values. Same DI-provided RuleEngine as /rules/evaluate: one configured engine,
two views over it.
"""
import logging

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from src.container import ApiContainer
from src.schemas.explainability import ExplainResponse, FiredRuleExplanation
from src.services.rules.data import load_transaction_row
from src.services.rules.engine import RuleEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explain", tags=["explainability"])


@router.get("/{transaction_id}", response_model=ExplainResponse)
@inject
def explain_endpoint(
    transaction_id: int,
    engine: RuleEngine = Depends(Provide[ApiContainer.rule_engine_container.rule_engine]),
):
    logger.info("GET /explain/%s", transaction_id)
    try:
        row = load_transaction_row(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    explanation = engine.explain(row)
    return ExplainResponse(
        transaction_id=transaction_id,
        fired_rules=[FiredRuleExplanation(**f) for f in explanation["fired_rules"]],
        verdict_severity=explanation["verdict_severity"],
        verdict_action=explanation["verdict_action"],
        verdict_rule_id=explanation["verdict_rule_id"],
    )
