"""Case 10 — top-level API container: composes Case 7's RuleEngineContainer and Case 8's
RAGContainer (both already built, not duplicated here) behind dependency_injector's FastAPI wiring
(`@inject` + `Provide[...]`) — the bonus requirement's DI/Container mechanism, applied where it
manages real, configurable objects (RuleEngine, RAGPipeline). Trivial per-request state (the DB
session) still goes through FastAPI's own `Depends(get_db)`, matching the reference project's own
convention — DI everywhere would be DI for its own sake, not for a real need.
"""
from dependency_injector import containers, providers

from src.config import REPO_ROOT
from src.services.rag.container import DEFAULT_CONFIG as RAG_DEFAULT_CONFIG
from src.services.rag.container import RAGContainer
from src.services.rules.container import RuleEngineContainer

RULES_PATH = REPO_ROOT / "src" / "services" / "rules" / "definitions" / "fraud_rules.yaml"


class ApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=[
        "src.routes.rules",
        "src.routes.explainability",
        "src.routes.rag",
    ])

    rule_engine_container = providers.Container(RuleEngineContainer)
    rag_container = providers.Container(RAGContainer)


def build_container() -> ApiContainer:
    container = ApiContainer()
    container.rule_engine_container.config.rules_path.from_value(str(RULES_PATH))
    container.rag_container.config.from_dict(RAG_DEFAULT_CONFIG)
    return container
