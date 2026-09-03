"""Case 7 — the project's first actual use of `dependency_injector` (it has been a listed
dependency since early in this project but never wired up anywhere until now). Deliberately
narrow and single-purpose: one Container wiring the rules-file path -> RuleLoader -> parsed
Rule list -> resolution chain -> RuleEngine, so swapping which rules file is active (e.g. the
canonical YAML set vs. the small JSON format-compatibility example) is a one-line config change,
not a code change — exactly the "configurable" requirement the case brief asks for.
"""
from dependency_injector import containers, providers

from src.services.rules.engine import RuleEngine
from src.services.rules.loader import RuleLoader
from src.services.rules.resolution import build_default_resolution_chain


class RuleEngineContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    rule_loader = providers.Singleton(RuleLoader)

    rules = providers.Factory(
        lambda loader, path: loader.load(path),
        loader=rule_loader,
        path=config.rules_path,
    )

    resolution_chain = providers.Factory(build_default_resolution_chain)

    rule_engine = providers.Factory(
        RuleEngine,
        rules=rules,
        resolution_chain=resolution_chain,
    )
