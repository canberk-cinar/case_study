"""Case 9 — policy_explanation agent node: the ONE genuinely LLM-backed node in this graph. Wraps
Case 8's RAGPipeline for retrieval/context-injection/generation, but owns its own task-specific
prompt content: the question template that turns a Case 7 rule verdict into a natural-language
request (static/question_template.json). This is what makes it genuinely THIS agent's prompt,
distinct from RAG's generic system instructions (services/rag/static/system_prompt.json), which
apply to any RAGPipeline caller — the question TEMPLATE is specific to "explain a flagged
transaction," this agent's one job; the system prompt is generic to "answer from these sources,"
useful to any caller including Case 10's bare /rag/query.

Takes the RAGPipeline as a parameter rather than building its own RAGContainer inline — graph.py's
closures inject it from the same DI container Case 10's /rag/query route uses
(ApiContainer.rag_container), so swapping the embedding/LLM provider is one config change that
affects every consumer, not just the ones that happen to build their own container.
"""
import json
import logging
from pathlib import Path

from src.agents.schemas.state import AgentState
from src.config import REPO_ROOT
from src.database.db_services import close, new_session
from src.services.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_QUESTION_TEMPLATE = json.loads((_STATIC_DIR / "question_template.json").read_text())["question_template"]


def _load_knowledge_base_documents() -> list[tuple[str, str, str]]:
    kb_dir = REPO_ROOT / "data" / "knowledge_base"
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        source = "experimental" if "experimental" in path.stem else "case_06_07"
        documents.append((title, source, text))
    return documents


def build_flagged_transaction_question(explanation: dict) -> str:
    """explanation: the dict shape produced by Case 7's RuleEngine.explain()."""
    rule_names = ", ".join(f["rule_name"] for f in explanation["fired_rules"])
    return _QUESTION_TEMPLATE.format(
        rule_names=rule_names,
        verdict_severity=explanation["verdict_severity"],
        verdict_action=explanation["verdict_action"],
    )


def explain_verdict(state: AgentState, pipeline: RAGPipeline) -> dict:
    transaction_id = state["transaction_id"]
    rule_verdict = state["rule_verdict"]
    logger.info("explain_verdict — transaction_id=%s", transaction_id)

    question = build_flagged_transaction_question(rule_verdict)

    db = new_session()
    try:
        pipeline.ingest(db, _load_knowledge_base_documents())
        result = pipeline.answer(db, question)
    finally:
        close(db)

    logger.info("explain_verdict — note=%s", result.get("note"))
    return {
        "policy_explanation": {
            "question": result["question"],
            "answer": result["answer"],
            "note": result["note"],
            "sources": [rc.chunk.document_title for rc in result["sources"]],
        }
    }
