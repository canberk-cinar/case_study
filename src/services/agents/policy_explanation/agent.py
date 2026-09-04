"""Case 9 — policy_explanation agent node: the ONE genuinely LLM-backed node in this graph. Wraps
Case 8's RAGPipeline.answer_for_flagged_transaction() directly — its retrieval, prompt construction
(PromptBuilder), and graceful-degradation-without-Ollama behavior all already exist in Case 8 and
aren't duplicated here. This node's whole job is bridging graph.py's AgentState into the call Case
8 already knows how to make.
"""
import logging

from src.config import REPO_ROOT
from src.database.db_services import close, new_session
from src.services.agents.state import AgentState
from src.services.rag.container import DEFAULT_CONFIG, RAGContainer

logger = logging.getLogger(__name__)


def _load_knowledge_base_documents() -> list[tuple[str, str, str]]:
    kb_dir = REPO_ROOT / "data" / "knowledge_base"
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        source = "experimental" if "experimental" in path.stem else "case_06_07"
        documents.append((title, source, text))
    return documents


def explain_verdict(state: AgentState) -> dict:
    transaction_id = state["transaction_id"]
    rule_verdict = state["rule_verdict"]
    logger.info("explain_verdict — transaction_id=%s", transaction_id)

    container = RAGContainer()
    container.config.from_dict(DEFAULT_CONFIG)
    pipeline = container.rag_pipeline()

    db = new_session()
    try:
        pipeline.ingest(db, _load_knowledge_base_documents())
        result = pipeline.answer_for_flagged_transaction(db, rule_verdict)
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
