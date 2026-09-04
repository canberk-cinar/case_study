"""Case 10 — FastAPI entry point. Structure mirrors the user's own reference project
(dashboard/backend/main.py): load .env, configure logging, build the app, run migrations on
startup, expose /healthz, include routers. Auth/rate-limiting/scheduler middleware from that
reference are deliberately not carried over — this is a single-user case-study demo API, not a
multi-tenant production service.
"""
from dotenv import load_dotenv

load_dotenv()

from src.config import configure_logging, settings  # noqa: E402

configure_logging()

import logging  # noqa: E402

logger = logging.getLogger(__name__)

from fastapi import FastAPI  # noqa: E402

from src.container import build_container  # noqa: E402
from src.database.db import run_migrations  # noqa: E402
from src.routes import agent, anomaly, explainability, rag, rules  # noqa: E402

app = FastAPI(title="Fraud/Anomaly Detection Platform API")

container = build_container()
app.state.container = container
container.wire(modules=[
    "src.routes.rules",
    "src.routes.explainability",
    "src.routes.rag",
])


@app.on_event("startup")
def startup():
    logger.info("Starting Fraud/Anomaly Detection Platform API")
    run_migrations()
    logger.info("Startup complete")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


app.include_router(anomaly.router)
app.include_router(rules.router)
app.include_router(explainability.router)
app.include_router(rag.router)
app.include_router(agent.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
