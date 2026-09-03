import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import REPO_ROOT, settings

logger = logging.getLogger(__name__)

DATABASE_URL = settings.DATABASE_URL

if DATABASE_URL.startswith("sqlite:///./"):
    # A relative sqlite path resolves against the process's cwd, not the repo root — wrong
    # whenever something runs from elsewhere (a notebook's kernel cwd is its own directory, not
    # the repo root). Anchor it explicitly, the same way config.py anchors every other path.
    db_filename = DATABASE_URL.removeprefix("sqlite:///./")
    DATABASE_URL = f"sqlite:///{REPO_ROOT / db_filename}"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """Creates all tables directly from the ORM models. No Alembic yet — this case study's schema
    has no history to migrate; add Alembic when a real migration (not a fresh create_all) is
    actually needed."""
    from src.database import models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)
    logger.info("database tables ensured at %s", DATABASE_URL)