from sqlalchemy.orm import Session

from ..db import SessionLocal


def new_session() -> Session:
    """Opens a new DB session for code that runs outside the request-scoped get_db() dependency
    (e.g. a script's __main__ block, a background pipeline stage)."""
    return SessionLocal()


def rollback(db: Session) -> None:
    db.rollback()


def close(db: Session) -> None:
    db.close()