"""Repository pattern — CRUD access to the precomputed scoring artifact, same module-level-
function style as db_services/artifact.py and db_services/rag.py."""
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..models.scoring import ScoredTransaction


def replace_scored_transactions(db: Session, df: pd.DataFrame) -> int:
    """Wipes the table and bulk-loads `df` — used by the build pipeline to make a rebuild
    idempotent. Deletes rows explicitly (not `if_exists="replace"` on to_sql, which would drop
    and recreate the table, losing the primary key/index) then appends via pandas' bulk insert."""
    db.execute(delete(ScoredTransaction))
    db.commit()

    df.to_sql("scored_transactions", con=db.get_bind(), if_exists="append", index=False, chunksize=10_000)
    return len(df)


def get_scored_transaction(db: Session, transaction_id: int) -> ScoredTransaction | None:
    return db.get(ScoredTransaction, transaction_id)


def count_scored_transactions(db: Session) -> int:
    return db.query(ScoredTransaction).count()
