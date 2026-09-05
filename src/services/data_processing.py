import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.database import db_services
from src.database.db_services.artifact import record_artifact

logger = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_data(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left-joins identity onto transaction on TransactionID. Left join because identity only
    covers ~24% of transactions: every transaction row must be kept, with NaN identity columns
    where there's no match."""
    return transaction.merge(identity, on="TransactionID", how="left")


def save_merged(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("saved merged dataset to %s (%s rows, %s cols)", path, *df.shape)


def record_merge_artifact(df: pd.DataFrame, path: Path) -> None:
    """Writes a pointer row to the DB (path + shape): not the data itself. The parquet file stays
    the source of truth for the actual rows so later steps can keep reading it column-batched."""
    db = db_services.new_session()
    try:
        record_artifact(
            db,
            case_name="case_01_merge",
            kind="parquet",
            path=str(path),
            row_count=df.shape[0],
            col_count=df.shape[1],
        )
    finally:
        db_services.close(db)


def run() -> pd.DataFrame:
    transaction = load_csv(settings.raw_data_path / "train_transaction.csv")
    identity = load_csv(settings.raw_data_path / "train_identity.csv")
    merged = merge_data(transaction, identity)

    out_path = settings.processed_data_path / "merged_transactions.parquet"
    save_merged(merged, out_path)
    record_merge_artifact(merged, out_path)

    return merged


if __name__ == "__main__":
    from src.config import configure_logging
    from src.database.db import run_migrations

    configure_logging()
    run_migrations()
    merged = run()