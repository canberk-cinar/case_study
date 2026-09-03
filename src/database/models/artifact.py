from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from ..db import Base


class Artifact(Base):
    """
    Tracks where a generated file (parquet, profile.json, report.md, ...) lives and what it
    contains — not the data itself. Large tabular outputs (e.g. the merged transaction+identity
    parquet) stay on disk for columnar, memory-bounded reads; only the pointer and shape go here,
    so any case can look up "what was last produced, when, how big" without re-reading the file.
    """
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    case_name = Column(String, nullable=False, index=True)  # e.g. "case_01_merge"
    kind = Column(String, nullable=False)                   # e.g. "parquet", "profile_json", "report_md"
    path = Column(String, nullable=False)
    row_count = Column(Integer, nullable=True)
    col_count = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())