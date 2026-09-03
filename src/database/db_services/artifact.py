from sqlalchemy.orm import Session

from ..models.artifact import Artifact


def record_artifact(
    db: Session,
    case_name: str,
    kind: str,
    path: str,
    row_count: int | None = None,
    col_count: int | None = None,
) -> Artifact:
    artifact = Artifact(
        case_name=case_name,
        kind=kind,
        path=path,
        row_count=row_count,
        col_count=col_count,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def list_artifacts(db: Session, case_name: str | None = None) -> list[Artifact]:
    query = db.query(Artifact)
    if case_name is not None:
        query = query.filter(Artifact.case_name == case_name)
    return query.order_by(Artifact.created_at.desc()).all()