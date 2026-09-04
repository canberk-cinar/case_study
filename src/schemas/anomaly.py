from pydantic import BaseModel


class ScoreResponse(BaseModel):
    transaction_id: int
    final_raw_anomaly_score: float
    risk_level: str
