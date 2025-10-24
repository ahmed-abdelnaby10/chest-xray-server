from typing import Dict, List, Optional
from pydantic import BaseModel

class HealthResp(BaseModel):
    ok: bool
    models: List[str]

class PredictSingleResp(BaseModel):
    model: str
    label: str
    probability: float

class PredictItem(BaseModel):
    file: str
    label: str
    probability: float

class PredictBatchResp(BaseModel):
    model: str
    count: int
    results: List[PredictItem]
