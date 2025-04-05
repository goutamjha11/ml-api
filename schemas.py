from pydantic import BaseModel

from typing import List, Dict

class PredictionRequest(BaseModel):
    features : List[Dict[str, float]]


class PredictionResponse(BaseModel):
    prediction : List[int]

    