from pydantic import BaseModel

from typing import List

class PredictionRequest(BaseModel):
    features : List[List[float]]


class PredictionResponse(BaseModel):
    prediction : List[int]

    