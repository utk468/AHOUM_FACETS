from pydantic import BaseModel, Field
from typing import List, Optional

class FacetBase(BaseModel):
    raw_name: str
    name: str
    category: str
    description: str

class FacetInDB(FacetBase):
    embedding: List[float]
    weights: List[float]

class FacetOut(FacetBase):
    pass

class FacetScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
