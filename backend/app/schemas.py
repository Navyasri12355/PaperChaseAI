from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    abstract: str = Field(..., min_length=50, max_length=3000)

class CategoryPrediction(BaseModel):
    category: str
    confidence: float

class PredictResponse(BaseModel):
    main_categories: list[CategoryPrediction]
    sub_categories: list[CategoryPrediction]
    inference_time_ms: float
