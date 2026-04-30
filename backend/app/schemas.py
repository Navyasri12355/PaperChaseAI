from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    abstract: str = Field(..., min_length=50, max_length=3000)

class PredictResponse(BaseModel):
    main_category: str
    sub_category: str
    main_confidence: float
    sub_confidence: float
    inference_time_ms: float
