from fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.model_service import model_service

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    result = model_service.predict(request.title, request.abstract)
    return result
