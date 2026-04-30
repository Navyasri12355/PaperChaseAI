from fastapi import APIRouter
from app.model_service import model_service

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service.loaded}
