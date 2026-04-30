from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model_service import model_service
from app.routers import predict, health, categories

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load(
        main_model_path="../Main Category",
        sub_model_path="../Sub Category/best_model",
        le_main_path="../le_main_category.pkl",
        le_sub_path="../le_sub_category.pkl",
        hierarchy_path="../phase4_hierarchy_map.json"
    )
    yield

app = FastAPI(title="ArXiv Paper Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(categories.router)
