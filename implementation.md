
Full Implementation Plan — ArXiv Hierarchical Classifier
Context for the agent
This is a FastAPI + React app that serves two fine-tuned SciBERT models for hierarchical ArXiv paper classification. The ML models already exist — you are only building the backend and frontend. Do not touch any existing files in the repo.

Assumptions / file paths the agent must know
# These files already exist in the repo root — read-only, do not modify
phase4_hierarchy_map.json
le_main_category.pkl
le_sub_category.pkl

# These folders already exist with config.json, tokenizer.json, tokenizer_config.json
# The model.safetensors files will be placed here by the user before running
Main Category/
Sub Category/best_model/

Step 1 — Backend
Create backend/requirements.txt
fastapi
uvicorn[standard]
transformers
torch
scikit-learn
joblib
python-dotenv
pydantic>=2.0
numpy
safetensors

Create backend/app/schemas.py
pythonfrom pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    abstract: str = Field(..., min_length=50, max_length=3000)

class PredictResponse(BaseModel):
    main_category: str
    sub_category: str
    main_confidence: float
    sub_confidence: float
    inference_time_ms: float

Create backend/app/model_service.py
pythonimport json, time, joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelService:
    def __init__(self):
        self.loaded = False

    def load(self,
             main_model_path: str,
             sub_model_path: str,
             le_main_path: str,
             le_sub_path: str,
             hierarchy_path: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizers
        self.main_tokenizer = AutoTokenizer.from_pretrained(main_model_path)
        self.sub_tokenizer = AutoTokenizer.from_pretrained(sub_model_path)

        # Load models
        self.main_model = AutoModelForSequenceClassification.from_pretrained(main_model_path)
        self.main_model.to(self.device)
        self.main_model.eval()

        self.sub_model = AutoModelForSequenceClassification.from_pretrained(sub_model_path)
        self.sub_model.to(self.device)
        self.sub_model.eval()

        # Load label encoders and hierarchy
        self.le_main = joblib.load(le_main_path)
        self.le_sub = joblib.load(le_sub_path)
        self.hierarchy = json.load(open(hierarchy_path))

        self.loaded = True

    def predict(self, title: str, abstract: str) -> dict:
        start = time.time()
        text = f"{title} [SEP] {abstract}"

        # --- Main category inference ---
        main_inputs = self.main_tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            main_logits = self.main_model(**main_inputs).logits

        main_probs = torch.softmax(main_logits, dim=-1).squeeze()
        main_idx = main_probs.argmax().item()
        main_conf = main_probs[main_idx].item()
        main_label = self.le_main.inverse_transform([main_idx])[0]

        # --- Sub category inference with constraint masking ---
        sub_inputs = self.sub_tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sub_logits = self.sub_model(**sub_inputs).logits.squeeze()

        # Mask out sub-categories not belonging to predicted main category
        valid_subs = set(self.hierarchy.get(main_label, []))
        for i, label in enumerate(self.le_sub.classes_):
            if label not in valid_subs:
                sub_logits[i] = -1e9

        sub_probs = torch.softmax(sub_logits, dim=-1)
        sub_idx = sub_probs.argmax().item()
        sub_conf = sub_probs[sub_idx].item()
        sub_label = self.le_sub.inverse_transform([sub_idx])[0]

        inference_time = (time.time() - start) * 1000

        return {
            "main_category": main_label,
            "sub_category": sub_label,
            "main_confidence": round(main_conf, 4),
            "sub_confidence": round(sub_conf, 4),
            "inference_time_ms": round(inference_time, 2)
        }

model_service = ModelService()

Create backend/app/routers/predict.py
pythonfrom fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.model_service import model_service

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    result = model_service.predict(request.title, request.abstract)
    return result

Create backend/app/routers/health.py
pythonfrom fastapi import APIRouter
from app.model_service import model_service

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service.loaded}

Create backend/app/routers/categories.py
pythonimport json
from fastapi import APIRouter

router = APIRouter()

@router.get("/categories")
def categories():
    with open("phase4_hierarchy_map.json") as f:
        return json.load(f)

Create backend/app/main.py
pythonfrom contextlib import asynccontextmanager
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

Create backend/.env
MODEL_MAIN_PATH=../Main Category
MODEL_SUB_PATH=../Sub Category/best_model
(Not strictly needed since paths are hardcoded above, but good practice for the agent to create)

Step 2 — Frontend
Scaffold command (run from repo root):
bashnpm create vite@latest frontend -- --template react-ts
cd frontend
npm install axios @tanstack/react-query tailwindcss @tailwindcss/forms postcss autoprefixer
npx tailwindcss init -p

Update frontend/tailwind.config.js
jsexport default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [require("@tailwindcss/forms")],
}
Update frontend/src/index.css — replace contents with:
css@tailwind base;
@tailwind components;
@tailwind utilities;

Create frontend/src/api/client.ts
tsimport axios from "axios"

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 30000,
})

export interface PredictRequest {
  title: string
  abstract: string
}

export interface PredictResponse {
  main_category: string
  sub_category: string
  main_confidence: number
  sub_confidence: number
  inference_time_ms: number
}

export async function classifyPaper(req: PredictRequest): Promise<PredictResponse> {
  const { data } = await api.post<PredictResponse>("/predict", req)
  return data
}

Create frontend/src/components/PaperForm.tsx
A controlled form with:

Title input — required, max 300 chars, character counter bottom-right, red border + error message if submitted empty or over limit
Abstract textarea — required, min 50 / max 3000 chars, character counter, auto-resize via rows prop, same error behaviour
Submit button — full width, blue, shows a spinner (animate-spin border trick) and "Classifying..." text while loading, otherwise "Classify Paper"
Props: onResult(data: PredictResponse): void, so parent receives the result


Create frontend/src/components/ResultsPanel.tsx
Receives a PredictResponse as prop. Renders:

A section header: "Classification Results"
Two side-by-side cards:

Main Category card — large bold label, confidence as a labelled progress bar (e.g. "84%"), bar fills blue proportional to confidence
Sub-Category card — same layout, bar fills indigo


Small grey metadata line at bottom: "Inference time: Xms"
A "Classify another paper" button (outlined, full width) that calls an onReset() prop to clear results and show the form again


Create frontend/src/App.tsx
tsximport { useState } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import PaperForm from "./components/PaperForm"
import ResultsPanel from "./components/ResultsPanel"
import { PredictResponse } from "./api/client"

const queryClient = new QueryClient()

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null)

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 text-center">
            ArXiv Paper Classifier
          </h1>
          <p className="text-gray-500 text-center mb-8">
            Paste a paper's title and abstract to classify it into an ArXiv category
          </p>
          <div className="bg-white rounded-2xl shadow-md p-8">
            {result
              ? <ResultsPanel result={result} onReset={() => setResult(null)} />
              : <PaperForm onResult={setResult} />
            }
          </div>
        </div>
      </div>
    </QueryClientProvider>
  )
}

Update frontend/src/main.tsx — standard Vite entry, no changes needed beyond what's scaffolded.

Step 3 — How to run
bash# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# App opens at http://localhost:5173

Important notes for the agent

The paths ../Main Category and ../Sub Category/best_model assume the backend is run from the backend/ directory. If the agent changes the working directory, adjust accordingly.
The folder name Main Category has a space — make sure path strings are quoted correctly everywhere.
Do not move or rename any existing ML files.
The model_service singleton is imported by the router — the agent must not re-instantiate it.
PaperForm.tsx and ResultsPanel.tsx should be fully implemented components, not stubs — the agent should write the complete JSX and Tailwind styling.