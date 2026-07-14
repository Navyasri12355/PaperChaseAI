import axios from "axios"

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api",
  timeout: 30000,
})

export interface PredictRequest {
  title: string
  abstract: string
  top_k?: number
  min_confidence?: number
}

export interface CategoryPrediction {
  category: string
  confidence: number
}

export interface PredictResponse {
  main_categories: CategoryPrediction[]
  sub_categories: CategoryPrediction[]
  inference_time_ms: number
}

export async function classifyPaper(req: PredictRequest): Promise<PredictResponse> {
  const { data } = await api.post<PredictResponse>("/predict", req)
  return data
}
