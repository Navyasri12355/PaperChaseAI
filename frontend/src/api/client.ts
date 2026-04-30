import axios from "axios"

const api = axios.create({
  baseURL: "/api",
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
