import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { classifyPaper, PredictResponse } from "../api/client"

interface PaperFormProps {
  onResult: (data: PredictResponse) => void
}

export default function PaperForm({ onResult }: PaperFormProps) {
  const [title, setTitle] = useState("")
  const [abstract, setAbstract] = useState("")
  const [errors, setErrors] = useState<{ title?: string; abstract?: string }>({})

  const mutation = useMutation({
    mutationFn: (data: { title: string; abstract: string }) => classifyPaper(data),
    onSuccess: (data) => {
      onResult(data)
    },
  })

  const validate = () => {
    const newErrors: { title?: string; abstract?: string } = {}

    if (!title.trim()) {
      newErrors.title = "Title is required"
    } else if (title.length > 300) {
      newErrors.title = "Title must be at most 300 characters"
    }

    if (!abstract.trim()) {
      newErrors.abstract = "Abstract is required"
    } else if (abstract.length < 50) {
      newErrors.abstract = "Abstract must be at least 50 characters"
    } else if (abstract.length > 3000) {
      newErrors.abstract = "Abstract must be at most 3000 characters"
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (validate()) {
      mutation.mutate({ title, abstract })
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="title" className="block text-sm font-medium text-gray-900 mb-2">
          Title
        </label>
        <div className="relative">
          <input
            id="title"
            type="text"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value)
              if (errors.title) setErrors({ ...errors, title: undefined })
            }}
            placeholder="Enter paper title"
            className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              errors.title ? "border-red-500" : "border-gray-300"
            }`}
            maxLength={300}
          />
          <span className="absolute right-3 top-2 text-sm text-gray-500">
            {title.length}/300
          </span>
        </div>
        {errors.title && <p className="text-red-500 text-sm mt-1">{errors.title}</p>}
      </div>

      <div>
        <label htmlFor="abstract" className="block text-sm font-medium text-gray-900 mb-2">
          Abstract
        </label>
        <div className="relative">
          <textarea
            id="abstract"
            value={abstract}
            onChange={(e) => {
              setAbstract(e.target.value)
              if (errors.abstract) setErrors({ ...errors, abstract: undefined })
            }}
            placeholder="Enter paper abstract (minimum 50 characters)"
            rows={6}
            className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none ${
              errors.abstract ? "border-red-500" : "border-gray-300"
            }`}
            maxLength={3000}
          />
          <span className="absolute right-3 bottom-2 text-sm text-gray-500">
            {abstract.length}/3000
          </span>
        </div>
        {errors.abstract && <p className="text-red-500 text-sm mt-1">{errors.abstract}</p>}
      </div>

      <button
        type="submit"
        disabled={mutation.isPending}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2 px-4 rounded-lg transition flex items-center justify-center gap-2"
      >
        {mutation.isPending ? (
          <>
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
            Classifying...
          </>
        ) : (
          "Classify Paper"
        )}
      </button>

      {mutation.isError && (
        <p className="text-red-500 text-sm">
          {mutation.error instanceof Error ? mutation.error.message : "An error occurred"}
        </p>
      )}
    </form>
  )
}
