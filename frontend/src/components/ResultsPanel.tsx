import { PredictResponse } from "../api/client"

interface ResultsPanelProps {
  result: PredictResponse
  onReset: () => void
}

export default function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Classification Results</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Main Categories Card */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h3 className="text-sm font-medium text-gray-600 mb-4">Main Categories</h3>
          <div className="space-y-4">
            {result.main_categories.map((cat, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between items-center">
                  <p className="font-semibold text-gray-900">{cat.category}</p>
                  <span className="text-sm font-semibold text-gray-900">
                    {Math.round(cat.confidence * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-blue-600 h-full rounded-full transition-all"
                    style={{ width: `${Math.round(cat.confidence * 100)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Sub Categories Card */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h3 className="text-sm font-medium text-gray-600 mb-4">Sub-Categories</h3>
          <div className="space-y-4">
            {result.sub_categories.map((cat, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between items-center">
                  <p className="font-semibold text-gray-900">{cat.category}</p>
                  <span className="text-sm font-semibold text-gray-900">
                    {Math.round(cat.confidence * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-600 h-full rounded-full transition-all"
                    style={{ width: `${Math.round(cat.confidence * 100)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="text-center text-sm text-gray-500">
        Inference time: {result.inference_time_ms}ms
      </p>

      <button
        onClick={onReset}
        className="w-full border-2 border-blue-600 text-blue-600 hover:bg-blue-50 font-medium py-2 px-4 rounded-lg transition"
      >
        Classify another paper
      </button>
    </div>
  )
}
