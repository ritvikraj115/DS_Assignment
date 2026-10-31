import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: Number(import.meta.env.VITE_API_TIMEOUT_MS || 600000),
})

const debug = import.meta.env.VITE_DEBUG === 'true' || (import.meta.env.DEV && import.meta.env.VITE_DEBUG !== 'false')
export const confidenceThresholds = {
  high: Number(import.meta.env.VITE_CONFIDENCE_HIGH_THRESHOLD || 85),
  moderate: Number(import.meta.env.VITE_CONFIDENCE_MODERATE_THRESHOLD || 70),
}
api.interceptors.request.use((config) => {
  config.metadata = { startedAt: performance.now() }
  if (debug) console.info(`[DocuMind API] → ${config.method?.toUpperCase()} ${config.url}`)
  return config
})
api.interceptors.response.use((response) => {
  if (debug) console.info(`[DocuMind API] ← ${response.status} ${response.config.url} (${Math.round(performance.now() - response.config.metadata.startedAt)} ms)`)
  return response
}, (error) => {
  if (debug) console.warn(`[DocuMind API] × ${error.response?.status || 'network'} ${error.config?.url || 'request'}: ${friendlyError(error)}`)
  return Promise.reject(error)
})

export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function askQuestion({ question, sessionId }) {
  const { data } = await api.post('/query', { question, session_id: sessionId })
  return data
}

const STREAM_STAGE_LABELS = {
  started: 'Starting the documentation assistant',
  query_analysis: 'Understanding the question',
  retrieval: 'Searching indexed documentation',
  source_review: 'Checking retrieved sources for relevance',
  retrieval_retry: 'No suitable sources yet — revising the search',
  generation: 'Drafting an answer from relevant sources',
  evaluation: 'Verifying the draft against documentation',
  answer_retry: 'Revising the draft using verification feedback',
  approved: 'Answer approved — streaming response',
  unverified: 'Verification failed — preparing a safe response',
  insufficient_evidence: 'No sufficiently relevant documentation found',
}

export async function streamQuestion({ question, sessionId, onStage, onToken }) {
  const baseURL = api.defaults.baseURL
  const startedAt = performance.now()
  if (debug) console.info('[DocuMind API] → POST /query/stream')
  const response = await fetch(`${baseURL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question, session_id: sessionId }),
  })
  if (!response.ok) {
    let body = {}
    try { body = await response.json() } catch { /* response may have no JSON body */ }
    const error = new Error(body.detail || 'The query could not be completed.')
    error.response = { status: response.status, data: body }
    throw error
  }
  if (!response.body) throw new Error('Streaming is unavailable in this browser.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result = null
  const dispatch = (frame) => {
    let eventName = 'message'
    const data = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim()
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
    }
    if (!data.length) return
    const payload = JSON.parse(data.join('\n'))
    if (eventName === 'stage') onStage?.({ ...payload, label: STREAM_STAGE_LABELS[payload.stage] || payload.detail })
    if (eventName === 'token') onToken?.(payload.text || '')
    if (eventName === 'result') result = payload.response
    if (eventName === 'error') throw new Error(payload.message || 'The query could not be completed.')
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
      let boundary
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const frame = buffer.slice(0, boundary)
        const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/)[0]
        buffer = buffer.slice(boundary + separator.length)
        dispatch(frame)
      }
      if (done) break
    }
  } finally {
    reader.releaseLock()
  }
  if (debug) console.info(`[DocuMind API] ← ${response.status} /query/stream (${Math.round(performance.now() - startedAt)} ms)`)
  if (!result) throw new Error('The response stream ended before the answer was complete.')
  return result
}

export async function fetchDocuments() {
  const { data } = await api.get('/documents')
  return data.documents || []
}

export async function ingestFile(file, onUploadProgress) {
  const body = new FormData()
  body.append('file', file)
  const { data } = await api.post('/ingest', body, {
    onUploadProgress: (event) => {
      if (event.total) onUploadProgress?.(Math.round((event.loaded * 100) / event.total))
    },
  })
  return data
}

export async function ingestUrl(url) {
  const body = new FormData()
  body.append('url', url)
  const { data } = await api.post('/ingest', body)
  return data
}

export async function submitFeedback({ question, sessionId, rating, comment }) {
  const { data } = await api.post('/feedback', {
    question,
    session_id: sessionId,
    rating,
    comment: comment || null,
  })
  return data
}

export function friendlyError(error) {
  if (error?.response?.data?.answer) return error.response.data.answer
  if (error?.response?.data?.detail) {
    const detail = error.response.data.detail
    if (typeof detail === 'string') return detail
    return 'The request was rejected. Check the fields and try again.'
  }
  if (error?.code === 'ECONNABORTED') return 'The request took too long. Please try again.'
  if (error?.request) return 'Could not reach DocuMind. Check that the API server is running.'
  return error?.message || 'Something went wrong. Please try again.'
}

export { api }
