/**
 * Browser-side ONNX model registry: model fetch with download progress
 * and WebGPU→WASM provider selection (a proven onnxruntime-web pattern).
 */

import * as ort from 'onnxruntime-web'

// ort-web 1.26.0 WASM files are loaded from CDN (same scheme as the admin client)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface ModelInfo {
  name: string
  status: ModelStatus
  /** Download progress 0..1 (exact: stream is read chunk by chunk) */
  progress: number
  loadedBytes: number
  totalBytes: number
  error?: string
}

/** Engine model files; the OCR dictionary is fetched separately as text */
export const MODEL_FILES = {
  yolo: 'yolov8n.onnx',
  plate: 'plate.onnx',
  ocr: 'ocr_rec_en.onnx',
  faceDet: 'det_500m.onnx',
  faceEmb: 'w600k_mbf.onnx',
  fastPlate: 'fastplate/model.onnx',
} as const

export const OCR_DICT_FILE = 'ocr_rec_en.dict.txt'

const MODELS_BASE = `${import.meta.env.BASE_URL}models/`

/** Download with real progress (ReadableStream) */
async function fetchBuffer(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while downloading ${url}`)
  }
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body?.getReader()
  if (!reader) {
    const buf = await res.arrayBuffer()
    onProgress(buf.byteLength, buf.byteLength)
    return new Uint8Array(buf)
  }
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    onProgress(received, total)
  }
  const out = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  return out
}

export class ModelRegistry {
  private readonly sessions = new Map<string, Promise<ort.InferenceSession>>()
  private readonly texts = new Map<string, Promise<string>>()
  private providersPromise: Promise<string[]> | null = null
  private listeners = new Set<(infos: ModelInfo[]) => void>()
  private readonly infos = new Map<string, ModelInfo>()

  constructor() {
    for (const file of Object.values(MODEL_FILES)) {
      this.infos.set(file, { name: file, status: 'idle', progress: 0, loadedBytes: 0, totalBytes: 0 })
    }
  }

  /** UI subscription to model loading statuses */
  subscribe(listener: (infos: ModelInfo[]) => void): () => void {
    this.listeners.add(listener)
    listener(this.infoList())
    return () => this.listeners.delete(listener)
  }

  infoList(): ModelInfo[] {
    return [...this.infos.values()]
  }

  isReady(file: string): boolean {
    return this.infos.get(file)?.status === 'ready'
  }

  modelUrl(file: string): string {
    return MODELS_BASE + file
  }

  /** Providers: WebGPU when available, otherwise WASM (single probe per process) */
  private async providers(): Promise<string[]> {
    if (!this.providersPromise) {
      this.providersPromise = (async () => {
        const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
        if (typeof navigator !== 'undefined' && gpu) {
          try {
            const adapter = await gpu.requestAdapter()
            if (adapter) {
              return ['webgpu', 'wasm']
            }
          } catch {
            // WebGPU unavailable — silently fall back to WASM
          }
        }
        return ['wasm']
      })()
    }
    return this.providersPromise
  }

  /** Model session (lazy, cached); on WebGPU failure — explicit WASM retry */
  get(file: string): Promise<ort.InferenceSession> {
    const cached = this.sessions.get(file)
    if (cached) {
      return cached
    }
    const created = this.createSession(file)
    this.sessions.set(file, created)
    return created
  }

  private setStatus(file: string, patch: Partial<ModelInfo>) {
    const info = this.infos.get(file)
    if (!info) {
      return
    }
    Object.assign(info, patch)
    const snapshot = this.infoList()
    for (const listener of this.listeners) {
      listener(snapshot)
    }
  }

  private async createSession(file: string): Promise<ort.InferenceSession> {
    this.setStatus(file, { status: 'loading', progress: 0, loadedBytes: 0, totalBytes: 0, error: undefined })
    try {
      const bytes = await fetchBuffer(this.modelUrl(file), (loaded, total) => {
        this.setStatus(file, {
          loadedBytes: loaded,
          totalBytes: total || loaded,
          progress: total > 0 ? Math.min(1, loaded / total) : 0,
        })
      })
      this.setStatus(file, { progress: 1, loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength })
      let session: ort.InferenceSession
      try {
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: await this.providers(),
          graphOptimizationLevel: 'all',
        })
      } catch {
        // The WebGPU provider may fail on this model — explicit WASM fallback
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        })
      }
      this.setStatus(file, { status: 'ready' })
      return session
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.setStatus(file, { status: 'error', error: message })
      // Do not cache the failure — the next call retries
      this.sessions.delete(file)
      throw error
    }
  }

  /** Text resource of the models directory (OCR dictionary) */
  fetchModelText(file: string): Promise<string> {
    const cached = this.texts.get(file)
    if (cached) {
      return cached
    }
    const created = fetch(this.modelUrl(file)).then(res => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} while downloading ${file}`)
      }
      return res.text()
    })
    this.texts.set(file, created)
    return created
  }
}

export const modelRegistry = new ModelRegistry()
