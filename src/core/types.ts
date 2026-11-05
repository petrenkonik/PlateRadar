/**
 * Recognition types (recognition contract + demo extensions).
 * Object coordinates are normalized (0..1), same as the production contract.
 */

/** Raw RGB pixels (3 bytes per pixel) — Uint8Array in the browser instead of Node Buffer */
export interface RgbImage {
  data: Uint8Array
  width: number
  height: number
}

/** Rectangle in source image pixels */
export interface PixelBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface RecognizedBbox {
  x: number
  y: number
  w: number
  h: number
}

export interface RecognizedPlate {
  /** Normalized plate: no spaces or separators */
  value: string
  /** Raw reading — as the text appears on the plate */
  raw: string
  confidence: number
}

export interface DemoFaceInfo {
  detScore: number
  /** ArcFace embedding dimensionality (512) — the vector itself is not passed to the UI */
  dims: number
}

/** Debug crops for the panel (JPEG dataURL) */
export interface DemoCrops {
  car?: string
  plate?: string
  plateDeskew?: string
}

export interface DemoObject {
  type: 'car' | 'person'
  bbox: RecognizedBbox
  /** Box in frame pixels — for overlay and crops */
  box: PixelBox
  score: number
  /** Plate: production OCR (PaddleOCR ONNX ocr_rec_en) */
  plate?: RecognizedPlate
  /** Plate: fast-plate-ocr (cct-xs-v2-global) — engine comparison */
  plateFast?: RecognizedPlate
  /** Region/country from the fast-plate-ocr region head (v2 models) */
  plateRegion?: string
  /** Plate box in normalized 0..1 frame coordinates — for tracking */
  plateBox?: RecognizedBbox
  /** Plate detection confidence 0..1 */
  plateScore?: number
  color?: string
  face?: DemoFaceInfo
  crops?: DemoCrops
}

export interface PipelineTimings {
  /** YOLO: people/vehicles over the frame */
  detectMs: number
  /** Plate detection (car crops + full frame) */
  plateDetectMs: number
  /** Production engine OCR (with deskew) */
  ocrNativeMs: number
  /** fast-plate-ocr OCR */
  ocrFastMs: number
  /** SCRFD + ArcFace */
  facesMs: number
  /** Body color k-means */
  colorMs: number
  totalMs: number
}

export interface PipelineResult {
  objects: DemoObject[]
  timings: PipelineTimings
  width: number
  height: number
  /** Per-stage errors — stages are independent, the frame is still displayed */
  warnings: string[]
}

export interface PipelineOptions {
  /** YOLO confidence threshold (0.05–0.95), engine default 0.35 */
  yoloScore: number
  /**
   * Display/confirmation threshold (0..1): a vehicle box and its plate are
   * shown only when BOTH the vehicle score and the plate confidence are at
   * or above this level (multi-frame voting applies on top — see plate-tracker).
   */
  confirmScore: number
  detectPersons: boolean
  detectCars: boolean
  detectFaces: boolean
  detectPlates: boolean
  /** Body color k-means */
  carColor: boolean
  /** Run fast-plate-ocr on plate crops */
  fastPlate: boolean
}
