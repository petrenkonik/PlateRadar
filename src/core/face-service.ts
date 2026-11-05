/**
 * Faces: SCRFD (insightface det_500m) detection + ArcFace (w600k_mbf)
 * 512-d embedding.
 */

import * as ort from 'onnxruntime-web'
import {
  invertAffine,
  nms,
  resizeRgb,
  rgbToChw,
  similarityTransform,
  warpAffineRgb,
} from './tensor.util'
import type { PixelBox, RgbImage } from './types'
import { MODEL_FILES, modelRegistry } from './model-registry'

/** SCRFD det_500m: 640×640 input (stretch), 2 anchors per cell */
const SCRFD_SIZE = 640
const SCRFD_STRIDES = [8, 16, 32]
const SCRFD_ANCHORS_PER_CELL = 2
const DET_THRESHOLD = 0.5
const DET_NMS_IOU = 0.4
const MAX_FACES = 12

/** ArcFace (w600k_mbf): 112×112 input, 5-point alignment */
const EMBEDDING_SIZE = 112
const ARC_TEMPLATE = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
]

export interface DetectedFace {
  box: PixelBox
  score: number
  /** 5 face points (left/right eye, nose tip, mouth corners) in image pixels */
  kps: [number, number][]
}

export class FaceService {
  /** All faces on the image (SCRFD over the whole image, single run) */
  async detectFaces(img: RgbImage): Promise<DetectedFace[]> {
    const session = await modelRegistry.get(MODEL_FILES.faceDet)
    const resized = resizeRgb(img, SCRFD_SIZE, SCRFD_SIZE)
    const chw = rgbToChw(resized, v => (v - 127) / 128)
    const results = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, SCRFD_SIZE, SCRFD_SIZE]),
    })

    // 9 outputs: scores/boxes/landmarks × strides 8/16/32 — grouped by width,
    // within a group ordered by descending N (stride 8 → 16 → 32)
    const groups: Record<number, { n: number; data: Float32Array }[]> = { 1: [], 4: [], 10: [] }
    for (const name of session.outputNames) {
      const tensor = results[name]
      const dims = tensor.dims
      if (dims.length !== 2 || !groups[dims[1]]) {
        continue
      }
      groups[dims[1]].push({ n: dims[0], data: tensor.data as Float32Array })
    }
    for (const width of [1, 4, 10]) {
      groups[width].sort((a, b) => b.n - a.n)
      if (groups[width].length < SCRFD_STRIDES.length) {
        return []
      }
    }

    const scaleX = img.width / SCRFD_SIZE
    const scaleY = img.height / SCRFD_SIZE
    const faces: DetectedFace[] = []
    SCRFD_STRIDES.forEach((stride, strideIndex) => {
      const cells = SCRFD_SIZE / stride
      const scores = groups[1][strideIndex].data
      const boxes = groups[4][strideIndex].data
      const landmarks = groups[10][strideIndex].data
      const count = groups[1][strideIndex].n
      if (boxes.length / 4 < count || landmarks.length / 10 < count) {
        return
      }

      for (let i = 0; i < count; i++) {
        const score = scores[i]
        if (score < DET_THRESHOLD) {
          continue
        }
        // Anchor center: grid (col*s, row*s), 2 anchors per cell
        const cell = Math.floor(i / SCRFD_ANCHORS_PER_CELL)
        const row = Math.floor(cell / cells)
        const col = cell % cells
        const cx = col * stride
        const cy = row * stride

        const b = i * 4
        const face: DetectedFace = {
          box: {
            x1: (cx - boxes[b] * stride) * scaleX,
            y1: (cy - boxes[b + 1] * stride) * scaleY,
            x2: (cx + boxes[b + 2] * stride) * scaleX,
            y2: (cy + boxes[b + 3] * stride) * scaleY,
          },
          score,
          kps: [],
        }
        const k = i * 10
        for (let j = 0; j < 5; j++) {
          face.kps.push([
            (cx + landmarks[k + j * 2] * stride) * scaleX,
            (cy + landmarks[k + j * 2 + 1] * stride) * scaleY,
          ])
        }
        faces.push(face)
      }
    })

    const kept = nms(faces.map(f => ({ ...f.box, score: f.score })), DET_NMS_IOU)
    return kept
      .map(index => faces[index])
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_FACES)
  }

  /**
   * Face embedding from 5 points (ArcFace 512-d, L2-normalized).
   * The demo returns only the dimensionality — the vector itself goes nowhere.
   */
  async embedFaceDims(img: RgbImage, kps: [number, number][]): Promise<number> {
    if (kps.length < 5) {
      return 0
    }
    const forward = similarityTransform(kps, ARC_TEMPLATE)
    const inverse = invertAffine(forward)
    const aligned = warpAffineRgb(img, inverse, EMBEDDING_SIZE, EMBEDDING_SIZE)
    const chw = rgbToChw(aligned, v => (v - 127.5) / 127.5)

    const session = await modelRegistry.get(MODEL_FILES.faceEmb)
    const results = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, EMBEDDING_SIZE, EMBEDDING_SIZE]),
    })
    return (results[session.outputNames[0]].data as Float32Array).length
  }
}
