/**
 * yolov8-format detection (onnxruntime-web).
 */

import * as ort from 'onnxruntime-web'
import { letterboxToTensor, nms, unletterbox } from './tensor.util'
import type { PixelBox, RgbImage } from './types'
import { MODEL_FILES, modelRegistry } from './model-registry'

/** COCO classes looked for by the main detector */
const PERSON_CLASS = 0
/** Vehicles: car (2) + bus (5) + truck (7) — COCO calls minibuses/vans not "car" */
const VEHICLE_CLASSES = new Set([2, 5, 7])

const YOLO_INPUT = 640

export interface YoloDetection extends PixelBox {
  score: number
  classId: number
}

/** Default yolov8n detection confidence threshold (detections below are dropped) */
export const YOLO_DEFAULT_SCORE = 0.35

/** Share of the smaller box inside the other (1 = fully contained) */
function containment(a: PixelBox, b: PixelBox): number {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  return inter / Math.max(Math.min(areaA, areaB), 1e-6)
}

/**
 * Merge duplicate detections of one object: same class + the smaller box
 * mostly inside the bigger one (containment ≥ 0.7). NMS alone misses
 * these — its IoU is low when a duplicate is much smaller than the full
 * object, so the detector can report two boxes for one vehicle. The host
 * box expands to the union, the score is the best of the two. Partial
 * boxes (containment ~0.5, e.g. neighboring cars split badly) stay
 * separate — merging them would fuse distinct objects.
 */
export function mergeContainedDetections(
  detections: (YoloDetection & { area: number })[],
): (YoloDetection & { area: number })[] {
  const merged: (YoloDetection & { area: number })[] = []
  for (const det of [...detections].sort((a, b) => b.score - a.score)) {
    const host = merged.find(
      other => other.classId === det.classId && containment(det, other) >= 0.7,
    )
    if (host) {
      host.x1 = Math.min(host.x1, det.x1)
      host.y1 = Math.min(host.y1, det.y1)
      host.x2 = Math.max(host.x2, det.x2)
      host.y2 = Math.max(host.y2, det.y2)
      host.area = (host.x2 - host.x1) * (host.y2 - host.y1)
      host.score = Math.max(host.score, det.score)
    } else {
      merged.push({ ...det })
    }
  }
  return merged
}

export class YoloDetector {
  /** Vehicles and pedestrians on the whole image (yolov8n) */
  async detectObjects(img: RgbImage, minScore = YOLO_DEFAULT_SCORE): Promise<YoloDetection[]> {
    const detections = await this.run(MODEL_FILES.yolo, img, minScore, 0.45)
    return detections.filter(d => VEHICLE_CLASSES.has(d.classId) || d.classId === PERSON_CLASS)
  }

  /** License plates inside a car crop (separate yolov8n model, 1 class) */
  async detectPlates(img: RgbImage): Promise<YoloDetection[]> {
    return this.run(MODEL_FILES.plate, img, 0.35, 0.45)
  }

  private async run(
    modelFile: string,
    img: RgbImage,
    confThreshold: number,
    iouThreshold: number,
  ): Promise<YoloDetection[]> {
    const session = await modelRegistry.get(modelFile)
    const letterbox = letterboxToTensor(img, YOLO_INPUT, v => v / 255)
    const inputName = session.inputNames[0]
    const outputName = session.outputNames[0]
    const results = await session.run({
      [inputName]: new ort.Tensor('float32', letterbox.data, [1, 3, YOLO_INPUT, YOLO_INPUT]),
    })

    // Output [1, 4+classes, N]: cx, cy, w, h + per-anchor class scores
    const output = results[outputName]
    const [, channels, anchors] = output.dims
    const numClasses = channels - 4
    const raw = output.data as Float32Array

    const candidates: (YoloDetection & { area: number })[] = []
    for (let i = 0; i < anchors; i++) {
      let bestClass = -1
      let bestScore = confThreshold
      for (let c = 0; c < numClasses; c++) {
        const score = raw[(4 + c) * anchors + i]
        if (score > bestScore) {
          bestScore = score
          bestClass = c
        }
      }
      if (bestClass < 0) {
        continue
      }
      const cx = raw[i]
      const cy = raw[anchors + i]
      const w = raw[2 * anchors + i]
      const h = raw[3 * anchors + i]
      const box: PixelBox = {
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
      }
      const original = unletterbox(box, img, letterbox)
      const width = original.x2 - original.x1
      const height = original.y2 - original.y1
      if (width < 2 || height < 2) {
        continue
      }
      candidates.push({
        ...original,
        score: bestScore,
        classId: bestClass,
        area: width * height,
      })
    }

    const kept = nms(candidates, iouThreshold)
    return mergeContainedDetections(kept.map(index => candidates[index]))
      .sort((a, b) => b.score - a.score)
      .slice(0, 30)
      .map(({ area: _area, ...detection }) => detection)
  }
}
