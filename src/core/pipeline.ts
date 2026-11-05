/**
 * Frame recognition pipeline (YOLO → plate detect → dual OCR → faces →
 * body color) with per-stage timings, OCR engine comparison, debug crops
 * and stage toggles.
 */

import { dominantCarColor } from './car-color'
import { FaceService, type DetectedFace } from './face-service'
import { cropRgb, rgbCropToDataUrl, clampPixel } from './tensor.util'
import { NativePlateOcr } from './plate-ocr-native'
import { FastPlateOcr } from './plate-ocr-fastplate'
import { YoloDetector, type YoloDetection } from './yolo-detector'
import type {
  DemoObject,
  PipelineOptions,
  PipelineResult,
  PipelineTimings,
  RecognizedBbox,
  RecognizedPlate,
  RgbImage,
} from './types'
import { MODEL_FILES, modelRegistry } from './model-registry'

/** Max objects per frame (guard against a cluttered frame) */
const MAX_OBJECTS = 16

/** Read plate normalization: uppercase, Latin/digits only */
const cleanPlate = (text: string) => (text || '').toUpperCase().replace(/[^A-Z0-9]/g, '')

/**
 * Does the reading look like a plate: the Ukrainian format always has
 * digits (at least 2 even in a partial reading) — filters out bumper
 * captions the plate detector picks up as plates.
 */
const looksLikePlate = (value: string) =>
  value.length >= 4 && value.length <= 10 && /\d.*\d/.test(value)

export const defaultPipelineOptions = (): PipelineOptions => ({
  yoloScore: 0.35,
  // Vehicles + plates only by default: a vehicle box and its plate are
  // displayed when BOTH scores are >= confirmScore (0.7 by default)
  confirmScore: 0.7,
  detectPersons: false,
  detectCars: true,
  detectFaces: false,
  detectPlates: true,
  carColor: false,
  fastPlate: true,
})

export class RecognitionPipeline {
  readonly yolo = new YoloDetector()
  readonly nativeOcr = new NativePlateOcr()
  readonly fastPlateOcr = new FastPlateOcr()
  private readonly faces = new FaceService()

  /** Whether all models required by the current options are ready */
  requiredModels(options: PipelineOptions): string[] {
    const files: string[] = [MODEL_FILES.yolo]
    if (options.detectPlates) {
      files.push(MODEL_FILES.plate, MODEL_FILES.ocr)
    }
    if (options.detectFaces) {
      files.push(MODEL_FILES.faceDet, MODEL_FILES.faceEmb)
    }
    if (options.fastPlate) {
      files.push(MODEL_FILES.fastPlate)
    }
    return files
  }

  /** Preload the models required by the current options — one by one for readable progress */
  async preloadModels(options: PipelineOptions): Promise<void> {
    for (const file of this.requiredModels(options)) {
      if (!modelRegistry.isReady(file)) {
        try {
          await modelRegistry.get(file)
        } catch {
          // the error is already visible in the registry statuses; recognition reports it separately
        }
      }
    }
  }

  async recognize(img: RgbImage, options: PipelineOptions): Promise<PipelineResult> {
    const started = performance.now()
    const timings: PipelineTimings = {
      detectMs: 0,
      plateDetectMs: 0,
      ocrNativeMs: 0,
      ocrFastMs: 0,
      facesMs: 0,
      colorMs: 0,
      totalMs: 0,
    }
    const warnings: string[] = []
    const objects: DemoObject[] = []
    if (!img.width || !img.height) {
      timings.totalMs = performance.now() - started
      return { objects, timings, width: img.width, height: img.height, warnings }
    }

    // YOLO: people/vehicles over the whole frame
    let detections: YoloDetection[] = []
    try {
      const t = performance.now()
      detections = (await this.yolo.detectObjects(img, options.yoloScore)).slice(0, MAX_OBJECTS)
      timings.detectMs = performance.now() - t
    } catch (error) {
      warnings.push(`YOLO: ${messageOf(error)}`)
    }
    const wanted = detections.filter(d =>
      d.classId === 0 ? options.detectPersons : options.detectCars,
    )

    // Faces are searched once for the whole frame — lazily, if there are people at all
    let faceCache: DetectedFace[] | null = null
    const getFaces = async (): Promise<DetectedFace[]> => {
      if (!faceCache) {
        const t = performance.now()
        try {
          faceCache = await this.faces.detectFaces(img)
        } catch (error) {
          warnings.push(`Faces (SCRFD): ${messageOf(error)}`)
          faceCache = []
        }
        timings.facesMs = performance.now() - t
      }
      return faceCache
    }

    const carsWithoutPlate: [DemoObject, YoloDetection][] = []
    for (const detection of wanted) {
      if (detection.classId === 0) {
        objects.push(await this.recognizePerson(img, detection, options, getFaces, timings, warnings))
      } else {
        const car = await this.recognizeCar(img, detection, options, timings, warnings)
        carsWithoutPlate.push([car, detection])
        objects.push(car)
      }
    }

    // A vehicle was found but its plate was not read: look for the plate
    // over the whole frame — the vehicle box may have cut it off (typical
    // for street shots)
    const openCars = carsWithoutPlate.filter(([obj]) => !obj.plate)
    if (options.detectPlates && openCars.length > 0) {
      await this.attachFullFramePlate(img, openCars, options, timings, warnings)
    }

    // Close-up without a vehicle detection: plate right over the whole frame
    if (options.detectPlates && !objects.some(o => o.type === 'car')) {
      const closeUp = await this.recognizeCloseUpPlate(img, options, timings, warnings)
      if (closeUp) {
        objects.push(closeUp)
      }
    }

    timings.totalMs = performance.now() - started
    return { objects, timings, width: img.width, height: img.height, warnings }
  }

  /** A full-frame plate is attached to the vehicle whose box contains its center */
  private async attachFullFramePlate(
    img: RgbImage,
    cars: [DemoObject, YoloDetection][],
    options: PipelineOptions,
    timings: PipelineTimings,
    warnings: string[],
  ): Promise<void> {
    let plate: YoloDetection | undefined
    try {
      const t = performance.now()
      const plates = await this.yolo.detectPlates(img)
      timings.plateDetectMs += performance.now() - t
      plate = plates[0]
    } catch (error) {
      warnings.push(`Plates (full frame): ${messageOf(error)}`)
      return
    }
    if (!plate) {
      return
    }
    const recognized = await this.readPlate(cropRgb(img, plate), options, timings, warnings)
    if (!recognized) {
      return
    }
    const centerX = (plate.x1 + plate.x2) / 2
    const centerY = (plate.y1 + plate.y2) / 2
    const target = cars.find(
      ([, detection]) =>
        centerX >= detection.x1 &&
        centerX <= detection.x2 &&
        centerY >= detection.y1 &&
        centerY <= detection.y2,
    )
    if (target) {
      target[0].plate = recognized.native
      target[0].plateFast = recognized.fast
      target[0].plateRegion = recognized.region
      target[0].plateBox = toNormalized(plate, img)
      target[0].plateScore = plate.score
    }
  }

  /** Plate/bumper photo without the whole vehicle: object = region around the plate */
  private async recognizeCloseUpPlate(
    img: RgbImage,
    options: PipelineOptions,
    timings: PipelineTimings,
    warnings: string[],
  ): Promise<DemoObject | undefined> {
    let plate: YoloDetection | undefined
    try {
      const t = performance.now()
      const plates = await this.yolo.detectPlates(img)
      timings.plateDetectMs += performance.now() - t
      plate = plates[0]
    } catch (error) {
      warnings.push(`Plates (full frame): ${messageOf(error)}`)
      return undefined
    }
    if (!plate) {
      return undefined
    }
    const recognized = await this.readPlate(cropRgb(img, plate), options, timings, warnings)
    if (!recognized) {
      return undefined
    }
    const width = plate.x2 - plate.x1
    const height = plate.y2 - plate.y1
    const region = {
      x1: plate.x1 - width * 0.35,
      y1: plate.y1 - height * 0.9,
      x2: plate.x2 + width * 0.35,
      y2: plate.y2 + height * 0.9,
    }
    let color: string | undefined
    if (options.carColor) {
      const t = performance.now()
      color = dominantCarColor(cropRgb(img, region))
      timings.colorMs += performance.now() - t
    }
    return {
      type: 'car',
      bbox: toNormalized(region, img),
      box: region,
      // plate detection confidence — it is the basis of the "vehicle" object here
      score: plate.score,
      plate: recognized.native,
      plateFast: recognized.fast,
      plateRegion: recognized.region,
      plateBox: toNormalized(plate, img),
      plateScore: plate.score,
      color,
      crops: {
        plate: rgbCropToDataUrl(img, expand(plate, 0.15)),
      },
    }
  }

  private async recognizePerson(
    img: RgbImage,
    detection: YoloDetection,
    options: PipelineOptions,
    getFaces: () => Promise<DetectedFace[]>,
    timings: PipelineTimings,
    warnings: string[],
  ): Promise<DemoObject> {
    const object: DemoObject = {
      type: 'person',
      bbox: toNormalized(detection, img),
      box: detection,
      score: detection.score,
    }
    if (!options.detectFaces) {
      return object
    }
    const face = await this.pickPersonFace(detection, getFaces)
    if (!face) {
      return object
    }
    try {
      const t = performance.now()
      const dims = await this.faces.embedFaceDims(img, face.kps)
      timings.facesMs += performance.now() - t
      object.face = { detScore: face.score, dims }
    } catch (error) {
      warnings.push(`Face (ArcFace): ${messageOf(error)}`)
    }
    return object
  }

  private async recognizeCar(
    img: RgbImage,
    detection: YoloDetection,
    options: PipelineOptions,
    timings: PipelineTimings,
    warnings: string[],
  ): Promise<DemoObject> {
    const crop = cropRgb(img, detection)
    const object: DemoObject = {
      type: 'car',
      bbox: toNormalized(detection, img),
      box: detection,
      score: detection.score,
    }

    // Plate: specialized detector + crop OCR
    let bestPlate: YoloDetection | undefined
    if (options.detectPlates) {
      try {
        const t = performance.now()
        const plates = await this.yolo.detectPlates(crop)
        timings.plateDetectMs += performance.now() - t
        bestPlate = plates[0]
      } catch (error) {
        warnings.push(`Plates (car crop): ${messageOf(error)}`)
      }
    }

    if (bestPlate) {
      const plateCrop = cropRgb(crop, bestPlate)
      const recognized = await this.readPlate(plateCrop, options, timings, warnings)
      if (recognized.native) {
        object.plate = recognized.native
      }
      object.plateFast = recognized.fast
      object.plateRegion = recognized.region
      // Plate box in full-frame normalized coordinates (the detector ran on
      // the car crop — offset back to the frame) for the tracker
      const cropX1 = Math.round(Math.max(0, Math.min(detection.x1, img.width - 1)))
      const cropY1 = Math.round(Math.max(0, Math.min(detection.y1, img.height - 1)))
      object.plateBox = toNormalized(
        {
          x1: cropX1 + bestPlate.x1,
          y1: cropY1 + bestPlate.y1,
          x2: cropX1 + bestPlate.x2,
          y2: cropY1 + bestPlate.y2,
        },
        img,
      )
      object.plateScore = bestPlate.score
      object.crops = {
        plate: rgbCropToDataUrl(crop, expand(bestPlate, 0.12)),
      }
    }

    if (options.carColor) {
      const t = performance.now()
      object.color = dominantCarColor(crop, bestPlate)
      timings.colorMs += performance.now() - t
    }
    return object
  }

  /**
   * Plate crop OCR by both engines (comparison): production one (with deskew)
   * and fast-plate-ocr; short/garbage production readings are dropped,
   * fast-plate is returned as-is.
   */
  private async readPlate(
    plateCrop: RgbImage,
    options: PipelineOptions,
    timings: PipelineTimings,
    warnings: string[],
  ): Promise<{
    native?: RecognizedPlate
    fast?: RecognizedPlate
    region?: string
  }> {
    const result: { native?: RecognizedPlate; fast?: RecognizedPlate; region?: string } = {}

    const tNative = performance.now()
    try {
      const ocr = await this.nativeOcr.recognizePlateText(plateCrop)
      if (ocr) {
        const value = cleanPlate(ocr.text)
        if (looksLikePlate(value)) {
          result.native = { value, raw: ocr.text.trim(), confidence: ocr.confidence }
        }
      }
    } catch (error) {
      warnings.push(`OCR (production): ${messageOf(error)}`)
    }
    timings.ocrNativeMs += performance.now() - tNative

    if (options.fastPlate) {
      const tFast = performance.now()
      try {
        const fast = await this.fastPlateOcr.recognize(plateCrop)
        if (fast && fast.plate) {
          result.fast = { value: fast.plate, raw: fast.plate, confidence: fast.confidence }
          result.region = fast.region
        }
      } catch (error) {
        warnings.push(`OCR (fast-plate-ocr): ${messageOf(error)}`)
      }
      timings.ocrFastMs += performance.now() - tFast
    }
    return result
  }

  /** Face belonging to this person: face box center inside the person box */
  private async pickPersonFace(
    person: YoloDetection,
    getFaces: () => Promise<DetectedFace[]>,
  ): Promise<DetectedFace | null> {
    const faces = await getFaces()
    let best: DetectedFace | null = null
    let bestArea = 0
    for (const face of faces) {
      const cx = (face.box.x1 + face.box.x2) / 2
      const cy = (face.box.y1 + face.box.y2) / 2
      const inside = cx >= person.x1 && cx <= person.x2 && cy >= person.y1 && cy <= person.y2
      if (!inside) {
        continue
      }
      const area = (face.box.x2 - face.box.x1) * (face.box.y2 - face.box.y1)
      if (area > bestArea) {
        bestArea = area
        best = face
      }
    }
    return best
  }
}

function toNormalized(box: { x1: number; y1: number; x2: number; y2: number }, img: RgbImage): RecognizedBbox {
  const x1 = clampPixel(box.x1, 0, img.width - 1) / img.width
  const y1 = clampPixel(box.y1, 0, img.height - 1) / img.height
  const x2 = clampPixel(box.x2, 1, img.width) / img.width
  const y2 = clampPixel(box.y2, 1, img.height) / img.height
  return { x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) }
}

/** Box expansion — margin for a readable plate crop */
function expand(box: { x1: number; y1: number; x2: number; y2: number }, margin: number) {
  const w = box.x2 - box.x1
  const h = box.y2 - box.y1
  return {
    x1: box.x1 - w * margin,
    y1: box.y1 - h * margin,
    x2: box.x2 + w * margin,
    y2: box.y2 + h * margin,
  }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
