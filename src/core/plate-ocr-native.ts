/**
 * Plate text OCR with a PaddleOCR rec model: CTC greedy decoding,
 * small-crop upscaling, ±40° deskew.
 */

import * as ort from 'onnxruntime-web'
import { resizeRgb, rgbToChw, warpAffineRgb } from './tensor.util'
import type { RgbImage } from './types'
import { MODEL_FILES, OCR_DICT_FILE, modelRegistry } from './model-registry'

/** PP-OCRv5 rec model input height (width is dynamic) */
const REC_HEIGHT = 48
const REC_MAX_WIDTH = 640
/** Minimum upscale width for small plate crops before OCR */
const PLATE_MIN_WIDTH = 128

/** Plate deskew: angle sweep ±DESKEW_MAX_ANGLE with DESKEW_STEP */
const DESKEW_MAX_ANGLE = 40
const DESKEW_STEP = 2

export interface OcrTextResult {
  text: string
  confidence: number
}

/** Canonical plate form: uppercase, Latin letters and digits only */
const cleanPlateText = (text: string) => (text || '').toUpperCase().replace(/[^A-Z0-9]/g, '')

/** Reading looks like a plate: 4–10 chars and at least 2 digits (UA format) */
const plateLike = (value: string) =>
  value.length >= 4 && value.length <= 10 && /\d.*\d/.test(value)

/**
 * Rotation around the center WITHOUT canvas expansion (dst→src matrix for
 * warpAffineRgb): size and center are preserved, frame corners get cut off.
 */
function rotateAboutCenter(img: RgbImage, angleDeg: number): RgbImage {
  const rad = (angleDeg * Math.PI) / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  const m = [cos, -sin, 0, sin, cos, 0]
  m[2] = (img.width * (1 - cos)) / 2 + (img.height * sin) / 2
  m[5] = (img.height * (1 - cos)) / 2 - (img.width * sin) / 2
  return warpAffineRgb(img, m, img.width, img.height)
}

/**
 * "Text line is horizontal" score: variance of the horizontal brightness
 * gradient profile — vertical letter strokes give a sharp peak when the
 * text is aligned; polarity independent (dark on light and vice versa).
 */
function horizontalTextScore(img: RgbImage): number {
  const rowSum = new Float64Array(img.height)
  const rowBytes = img.width * 3
  for (let y = 0; y < img.height; y++) {
    let sum = 0
    const row = y * rowBytes
    for (let x = 3; x < rowBytes; x += 3) {
      sum += Math.abs(img.data[row + x] - img.data[row + x - 3])
    }
    rowSum[y] = sum
  }
  let mean = 0
  for (const value of rowSum) {
    mean += value
  }
  mean /= rowSum.length || 1
  let variance = 0
  for (const value of rowSum) {
    variance += (value - mean) * (value - mean)
  }
  return variance / (rowSum.length || 1)
}

export class NativePlateOcr {
  /** [blank slot, ...dictionary chars]; index = CTC output class */
  private charset: string[] | null = null
  /** CTC blank class: in this export blank occupies the last slot (space slot) */
  private blankClass = -1

  /** Read text from a crop (single line: plate, sign, etc.) */
  async recognizeText(img: RgbImage): Promise<OcrTextResult | null> {
    if (img.width < 2 || img.height < 2) {
      return null
    }
    // Small plates are upscaled — CTC reads poorly on narrow inputs
    const source = this.upscaleSmall(img)
    const width = Math.max(
      16,
      Math.min(REC_MAX_WIDTH, Math.round((source.width / source.height) * REC_HEIGHT)),
    )
    const resized = resizeRgb(source, width, REC_HEIGHT)
    const chw = rgbToChw(resized, v => (v / 255 - 0.5) / 0.5)

    const session = await modelRegistry.get(MODEL_FILES.ocr)
    const results = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, REC_HEIGHT, width]),
    })

    const output = results[session.outputNames[0]]
    const dims = output.dims
    const classes = dims[dims.length - 1]
    const steps = dims[dims.length - 2]
    const raw = output.data as Float32Array
    const charset = await this.loadCharset(classes)

    // CTC greedy: argmax over classes, collapse repeats, skip blanks
    let text = ''
    let confidenceSum = 0
    let confidenceCount = 0
    let previous = -1
    for (let step = 0; step < steps; step++) {
      let bestClass = 0
      let bestScore = -1
      const offset = step * classes
      for (let c = 0; c < classes; c++) {
        const score = raw[offset + c]
        if (score > bestScore) {
          bestScore = score
          bestClass = c
        }
      }
      // Skip blank slots: class 0 (the '#' dictionary slot — between each
      // char in the decode) and the last class; anything else — charset[c]
      if (bestClass !== 0 && bestClass !== this.blankClass && bestClass !== previous) {
        const char = charset[bestClass] || ''
        if (char) {
          text += char
          confidenceSum += bestScore
          confidenceCount++
        }
      }
      previous = bestClass
    }
    if (!text) {
      return null
    }
    return { text, confidence: confidenceCount > 0 ? confidenceSum / confidenceCount : 0 }
  }

  /**
   * Plate reading from a plate crop with tilt compensation: direct reading
   * short-circuits on a confident "plate-like" result, otherwise the crop
   * is straightened (text-projection deskew) and the more plausible of the
   * two readings wins. Additionally returns the deskewed crop for the
   * debug panel.
   */
  async recognizePlateText(
    img: RgbImage,
  ): Promise<OcrTextResult & { deskewed: RgbImage | null } | null> {
    if (img.width < 8 || img.height < 8) {
      const simple = await this.recognizeText(img)
      return simple ? { ...simple, deskewed: null } : null
    }
    // Deskew over the upscaled crop: gradients are more reliable there
    const upscaled = this.upscaleSmall(img)
    const direct = await this.recognizeText(upscaled)
    if (direct && plateLike(cleanPlateText(direct.text)) && direct.confidence >= 0.8) {
      return { ...direct, deskewed: null }
    }
    const angle = this.bestDeskewAngle(upscaled)
    if (angle === 0) {
      return direct ? { ...direct, deskewed: null } : null
    }
    const rotated = rotateAboutCenter(upscaled, angle)
    const corrected = await this.recognizeText(rotated)
    if (!corrected) {
      return direct ? { ...direct, deskewed: rotated } : null
    }
    if (!direct) {
      return { ...corrected, deskewed: rotated }
    }
    const directPlate = plateLike(cleanPlateText(direct.text))
    const correctedPlate = plateLike(cleanPlateText(corrected.text))
    if (directPlate !== correctedPlate) {
      return { ...(directPlate ? direct : corrected), deskewed: rotated }
    }
    return {
      ...(corrected.confidence > direct.confidence ? corrected : direct),
      deskewed: rotated,
    }
  }

  /** Upscale a small crop (same rule as recognizeText) */
  private upscaleSmall(img: RgbImage): RgbImage {
    if (img.width >= PLATE_MIN_WIDTH) {
      return img
    }
    const upscale = Math.min(4, Math.ceil(PLATE_MIN_WIDTH / img.width))
    return resizeRgb(img, img.width * upscale, img.height * upscale)
  }

  /**
   * Crop deskew: coarse angle sweep with DESKEW_STEP, then refinement
   * ±DESKEW_STEP with 1° steps; 0 stays the winner on horizontal text.
   */
  private bestDeskewAngle(img: RgbImage): number {
    let bestAngle = 0
    let bestScore = horizontalTextScore(img)
    for (let angle = -DESKEW_MAX_ANGLE; angle <= DESKEW_MAX_ANGLE; angle += DESKEW_STEP) {
      if (angle === 0) {
        continue
      }
      const score = horizontalTextScore(rotateAboutCenter(img, angle))
      if (score > bestScore) {
        bestScore = score
        bestAngle = angle
      }
    }
    for (let fine = bestAngle - DESKEW_STEP + 1; fine <= bestAngle + DESKEW_STEP - 1; fine++) {
      if (fine === 0 || fine === bestAngle) {
        continue
      }
      const score = horizontalTextScore(rotateAboutCenter(img, fine))
      if (score > bestScore) {
        bestScore = score
        bestAngle = fine
      }
    }
    return bestAngle
  }

  /** blank (index 0) + dictionary; length adjusted to the model class count */
  private async loadCharset(classes: number): Promise<string[]> {
    if (this.charset) {
      return this.charset
    }
    let lines: string[] = []
    try {
      const content = await modelRegistry.fetchModelText(OCR_DICT_FILE)
      lines = content.split('\n')
      if (lines.length > 0 && lines[lines.length - 1] === '') {
        lines.pop()
      }
    } catch {
      lines = []
    }
    // Class c → dictionary[c]; length adjusted to the model class count.
    // Verified against references: in this export the blank class is the
    // last one (space slot), not the zeroth as in canonical PaddleOCR.
    const charset = [...lines]
    while (charset.length < classes) {
      charset.push(' ')
    }
    if (charset.length > classes) {
      charset.length = classes
    }
    this.blankClass = charset.length - 1
    charset[this.blankClass] = ''
    this.charset = charset
    return charset
  }
}
