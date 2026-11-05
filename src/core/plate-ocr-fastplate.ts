/**
 * fast-plate-ocr in the browser (https://github.com/ankandrew/fast-plate-ocr):
 * the same ONNX model (cct-xs-v2-global-model) the Python package uses, but
 * via onnxruntime-web. Preprocessing/decoding follow fast_plate_ocr.core.process:
 * resize to img_height×img_width (stretch, keep_aspect_ratio=false), uint8
 * RGB NHWC without manual normalization (the model normalizes internally),
 * decode — reshape (N, max_plate_slots, vocab) → argmax → alphabet, trailing
 * pad_char trim (rstrip), confidence — per-slot max probabilities.
 */

import * as ort from 'onnxruntime-web'
import { resizeRgb } from './tensor.util'
import type { RgbImage } from './types'
import { MODEL_FILES, modelRegistry } from './model-registry'

export interface FastPlateConfig {
  max_plate_slots: number
  alphabet: string
  pad_char: string
  img_height: number
  img_width: number
  keep_aspect_ratio: boolean
  image_color_mode: 'grayscale' | 'rgb'
  plate_regions?: string[]
}

export interface FastPlateResult {
  plate: string
  confidence: number
  region?: string
  regionProb?: number
}

export class FastPlateOcr {
  private config: FastPlateConfig | null = null
  private loadPromise: Promise<void> | null = null

  get loaded(): boolean {
    return this.config !== null
  }

  /** Config + session (lazy); errors propagate to the caller */
  async ensureLoaded(): Promise<void> {
    if (this.config) {
      return
    }
    if (!this.loadPromise) {
      this.loadPromise = (async () => {
        const res = await fetch(modelRegistry.modelUrl('fastplate/config.json'))
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: fast-plate-ocr config (${res.url})`)
        }
        this.config = (await res.json()) as FastPlateConfig
        await modelRegistry.get(MODEL_FILES.fastPlate)
      })()
      this.loadPromise.catch(() => {
        // Allow a retry after a load failure
        this.loadPromise = null
      })
    }
    return this.loadPromise
  }

  /** Read a plate from a plate crop */
  async recognize(img: RgbImage): Promise<FastPlateResult | null> {
    await this.ensureLoaded()
    const cfg = this.config!
    if (img.width < 4 || img.height < 4) {
      return null
    }

    // fast-plate-ocr expects uint8 (H, W, C) without manual normalization;
    // keep_aspect_ratio=false → plain stretch (bilinear, like cv2.INTER_LINEAR)
    const resized = resizeRgb(img, cfg.img_width, cfg.img_height)
    const channels = cfg.image_color_mode === 'rgb' ? 3 : 1
    const frame = new Uint8Array(cfg.img_height * cfg.img_width * channels)
    for (let y = 0; y < cfg.img_height; y++) {
      for (let x = 0; x < cfg.img_width; x++) {
        const src = (y * resized.width + x) * 3
        const dst = (y * cfg.img_width + x) * channels
        frame[dst] = resized.data[src]
        if (channels === 3) {
          frame[dst + 1] = resized.data[src + 1]
          frame[dst + 2] = resized.data[src + 2]
        }
      }
    }

    const session = await modelRegistry.get(MODEL_FILES.fastPlate)
    const results = await session.run({
      [session.inputNames[0]]: new ort.Tensor('uint8', frame, [
        1,
        cfg.img_height,
        cfg.img_width,
        channels,
      ]),
    })

    // Main output 'plate' (or the first one); shape (1, max_plate_slots, vocab)
    const plateName = session.outputNames.find(n => n === 'plate') || session.outputNames[0]
    const raw = results[plateName].data as Float32Array
    const vocab = cfg.alphabet.length
    const slots = cfg.max_plate_slots

    // argmax over each slot (predictions reshaped to (N, slots, vocab))
    const indices: number[] = []
    const probs: number[] = []
    for (let slot = 0; slot < slots; slot++) {
      let bestClass = 0
      let bestScore = -1
      const offset = slot * vocab
      for (let c = 0; c < vocab; c++) {
        const score = raw[offset + c]
        if (score > bestScore) {
          bestScore = score
          bestClass = c
        }
      }
      indices.push(bestClass)
      probs.push(bestScore)
    }

    // remove_pad_char=True: rstrip(pad_char) over the decoded string
    let last = slots
    while (last > 0 && cfg.alphabet[indices[last - 1]] === cfg.pad_char) {
      last--
    }
    let plate = ''
    let confidenceSum = 0
    for (let slot = 0; slot < last; slot++) {
      plate += cfg.alphabet[indices[slot]]
      confidenceSum += probs[slot]
    }
    plate = plate.replace(/\s+/g, '')
    if (!plate) {
      return null
    }
    const result: FastPlateResult = {
      plate,
      confidence: last > 0 ? confidenceSum / last : 0,
    }

    // Region head (v2 models): region probabilities
    const regionName = session.outputNames.find(n => n === 'region')
    if (regionName && cfg.plate_regions?.length) {
      const regionOut = results[regionName].data as Float32Array
      let bestIdx = 0
      for (let i = 1; i < regionOut.length; i++) {
        if (regionOut[i] > regionOut[bestIdx]) {
          bestIdx = i
        }
      }
      result.region = cfg.plate_regions[bestIdx]
      result.regionProb = regionOut[bestIdx]
    }
    return result
  }
}
