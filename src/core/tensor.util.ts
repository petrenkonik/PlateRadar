/**
 * Image-to-ONNX-tensor conversion utilities:
 * Uint8Array-based RGB buffers, plus browser helpers for ImageData/canvas
 * (decode step instead of a native image library).
 */

import type { PixelBox, RgbImage } from './types'

export const clampPixel = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value))

/** ImageData (RGBA canvas) → flat RGB buffer of the engine */
export function rgbaToRgb(imageData: ImageData): RgbImage {
  const { width, height } = imageData
  const rgba = imageData.data
  const data = new Uint8Array(width * height * 3)
  for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
    data[i] = rgba[j]
    data[i + 1] = rgba[j + 1]
    data[i + 2] = rgba[j + 2]
  }
  return { data, width, height }
}

/** Scale a frame from video/img/canvas into an RgbImage (resizes at the same time) */
export function captureScaled(
  source: CanvasImageSource,
  srcWidth: number,
  srcHeight: number,
  maxWidth: number,
): { img: RgbImage; scale: number } {
  const scale = srcWidth > maxWidth ? maxWidth / srcWidth : 1
  const width = Math.max(2, Math.round(srcWidth * scale))
  const height = Math.max(2, Math.round(srcHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) {
    throw new Error('canvas 2d context is not available')
  }
  ctx.drawImage(source, 0, 0, width, height)
  return { img: rgbaToRgb(ctx.getImageData(0, 0, width, height)), scale }
}

/** RgbImage crop → dataURL (for the debug panel and history) */
export function rgbCropToDataUrl(img: RgbImage, box: PixelBox, quality = 0.7): string | undefined {
  const crop = cropRgb(img, box)
  if (crop.width < 4 || crop.height < 4) {
    return undefined
  }
  const canvas = document.createElement('canvas')
  canvas.width = crop.width
  canvas.height = crop.height
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return undefined
  }
  const imageData = ctx.createImageData(crop.width, crop.height)
  for (let i = 0, j = 0; i < crop.data.length; i += 3, j += 4) {
    imageData.data[j] = crop.data[i]
    imageData.data[j + 1] = crop.data[i + 1]
    imageData.data[j + 2] = crop.data[i + 2]
    imageData.data[j + 3] = 255
  }
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/jpeg', quality)
}

/** Load an image file into RgbImage (for the single-image test tab) */
export async function imageFileToRgb(file: File, maxWidth = 1280): Promise<RgbImage> {
  const url = URL.createObjectURL(file)
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image()
      el.onload = () => resolve(el)
      el.onerror = () => reject(new Error('Failed to decode the image'))
      el.src = url
    })
    return captureScaled(img, img.naturalWidth, img.naturalHeight, maxWidth).img
  } finally {
    URL.revokeObjectURL(url)
  }
}

/** Bilinear pixel sampling; out-of-bounds reads clamp to the image edge */
function sampleBilinear(img: RgbImage, x: number, y: number, out: [number, number, number]) {
  const cx = clampPixel(x, 0, img.width - 1)
  const cy = clampPixel(y, 0, img.height - 1)
  const x0 = Math.floor(cx)
  const y0 = Math.floor(cy)
  const x1 = Math.min(x0 + 1, img.width - 1)
  const y1 = Math.min(y0 + 1, img.height - 1)
  const fx = cx - x0
  const fy = cy - y0
  for (let c = 0; c < 3; c++) {
    const top = img.data[(y0 * img.width + x0) * 3 + c] * (1 - fx) + img.data[(y0 * img.width + x1) * 3 + c] * fx
    const bottom = img.data[(y1 * img.width + x0) * 3 + c] * (1 - fx) + img.data[(y1 * img.width + x1) * 3 + c] * fx
    out[c] = top * (1 - fy) + bottom * fy
  }
}

/** Bilinear RGB resize */
export function resizeRgb(img: RgbImage, dstWidth: number, dstHeight: number): RgbImage {
  const data = new Uint8Array(dstWidth * dstHeight * 3)
  const sx = dstWidth > 1 ? (img.width - 1) / (dstWidth - 1) : 0
  const sy = dstHeight > 1 ? (img.height - 1) / (dstHeight - 1) : 0
  const pixel: [number, number, number] = [0, 0, 0]
  for (let y = 0; y < dstHeight; y++) {
    for (let x = 0; x < dstWidth; x++) {
      sampleBilinear(img, x * sx, y * sy, pixel)
      const o = (y * dstWidth + x) * 3
      data[o] = pixel[0]
      data[o + 1] = pixel[1]
      data[o + 2] = pixel[2]
    }
  }
  return { data, width: dstWidth, height: dstHeight }
}

/** Crop a pixel region (clamped to the image bounds) */
export function cropRgb(img: RgbImage, box: PixelBox): RgbImage {
  const x1 = Math.round(clampPixel(box.x1, 0, img.width - 1))
  const y1 = Math.round(clampPixel(box.y1, 0, img.height - 1))
  const x2 = Math.round(clampPixel(box.x2, x1 + 1, img.width))
  const y2 = Math.round(clampPixel(box.y2, y1 + 1, img.height))
  const width = x2 - x1
  const height = y2 - y1
  if (width <= 0 || height <= 0) {
    return { data: new Uint8Array(3), width: 1, height: 1 }
  }
  if (x1 === 0 && width === img.width) {
    return { data: img.data.subarray(y1 * img.width * 3, y2 * img.width * 3), width, height }
  }
  const data = new Uint8Array(width * height * 3)
  for (let row = 0; row < height; row++) {
    const src = ((y1 + row) * img.width + x1) * 3
    data.set(img.data.subarray(src, src + width * 3), row * width * 3)
  }
  return { data, width, height }
}

/** CHW Float32 from RGB with per-pixel normalization */
export function rgbToChw(img: RgbImage, normalize: (channelValue: number) => number): Float32Array {
  const pixels = img.width * img.height
  const out = new Float32Array(3 * pixels)
  const src = img.data
  for (let i = 0; i < pixels; i++) {
    out[i] = normalize(src[i * 3])
    out[pixels + i] = normalize(src[i * 3 + 1])
    out[pixels * 2 + i] = normalize(src[i * 3 + 2])
  }
  return out
}

export interface LetterboxResult {
  data: Float32Array
  /** original → letterbox scale */
  ratio: number
  padX: number
  padY: number
}

/**
 * Letterbox: fit the image into a size×size square with gray padding,
 * straight into CHW Float32 (normalization is set by the normalizer,
 * including the pad color — the normalized padValue).
 */
export function letterboxToTensor(
  img: RgbImage,
  size: number,
  normalize: (channelValue: number) => number,
  padValue = 114,
): LetterboxResult {
  const ratio = Math.min(size / img.width, size / img.height)
  const newWidth = Math.max(1, Math.round(img.width * ratio))
  const newHeight = Math.max(1, Math.round(img.height * ratio))
  const padX = Math.floor((size - newWidth) / 2)
  const padY = Math.floor((size - newHeight) / 2)

  const data = new Float32Array(3 * size * size).fill(normalize(padValue))
  const plane = size * size
  const pixel: [number, number, number] = [0, 0, 0]
  const inv = 1 / ratio
  for (let y = 0; y < newHeight; y++) {
    const srcY = y * inv
    for (let x = 0; x < newWidth; x++) {
      sampleBilinear(img, x * inv, srcY, pixel)
      // CHW: planar layout — channel × row × column
      const dst = (y + padY) * size + x + padX
      data[dst] = normalize(pixel[0])
      data[plane + dst] = normalize(pixel[1])
      data[plane * 2 + dst] = normalize(pixel[2])
    }
  }
  return { data, ratio, padX, padY }
}

/** Inverse letterbox → original coordinate mapping (pixels, clamped) */
export function unletterbox(
  box: PixelBox,
  img: RgbImage,
  letterbox: { ratio: number; padX: number; padY: number },
): PixelBox {
  const inv = 1 / letterbox.ratio
  return {
    x1: clampPixel((box.x1 - letterbox.padX) * inv, 0, img.width - 1),
    y1: clampPixel((box.y1 - letterbox.padY) * inv, 0, img.height - 1),
    x2: clampPixel((box.x2 - letterbox.padX) * inv, 1, img.width),
    y2: clampPixel((box.y2 - letterbox.padY) * inv, 1, img.height),
  }
}

interface NmsBox {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

/** Non-Maximum Suppression; returns indices of the kept boxes */
export function nms(boxes: NmsBox[], iouThreshold: number): number[] {
  const order = boxes.map((_, index) => index).sort((a, b) => boxes[b].score - boxes[a].score)
  const suppressed = new Set<number>()
  const result: number[] = []
  for (const candidate of order) {
    if (suppressed.has(candidate)) {
      continue
    }
    result.push(candidate)
    const a = boxes[candidate]
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
    for (const other of order) {
      if (other === candidate || suppressed.has(other)) {
        continue
      }
      const b = boxes[other]
      const interX1 = Math.max(a.x1, b.x1)
      const interY1 = Math.max(a.y1, b.y1)
      const interX2 = Math.min(a.x2, b.x2)
      const interY2 = Math.min(a.y2, b.y2)
      const inter = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1)
      const union = areaA + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
      if (inter / Math.max(union, 1e-6) > iouThreshold) {
        suppressed.add(other)
      }
    }
  }
  return result
}

/**
 * Least-squares similarity transform estimate (rotation + scale + shift,
 * no reflection) over src → dst point pairs. Complex form — exact analytic
 * 2D solution. Returns affine [a11, a12, tx, a21, a22, ty].
 */
export function similarityTransform(src: number[][], dst: number[][]): number[] {
  const n = Math.min(src.length, dst.length)
  let srcRe = 0
  let srcIm = 0
  let dstRe = 0
  let dstIm = 0
  for (let i = 0; i < n; i++) {
    srcRe += src[i][0]
    srcIm += src[i][1]
    dstRe += dst[i][0]
    dstIm += dst[i][1]
  }
  srcRe /= n
  srcIm /= n
  dstRe /= n
  dstIm /= n

  let numRe = 0
  let numIm = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    const zRe = src[i][0] - srcRe
    const zIm = src[i][1] - srcIm
    const wRe = dst[i][0] - dstRe
    const wIm = dst[i][1] - dstIm
    // num += z * conj(w)
    numRe += zRe * wRe + zIm * wIm
    numIm += zIm * wRe - zRe * wIm
    den += zRe * zRe + zIm * zIm
  }
  const aRe = den > 1e-12 ? numRe / den : 1
  const aIm = den > 1e-12 ? numIm / den : 0
  const tRe = dstRe - (aRe * srcRe - aIm * srcIm)
  const tIm = dstIm - (aIm * srcRe + aRe * srcIm)
  // z' = a*z + t: x' = aRe*x - aIm*y + tRe; y' = aIm*x + aRe*y + tIm
  return [aRe, -aIm, tRe, aIm, aRe, tIm]
}

/** Inverse 2×3 affine [a11, a12, tx, a21, a22, ty] */
export function invertAffine(m: number[]): number[] {
  const [a11, a12, tx, a21, a22, ty] = m
  const det = a11 * a22 - a12 * a21
  if (Math.abs(det) < 1e-12) {
    return [1, 0, 0, 0, 1, 0]
  }
  const inv = 1 / det
  return [
    a22 * inv,
    -a12 * inv,
    (a12 * ty - a22 * tx) * inv,
    -a21 * inv,
    a11 * inv,
    (a21 * tx - a11 * ty) * inv,
  ]
}

/** Warp: every output pixel is taken from src via a dst→src matrix (bilinear), RGB */
export function warpAffineRgb(img: RgbImage, m: number[], dstWidth: number, dstHeight: number): RgbImage {
  const data = new Uint8Array(dstWidth * dstHeight * 3)
  const pixel: [number, number, number] = [0, 0, 0]
  const [a11, a12, tx, a21, a22, ty] = m
  for (let y = 0; y < dstHeight; y++) {
    for (let x = 0; x < dstWidth; x++) {
      sampleBilinear(img, a11 * x + a12 * y + tx, a21 * x + a22 * y + ty, pixel)
      const o = (y * dstWidth + x) * 3
      data[o] = pixel[0]
      data[o + 1] = pixel[1]
      data[o + 2] = pixel[2]
    }
  }
  return { data, width: dstWidth, height: dstHeight }
}
