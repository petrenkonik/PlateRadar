/**
 * Car body color: k-means over the car crop pixels, cluster-center matching
 * against a name palette. The palette names are Ukrainian — they are the
 * production output values, kept as-is.
 */

import type { PixelBox, RgbImage } from './types'

/** Color name palette (uk) and RGB centers for cluster matching */
const PALETTE: [string, [number, number, number]][] = [
  ['білий', [240, 240, 240]],
  ['сріблястий', [190, 190, 194]],
  ['сірий', [126, 126, 130]],
  ['чорний', [28, 28, 32]],
  ['червоний', [190, 30, 35]],
  ['помаранчевий', [235, 125, 35]],
  ['жовтий', [235, 200, 55]],
  ['зелений', [55, 140, 65]],
  ['синій', [40, 80, 175]],
  ['блакитний', [115, 180, 225]],
  ['фіолетовий', [115, 60, 175]],
  ['рожевий', [225, 120, 160]],
  ['коричневий', [105, 70, 45]],
  ['бежевий', [215, 195, 165]],
  ['золотий', [205, 165, 60]],
]

const KMEANS_CLUSTERS = 4
const KMEANS_ITERATIONS = 8
/** Share of pixels a cluster needs to count as the body, not noise */
const MIN_CLUSTER_SHARE = 0.12

const rgbToHsv = (r: number, g: number, b: number): [number, number, number] => {
  const rn = r / 255
  const gn = g / 255
  const bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const d = max - min
  let h = 0
  if (d > 0) {
    if (max === rn) h = ((gn - bn) / d) % 6
    else if (max === gn) h = (bn - rn) / d + 2
    else h = (rn - gn) / d + 4
    h /= 6
    if (h < 0) h += 1
  }
  return [h, max > 0 ? d / max : 0, max]
}

/**
 * Body color: k-means over the car crop pixels (plate and crop border
 * excluded); the candidate cluster is large enough and the most "colored",
 * then its center is matched against the name palette.
 */
export function dominantCarColor(img: RgbImage, exclude?: PixelBox): string | undefined {
  if (img.width < 8 || img.height < 8) {
    return undefined
  }

  // Pixel sampling: ~4000-point grid, the 6% border is dropped (background)
  const step = Math.max(1, Math.floor(Math.sqrt((img.width * img.height) / 4000)))
  const borderX = Math.floor(img.width * 0.06)
  const borderY = Math.floor(img.height * 0.06)
  const samples: number[][] = []
  for (let y = borderY; y < img.height - borderY; y += step) {
    for (let x = borderX; x < img.width - borderX; x += step) {
      if (exclude && x >= exclude.x1 && x <= exclude.x2 && y >= exclude.y1 && y <= exclude.y2) {
        continue
      }
      const o = (y * img.width + x) * 3
      samples.push([img.data[o], img.data[o + 1], img.data[o + 2]])
    }
  }
  if (samples.length < KMEANS_CLUSTERS * 4) {
    return undefined
  }

  // Initial centers — brightness percentiles (stable and deterministic)
  const sorted = [...samples].sort((a, b) => a[0] + a[1] + a[2] - (b[0] + b[1] + b[2]))
  const centers: number[][] = []
  for (let k = 0; k < KMEANS_CLUSTERS; k++) {
    const index = Math.floor(((k + 0.5) / KMEANS_CLUSTERS) * (sorted.length - 1))
    centers.push([...sorted[index]])
  }

  const assignments = new Int32Array(samples.length)
  for (let iter = 0; iter < KMEANS_ITERATIONS; iter++) {
    for (let i = 0; i < samples.length; i++) {
      let best = 0
      let bestDist = Infinity
      for (let k = 0; k < centers.length; k++) {
        const dr = samples[i][0] - centers[k][0]
        const dg = samples[i][1] - centers[k][1]
        const db = samples[i][2] - centers[k][2]
        const dist = dr * dr + dg * dg + db * db
        if (dist < bestDist) {
          bestDist = dist
          best = k
        }
      }
      assignments[i] = best
    }
    const sums = centers.map(() => [0, 0, 0, 0])
    for (let i = 0; i < samples.length; i++) {
      const cluster = assignments[i]
      sums[cluster][0] += samples[i][0]
      sums[cluster][1] += samples[i][1]
      sums[cluster][2] += samples[i][2]
      sums[cluster][3]++
    }
    for (let k = 0; k < centers.length; k++) {
      if (sums[k][3] > 0) {
        centers[k] = [sums[k][0] / sums[k][3], sums[k][1] / sums[k][3], sums[k][2] / sums[k][3]]
      }
    }
  }

  const counts = new Array(centers.length).fill(0)
  for (let i = 0; i < assignments.length; i++) {
    counts[assignments[i]]++
  }

  // Candidates: clusters of sufficient size; saturated color preferred
  // (gray asphalt and black shadows lose to a painted body)
  let bestCluster = -1
  let bestScore = -1
  for (let k = 0; k < centers.length; k++) {
    const share = counts[k] / samples.length
    if (share < MIN_CLUSTER_SHARE) {
      continue
    }
    const [, saturation] = rgbToHsv(centers[k][0], centers[k][1], centers[k][2])
    const score = saturation + share * 0.5
    if (score > bestScore) {
      bestScore = score
      bestCluster = k
    }
  }
  if (bestCluster < 0) {
    let maxCount = 0
    for (let k = 0; k < counts.length; k++) {
      if (counts[k] > maxCount) {
        maxCount = counts[k]
        bestCluster = k
      }
    }
  }
  if (bestCluster < 0) {
    return undefined
  }

  const [r, g, b] = centers[bestCluster]
  let bestName = PALETTE[0][0]
  let bestDist = Infinity
  for (const [name, [pr, pg, pb]] of PALETTE) {
    const dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
    if (dist < bestDist) {
      bestDist = dist
      bestName = name
    }
  }
  return bestName
}
