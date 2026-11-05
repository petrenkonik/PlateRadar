/**
 * Plate tracking with multi-frame voting.
 *
 * Raw per-frame OCR readings are noisy: one random frame can produce a
 * false or garbled plate. The tracker associates readings across frames
 * into tracks (by text similarity + box overlap), keeps a sliding window
 * of the last N observations per track, and votes for the winning text
 * variant. A track is "confirmed" only when one text variant gathers
 * enough consistent votes — single-frame false positives are outvoted and
 * never reach the overlay or history. While a track is alive, its box is
 * smoothed (exponential moving average), so the frame keeps a stable box
 * even when a particular frame fails to detect the plate ("interpolation"
 * across frames).
 */

import type { PixelBox } from './types'

/** One per-frame plate reading accepted into a track */
export interface PlateObservation {
  /** Normalized text (Latin/digits only) */
  text: string
  /** OCR confidence of the reading 0..1 */
  confidence: number
  /** Plate box, normalized 0..1 frame coordinates */
  plateBox: PixelBox
  /** Vehicle box, normalized 0..1 (optional — close-up plates have none) */
  carBox?: PixelBox
  /** Vehicle detection score 0..1 */
  carScore?: number
  /** Region/country prediction from fast-plate-ocr */
  region?: string
  /** Body color (when the stage is enabled) */
  color?: string
  /** Raw production OCR reading of the same frame (comparison) */
  native?: string
  /** Raw fast-plate-ocr reading of the same frame (comparison) */
  fast?: string
  /** Debug crop dataURL captured with this observation */
  thumb?: string
  /** monotonic ms */
  time: number
}

export interface PlateTrack {
  id: number
  /** Winning (voted) text */
  text: string
  /** Mean confidence of the winning variant's observations */
  confidence: number
  /** Votes of the winning variant */
  votes: number
  /** Observations currently kept in the window */
  observations: number
  /** votes / observations (share of consistency) */
  share: number
  /** Region of the most recent observation */
  region?: string
  /** Smoothed plate box (normalized 0..1) */
  plateBox: PixelBox
  /** Smoothed vehicle box (normalized 0..1) */
  carBox?: PixelBox
  /** Latest vehicle detection score */
  carScore: number
  /** Latest body color */
  color?: string
  /** Latest raw production OCR reading */
  nativeLast?: string
  /** Latest raw fast-plate-ocr reading */
  fastLast?: string
  /** Latest debug crop */
  thumb?: string
  /** true once the winning variant has enough consistent votes */
  confirmed: boolean
  /** last update ms */
  lastSeen: number
}

interface TrackState {
  id: number
  window: PlateObservation[]
  plateBox: PixelBox
  carBox?: PixelBox
  carScore: number
  color?: string
  region?: string
  thumb?: string
  nativeLast?: string
  fastLast?: string
  lastSeen: number
  /** analysis frames since the last observation (0 right after absorb) */
  gap: number
}

/** Normalized box overlap */
function iou(a: PixelBox, b: PixelBox): number {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  return inter / Math.max(areaA + areaB - inter, 1e-6)
}

/** Levenshtein edit distance (small strings, plain DP) */
function levenshtein(a: string, b: string): number {
  const m = a.length
  const n = b.length
  if (!m) return n
  if (!n) return m
  const prev = new Uint16Array(n + 1)
  const curr = new Uint16Array(n + 1)
  for (let j = 0; j <= n; j++) prev[j] = j
  for (let i = 1; i <= m; i++) {
    curr[0] = i
    for (let j = 1; j <= n; j++) {
      curr[j] = Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      )
    }
    prev.set(curr)
  }
  return prev[n]
}

/** 1.0 — identical, 0.0 — completely different */
function textSimilarity(a: string, b: string): number {
  if (a === b) return 1
  const dist = levenshtein(a, b)
  return 1 - dist / Math.max(a.length, b.length)
}

const lerp = (a: number, b: number, alpha: number) => a + (b - a) * alpha

function lerpBox(prev: PixelBox, next: PixelBox, alpha: number): PixelBox {
  return {
    x1: lerp(prev.x1, next.x1, alpha),
    y1: lerp(prev.y1, next.y1, alpha),
    x2: lerp(prev.x2, next.x2, alpha),
    y2: lerp(prev.y2, next.y2, alpha),
  }
}

export interface TrackerOptions {
  /** Sliding window size per track (frames) */
  window: number
  /** Votes the winning variant needs for confirmation */
  minVotes: number
  /** Minimal consistency share for confirmation (votes / window) */
  minShare: number
  /** Track is dropped after this many analysis frames without updates */
  maxGapFrames: number
  /** Text similarity threshold for associating readings into one track */
  textSimilarity: number
  /** Text similarity that merges readings even when boxes don't overlap */
  strongTextSimilarity: number
  /** Plate box IoU threshold for associating readings into one track */
  boxIou: number
  /** Box exponential smoothing factor (0..1, higher = snappier) */
  boxEma: number
}

export const defaultTrackerOptions = (): TrackerOptions => ({
  window: 10,
  minVotes: 3,
  minShare: 0.5,
  maxGapFrames: 12,
  textSimilarity: 0.75,
  strongTextSimilarity: 0.85,
  boxIou: 0.25,
  boxEma: 0.45,
})

export class PlateTracker {
  private options: TrackerOptions
  private readonly states = new Map<number, TrackState>()
  private nextId = 0

  constructor(options: Partial<TrackerOptions> = {}) {
    this.options = { ...defaultTrackerOptions(), ...options }
  }

  /** Re-apply algorithm parameters (picked up by the next analysis pass) */
  configure(options: Partial<TrackerOptions>): void {
    this.options = { ...this.options, ...options }
  }

  /** Live tracks (any state) — newest first */
  get all(): PlateTrack[] {
    return [...this.states.values()]
      .map(s => this.snapshot(s))
      .sort((a, b) => b.lastSeen - a.lastSeen)
  }

  /** Confirmed tracks only — safe to show and record */
  get confirmed(): PlateTrack[] {
    return this.all.filter(t => t.confirmed)
  }

  /**
   * Feed one frame's readings and get the refreshed tracks.
   * Passing an empty array ages the tracks (a frame where nothing was
   * detected must not reset tracking — the smoothed box persists).
   */
  update(observations: PlateObservation[], now: number): PlateTrack[] {
    // Age out dead tracks first (in analysis frames, robust to any loop speed)
    for (const [id, state] of this.states) {
      state.gap++
      if (state.gap > this.options.maxGapFrames) {
        this.states.delete(id)
      }
    }
    for (const obs of observations) {
      const state = this.match(obs)
      if (state) {
        this.absorb(state, obs, now)
      } else {
        this.create(obs, now)
      }
    }
    return this.all
  }

  clear() {
    this.states.clear()
  }

  /** Find a track this observation belongs to: text match first, then box match */
  private match(obs: PlateObservation): TrackState | null {
    let best: TrackState | null = null
    let bestScore = 0
    for (const state of this.states.values()) {
      const lastText = state.window[state.window.length - 1]?.text
      const textSim = lastText ? textSimilarity(lastText, obs.text) : 0
      const overlap = iou(state.plateBox, obs.plateBox)
      // Same/similar text at the same place — the same plate. A very
      // strong text match (one substitution on a short string) merges
      // even without box overlap: seek-stepped sampling can jump the
      // detection box between far and close views of the same plate.
      const byText =
        textSim >= this.options.textSimilarity &&
        (overlap >= 0.05 || textSim >= this.options.strongTextSimilarity)
      // Different text but the same physical box — trust motion continuity
      const byBox = overlap >= this.options.boxIou + 0.1
      if (!byText && !byBox) {
        continue
      }
      const score = textSim * 2 + overlap
      if (score > bestScore) {
        bestScore = score
        best = state
      }
    }
    return best
  }

  private absorb(state: TrackState, obs: PlateObservation, now: number) {
    state.window.push(obs)
    if (state.window.length > this.options.window) {
      state.window.shift()
    }
    // Smoothed box: follow the plate between frames (EMA, configurable)
    state.plateBox = lerpBox(state.plateBox, obs.plateBox, this.options.boxEma)
    if (obs.carBox) {
      state.carBox = state.carBox
        ? lerpBox(state.carBox, obs.carBox, this.options.boxEma)
        : obs.carBox
    }
    if (obs.carScore !== undefined) {
      state.carScore = obs.carScore
    }
    if (obs.region) state.region = obs.region
    if (obs.color) state.color = obs.color
    if (obs.thumb) state.thumb = obs.thumb
    if (obs.native) state.nativeLast = obs.native
    if (obs.fast) state.fastLast = obs.fast
    state.gap = 0
    state.lastSeen = now
  }

  private create(obs: PlateObservation, now: number) {
    const id = ++this.nextId
    this.states.set(id, {
      id,
      window: [obs],
      plateBox: obs.plateBox,
      carBox: obs.carBox,
      carScore: obs.carScore ?? 0,
      color: obs.color,
      region: obs.region,
      thumb: obs.thumb,
      nativeLast: obs.native,
      fastLast: obs.fast,
      lastSeen: now,
      gap: 0,
    })
  }

  /** Vote over the window: the most frequent text variant wins */
  private snapshot(state: TrackState): PlateTrack {
    const votes = new Map<string, { count: number; confSum: number }>()
    for (const obs of state.window) {
      const entry = votes.get(obs.text) || { count: 0, confSum: 0 }
      entry.count++
      entry.confSum += obs.confidence
      votes.set(obs.text, entry)
    }
    let bestText = ''
    let bestCount = 0
    let bestConfSum = 0
    for (const [text, { count, confSum }] of votes) {
      // More votes win; on a tie — the more confident variant
      const better =
        count > bestCount ||
        (count === bestCount && count > 0 && confSum / count > bestConfSum / bestCount)
      if (better) {
        bestText = text
        bestCount = count
        bestConfSum = confSum
      }
    }
    const observations = state.window.length
    const confirmed =
      bestCount >= this.options.minVotes && bestCount / Math.max(observations, 1) >= this.options.minShare
    return {
      id: state.id,
      text: bestText,
      confidence: bestCount > 0 ? bestConfSum / bestCount : 0,
      votes: bestCount,
      observations,
      share: observations > 0 ? bestCount / observations : 0,
      region: state.region,
      plateBox: state.plateBox,
      carBox: state.carBox,
      carScore: state.carScore,
      color: state.color,
      nativeLast: state.nativeLast,
      fastLast: state.fastLast,
      thumb: state.thumb,
      confirmed,
      lastSeen: state.lastSeen,
    }
  }
}
