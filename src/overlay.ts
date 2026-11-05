/**
 * Overlay rendering over the video.
 *
 * Three display paths:
 *  - renderPlayback — phase 2 for video sources: the annotations are
 *    precomputed (useLoop.analyzeVideo → core/playback.ts), here they are
 *    only painted: interpolated between neighboring snapshots so the
 *    marks sit exactly on the shown video moment (no extra smoothing —
 *    the boxes must not trail the picture). No inference runs during
 *    playback.
 *  - renderAnalysisFrame — phase 1 for video sources: while the pass
 *    steps through the video, every analyzed frame is shown with its raw
 *    detections painted exactly as the analysis log records them
 *    (vehicle/person boxes with scores, raw OCR readings of both
 *    engines, the tracker's accepted variant with its votes).
 *  - renderOverlay — live camera mode: boxes come from the latest
 *    analysis result and glide toward their targets.
 *
 * Display policy (options.confirmScore, default 0.7): each mark is gated
 * by its own confidence — the vehicle box is drawn only above the vehicle
 * score threshold, the plate box with its caption only above the voted
 * plate confidence threshold.
 *
 * Anti-jitter (live camera only): rendering happens on every animation
 * frame; each displayed box glides toward its target (exponential
 * smoothing with a snap dead-zone). Playback rendering assigns targets
 * directly — the snapshot interpolation already gives smooth motion.
 * Coordinates are mapped through the video's letterbox (object-fit:
 * contain) rectangle, so boxes stay glued to the picture at any element
 * size.
 */

import type { PlaybackFrame } from './core/playback'
import type { PlateTrack } from './core/plate-tracker'
import type { PipelineResult } from './core/types'

const PERSON_COLOR = '#34c759'
const CAR_COLOR = '#4d8dff'
const PLATE_COLOR = '#ffd60a'
const LABEL_BG = 'rgba(13, 17, 23, 0.82)'

/** Smoothing factor per animation frame (0..1, higher = snappier) */
const SMOOTH_ALPHA = 0.18
/** Below this normalized distance the box snaps to the target (kills micro-jitter) */
const SNAP_EPSILON = 0.004
/** Fade-out time for disappeared tracks, ms */
const FADE_MS = 350
/** Interpolate between snapshots only when they are this close, seconds */
const PLAY_INTERP_GAP = 1.2

interface SmoothBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** Flat raw detection shape shared by both render paths */
interface RawBox {
  type: 'person' | 'car'
  score: number
  x1: number
  y1: number
  x2: number
  y2: number
}

interface TrackDisplay {
  car?: SmoothBox
  plate?: SmoothBox
  text?: string
  lastSeenFrame: number
}

/** key = track id (shared by playback and live rendering — never both) */
const displayed = new Map<number, TrackDisplay>()
let frameCounter = 0

const lerp = (a: number, b: number, alpha: number) => a + (b - a) * alpha

function lerpRect(a: SmoothBox, b: SmoothBox, alpha: number): SmoothBox {
  return {
    x1: lerp(a.x1, b.x1, alpha),
    y1: lerp(a.y1, b.y1, alpha),
    x2: lerp(a.x2, b.x2, alpha),
    y2: lerp(a.y2, b.y2, alpha),
  }
}

function smoothTo(current: SmoothBox | undefined, target: SmoothBox): SmoothBox {
  if (!current) {
    return { ...target }
  }
  const next = lerpRect(current, target, SMOOTH_ALPHA)
  // Dead-zone snap: stop the crawl when close enough to the target
  if (
    Math.abs(next.x2 - next.x1 - (target.x2 - target.x1)) < SNAP_EPSILON &&
    Math.abs(next.y2 - next.y1 - (target.y2 - target.y1)) < SNAP_EPSILON &&
    Math.abs(next.x1 - target.x1) < SNAP_EPSILON &&
    Math.abs(next.y1 - target.y1) < SNAP_EPSILON
  ) {
    return { ...target }
  }
  return next
}

/** Size the canvas bitmap to the video frame (1:1 pixels, crisp lines) */
function ensureSize(canvas: HTMLCanvasElement, video: HTMLVideoElement): boolean {
  if (!video.videoWidth || !video.videoHeight) {
    return false
  }
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
  }
  return true
}

/** object-fit: contain mapping — normalized frame coords → canvas pixels */
function letterbox(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
  const scale = Math.min(canvas.width / video.videoWidth, canvas.height / video.videoHeight)
  const contentW = video.videoWidth * scale
  const contentH = video.videoHeight * scale
  const offX = (canvas.width - contentW) / 2
  const offY = (canvas.height - contentH) / 2
  return {
    contentW,
    contentH,
    px: (nx: number) => offX + nx * contentW,
    py: (ny: number) => offY + ny * contentH,
  }
}

function labelBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  labels: string[],
  color: string,
  alpha = 1,
): void {
  const fontSize = 12
  ctx.font = `${fontSize}px 'Segoe UI', sans-serif`
  const lineHeight = fontSize + 3
  const boxWidth = Math.max(...labels.map(l => ctx.measureText(l).width)) + 10
  const boxHeight = labels.length * lineHeight + 6
  const labelY = y - boxHeight - 2 < 0 ? y + boxHeight + 2 : y - boxHeight - 2
  ctx.globalAlpha = alpha
  ctx.fillStyle = LABEL_BG
  ctx.fillRect(x, labelY, boxWidth, boxHeight)
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  ctx.strokeRect(x, labelY, boxWidth, boxHeight)
  ctx.fillStyle = color
  labels.forEach((label, i) => {
    ctx.fillText(label, x + 5, labelY + 15 + i * lineHeight)
  })
  ctx.globalAlpha = 1
}

/** Fade out tracks that disappeared, then draw all displayed boxes */
function paintTracks(
  ctx: CanvasRenderingContext2D,
  liveIds: Set<number>,
  px: (nx: number) => number,
  py: (ny: number) => number,
  contentW: number,
  contentH: number,
): void {
  for (const [id, display] of displayed) {
    if (liveIds.has(id)) {
      continue
    }
    if (frameCounter - display.lastSeenFrame > FADE_MS / 16) {
      displayed.delete(id)
    }
  }
  for (const [, display] of displayed) {
    const ageFrames = frameCounter - display.lastSeenFrame
    const alpha = Math.max(0, 1 - ageFrames / (FADE_MS / 16))
    if (alpha <= 0) {
      continue
    }
    const car = display.car
    if (car) {
      const x = px(car.x1)
      const y = py(car.y1)
      const w = (car.x2 - car.x1) * contentW
      const h = (car.y2 - car.y1) * contentH
      ctx.globalAlpha = alpha
      ctx.strokeStyle = CAR_COLOR
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.globalAlpha = 1
      if (display.text) {
        labelBox(ctx, x, y, [`# ${display.text}`], CAR_COLOR, alpha)
      }
    }
    const p = display.plate
    if (p) {
      const x = px(p.x1)
      const y = py(p.y1)
      const w = (p.x2 - p.x1) * contentW
      const h = (p.y2 - p.y1) * contentH
      ctx.globalAlpha = alpha
      ctx.strokeStyle = PLATE_COLOR
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.globalAlpha = 1
      // Without a vehicle box the caption travels with the plate box
      if (!car && display.text) {
        labelBox(ctx, x, y, [`# ${display.text}`], PLATE_COLOR, alpha)
      }
    }
  }
}

/** Secondary marks: people (when enabled) and plateless vehicles */
function paintRaw(
  ctx: CanvasRenderingContext2D,
  items: RawBox[],
  px: (nx: number) => number,
  py: (ny: number) => number,
  contentW: number,
  contentH: number,
  confirmScore: number,
  showPersons: boolean,
): void {
  for (const item of items) {
    const x = px(item.x1)
    const y = py(item.y1)
    const w = (item.x2 - item.x1) * contentW
    const h = (item.y2 - item.y1) * contentH
    if (item.type === 'person' && showPersons) {
      ctx.strokeStyle = PERSON_COLOR
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      labelBox(ctx, x, y, [`person ${(item.score * 100).toFixed(0)}%`], PERSON_COLOR)
    } else if (item.type === 'car' && item.score >= confirmScore) {
      // Vehicle without a detected plate: show the box only
      ctx.strokeStyle = CAR_COLOR
      ctx.lineWidth = 1
      ctx.strokeRect(x, y, w, h)
      labelBox(ctx, x, y, [`car ${(item.score * 100).toFixed(0)}%`], CAR_COLOR)
    }
  }
}

/**
 * Phase 2 — annotated playback for video sources. `frames` are the
 * precomputed snapshots; the ones at (and just after) the current video
 * time are interpolated so the marks move with the video between analysis
 * steps. Called from a rAF loop.
 */
export function renderPlayback(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  frames: PlaybackFrame[],
  confirmScore: number,
  showPersons: boolean,
): void {
  frameCounter++
  if (!ensureSize(canvas, video)) {
    return
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (frames.length === 0) {
    return
  }

  // Last snapshot at or before the current playback time
  const t = video.currentTime
  let lo = 0
  let hi = frames.length - 1
  let idx = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].time <= t) {
      idx = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  if (idx < 0) {
    return
  }
  const current = frames[idx]
  const next = idx + 1 < frames.length ? frames[idx + 1] : null
  const gap = next ? next.time - current.time : Number.POSITIVE_INFINITY
  const alpha =
    next && gap > 0 && gap <= PLAY_INTERP_GAP
      ? Math.min(1, Math.max(0, (t - current.time) / gap))
      : null

  const { px, py, contentW, contentH } = letterbox(canvas, video)
  const liveIds = new Set<number>()

  for (const track of current.tracks) {
    // Each mark is gated by its own confidence: the vehicle box appears
    // only above the vehicle threshold; the plate box with its caption
    // only above the plate threshold (close-up plates without a vehicle
    // box still display their plate marks).
    if (track.confidence < confirmScore) {
      continue
    }
    const following = next ? next.tracks.find(item => item.id === track.id) : undefined
    if (next && !following) {
      // Lost track: the next snapshot no longer has it, so its marks end
      // at this snapshot — the box must not sit frozen at the last
      // position across the whole gap until the track ages out.
      continue
    }
    let carTarget: SmoothBox | undefined
    if (track.carBox && track.carScore >= confirmScore) {
      carTarget = { ...track.carBox }
    }
    let plateTarget: SmoothBox = { ...track.plateBox }
    if (alpha !== null && following) {
      if (carTarget && following.carBox) {
        carTarget = lerpRect(carTarget, following.carBox, alpha)
      }
      plateTarget = lerpRect(plateTarget, following.plateBox, alpha)
    }
    let display = displayed.get(track.id)
    if (!display) {
      display = { lastSeenFrame: frameCounter }
      displayed.set(track.id, display)
    }
    display.lastSeenFrame = frameCounter
    display.text = track.text
    // Playback marks must not trail the picture: the snapshots are
    // per-video-frame and the interpolation between them is the motion
    // smoothing — an extra EMA here only makes the boxes lag behind the
    // moving vehicle (read as "detection is wrong"). Set targets directly;
    // the anti-jitter glide stays in the live-camera renderer.
    if (carTarget) {
      display.car = carTarget
    }
    display.plate = plateTarget
    liveIds.add(track.id)
  }

  paintTracks(ctx, liveIds, px, py, contentW, contentH)

  // Secondary marks come from the nearest snapshot as-is (no ids to pair)
  const rawItems: RawBox[] = current.raw.map(item => ({
    type: item.type,
    score: item.score,
    x1: item.bbox.x,
    y1: item.bbox.y,
    x2: item.bbox.x + item.bbox.w,
    y2: item.bbox.y + item.bbox.h,
  }))
  paintRaw(ctx, rawItems, px, py, contentW, contentH, confirmScore, showPersons)
}

/**
 * Live camera rendering: boxes come from the latest analysis result
 * (camera streams cannot be stepped, so analysis and display run live).
 */
export function renderOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  result: PipelineResult | null,
  tracks: PlateTrack[],
  confirmScore: number,
  showPersons: boolean,
): void {
  frameCounter++
  if (!ensureSize(canvas, video)) {
    return
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!result) {
    return
  }
  const { px, py, contentW, contentH } = letterbox(canvas, video)

  // Confirmed tracks: smooth boxes toward the latest tracker state
  // (per-mark thresholds — vehicle box needs the vehicle score, plate
  // marks need the plate confidence)
  const liveIds = new Set<number>()
  for (const track of tracks) {
    if (!track.confirmed || track.confidence < confirmScore) {
      continue
    }
    liveIds.add(track.id)
    let display = displayed.get(track.id)
    if (!display) {
      display = { lastSeenFrame: frameCounter }
      displayed.set(track.id, display)
    }
    display.lastSeenFrame = frameCounter
    display.text = track.text
    if (track.carBox && track.carScore >= confirmScore) {
      display.car = smoothTo(display.car, { ...track.carBox })
    } else {
      delete display.car
    }
    display.plate = smoothTo(display.plate, { ...track.plateBox })
  }

  paintTracks(ctx, liveIds, px, py, contentW, contentH)

  const rawItems: RawBox[] = []
  for (const object of result.objects) {
    if (object.type === 'person') {
      rawItems.push({
        type: 'person',
        score: object.score,
        x1: object.bbox.x,
        y1: object.bbox.y,
        x2: object.bbox.x + object.bbox.w,
        y2: object.bbox.y + object.bbox.h,
      })
    } else if (object.type === 'car' && !object.plateBox) {
      rawItems.push({
        type: 'car',
        score: object.score,
        x1: object.bbox.x,
        y1: object.bbox.y,
        x2: object.bbox.x + object.bbox.w,
        y2: object.bbox.y + object.bbox.h,
      })
    }
  }
  paintRaw(ctx, rawItems, px, py, contentW, contentH, confirmScore, showPersons)
}

/** Intersection-over-union of two normalized x1/y1/x2/y2 boxes */
function iou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number },
): number {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

/** Tracker state line for a car's plate: accepted variant + votes, as in the log */
function trackerLine(
  trackList: PlateTrack[],
  plate: { x1: number; y1: number; x2: number; y2: number },
): string {
  let best: PlateTrack | null = null
  let bestIoU = 0
  for (const track of trackList) {
    const overlap = iou(plate, track.plateBox)
    if (overlap > bestIoU) {
      bestIoU = overlap
      best = track
    }
  }
  if (!best || bestIoU < 0.3) {
    return '→ tracking…'
  }
  return best.confirmed
    ? `→ ${best.text} ${best.votes}/${best.observations} ✓`
    : `→ ${best.text} ${best.votes}/${best.observations}`
}

/**
 * Phase 1 — visual analysis: the video is paused on the frame that was
 * just analyzed and the overlay paints its raw detections exactly as the
 * analysis log row for this frame does: every vehicle/person box with
 * its score, the raw OCR readings of both engines («native» | «fast»)
 * and the tracker's accepted variant with its votes. Nothing is gated by
 * the confirm threshold here — the log shows everything, and so does the
 * overlay (gated marks come later, in the playback phase).
 */
export function renderAnalysisFrame(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  result: PipelineResult | null,
  trackList: PlateTrack[],
  progress: { index: number; total: number; time: number },
  showPersons: boolean,
): void {
  if (!ensureSize(canvas, video)) {
    return
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Progress caption (top-left) — mirrors the "#index time" of the log rows
  const caption = `analysis #${progress.index}/${progress.total} · ${progress.time.toFixed(2)}s`
  ctx.font = `13px 'Segoe UI', sans-serif`
  const captionWidth = ctx.measureText(caption).width + 12
  ctx.fillStyle = LABEL_BG
  ctx.fillRect(8, 8, captionWidth, 22)
  ctx.fillStyle = '#ffd60a'
  ctx.fillText(caption, 14, 23)

  // Marks accompany the analysis: while the pass advances (the video is
  // already seeking to / running inference on the next frame) the previous
  // frame's detections stay painted until the fresh result replaces them.
  // They are dropped only when the shown moment is far from the analyzed
  // region (e.g. the user scrubbed the seek bar mid-pass).
  if (!result || Math.abs(video.currentTime - progress.time) > 0.15) {
    return
  }

  const { px, py, contentW, contentH } = letterbox(canvas, video)
  for (const object of result.objects) {
    const x = px(object.bbox.x)
    const y = py(object.bbox.y)
    const w = object.bbox.w * contentW
    const h = object.bbox.h * contentH
    if (object.type === 'person') {
      if (!showPersons) {
        continue
      }
      ctx.strokeStyle = PERSON_COLOR
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      labelBox(ctx, x, y, [`person ${(object.score * 100).toFixed(0)}%`], PERSON_COLOR)
      continue
    }
    if (object.type !== 'car') {
      continue
    }
    // Vehicle box + score (all cars — the log does not filter either)
    ctx.strokeStyle = CAR_COLOR
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, w, h)
    const labels = [`car ${(object.score * 100).toFixed(0)}%`]
    if (object.plateBox) {
      const p = object.plateBox
      const plate = {
        x1: p.x,
        y1: p.y,
        x2: p.x + p.w,
        y2: p.y + p.h,
      }
      ctx.strokeStyle = PLATE_COLOR
      ctx.lineWidth = 2
      ctx.strokeRect(px(plate.x1), py(plate.y1), (plate.x2 - plate.x1) * contentW, (plate.y2 - plate.y1) * contentH)
      // Raw readings of both engines, then the tracker's accepted variant
      const native = object.plate?.raw ?? '—'
      const fast = object.plateFast?.raw ?? '—'
      labels.push(`«${native}» | «${fast}»`)
      labels.push(trackerLine(trackList, plate))
      if (object.plateScore !== undefined) {
        labels.push(`plate ${(object.plateScore * 100).toFixed(0)}%`)
      }
    }
    labelBox(ctx, x, y, labels, CAR_COLOR)
  }
}

/** Empty overlay — no marks before the analysis pass exists */
export function clearOverlay(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d')
  ctx?.clearRect(0, 0, canvas.width, canvas.height)
}

/** Clear all smoothing state (called when the analysis restarts) */
export function resetOverlay(): void {
  displayed.clear()
}
