/**
 * Two-phase video processing:
 *   Phase 1 — analyzeVideo: the video is stepped through **frame by
 *   frame** with seeks (it stays paused, nothing plays during analysis);
 *   every frame runs the recognition pipeline and produces a
 *   PlaybackFrame snapshot keyed by the video time. The selected
 *   smoothing algorithm + parameters (store.smoothing) are applied to the
 *   tracker for the pass. Panels update live, and each analyzed frame is
 *   shown with its raw detections painted (overlay.renderAnalysisFrame) —
 *   exactly what the analysis log records for that frame.
 *   Phase 2 — startPlayback: the video plays normally and the overlay
 *   renders the precomputed annotations (see overlay.renderPlayback).
 *   No inference runs during playback.
 *
 * Raw plate readings go into the PlateTracker during phase 1; tracks
 * confirmed by the smoothing algorithm AND above the display threshold
 * (options.confirmScore on the plate confidence) reach the snapshots,
 * the history and the playback marks.
 *
 * The webcam stream cannot be seeked, so it keeps a live mode (startLive):
 * each camera frame is analyzed once and its marks appear right after.
 */

import { onBeforeUnmount, type Ref } from 'vue'
import { buildPlaybackFrame, type PlaybackFrame } from '../core/playback'
import { captureScaled } from '../core/tensor.util'
import type { PipelineResult, RgbImage } from '../core/types'
import type { PlateObservation, PlateTrack } from '../core/plate-tracker'
import { resetOverlay } from '../overlay'
import {
  analysisFrames,
  analysisProgress,
  applySmoothingToTracker,
  clearAnalysisLog,
  confirmedPlates,
  lastResult,
  loopState,
  nextHistoryId,
  options,
  phase,
  pipeline,
  plateTracker,
  pushAnalysisLog,
  pushHistory,
  tracks,
  upsertConfirmedPlate,
  type AnalysisLogEntry,
  type HistoryRecord,
} from '../store'

const MAX_FRAME_WIDTH = 640
/** Normalized plate text: Latin/digits only */
const cleanPlate = (text: string) => (text || '').toUpperCase().replace(/[^A-Z0-9]/g, '')

/** Live-mode timer */
let liveTimer: ReturnType<typeof setTimeout> | null = null
/**
 * Guards against superseded runs: canceled analysis, a source switch or a
 * new analysis invalidates the running pass (token mismatch → return).
 */
let runToken = 0
/** Soft cancel: stop the pass, keep the frames collected so far */
let cancelRequested = false

/** Track ids already written to history (a track is recorded once) */
const announcedIds = new Set<number>()

function grabFrame(video: HTMLVideoElement): RgbImage {
  const { img } = captureScaled(video, video.videoWidth, video.videoHeight, MAX_FRAME_WIDTH)
  return img
}

/** Best per-frame reading: production OCR wins when plate-like, fast-plate otherwise */
function pickReading(result: PipelineResult): PlateObservation[] {
  const now = performance.now()
  const observations: PlateObservation[] = []
  for (const object of result.objects) {
    if (object.type !== 'car' || !object.plateBox) {
      continue
    }
    const native = object.plate?.value
    const fast = object.plateFast?.value
    const useNative =
      object.plate && native && native.length >= 4 && /\d.*\d/.test(cleanPlate(native))
    const text = cleanPlate(useNative ? native! : fast || '')
    if (!text || text.length < 4) {
      continue
    }
    const confidence = useNative ? object.plate!.confidence : object.plateFast!.confidence
    observations.push({
      text,
      confidence,
      plateBox: {
        x1: object.plateBox.x,
        y1: object.plateBox.y,
        x2: object.plateBox.x + object.plateBox.w,
        y2: object.plateBox.y + object.plateBox.h,
      },
      carBox: {
        x1: object.bbox.x,
        y1: object.bbox.y,
        x2: object.bbox.x + object.bbox.w,
        y2: object.bbox.y + object.bbox.h,
      },
      carScore: object.score,
      region: object.plateRegion,
      color: object.color,
      native: object.plate?.value,
      fast: object.plateFast?.value,
      thumb: object.crops?.plate,
      time: now,
    })
  }
  return observations
}

/**
 * Confirmed by the smoothing algorithm AND the voted-plate confidence at
 * or above the display threshold. (The vehicle score gates only the
 * vehicle box on the overlay — see overlay.ts — not the record itself.)
 */
function displayable(track: PlateTrack, confirmScore: number): boolean {
  return track.confirmed && track.confidence >= confirmScore
}

/** Push a history record the first time a track becomes displayable */
function collectConfirmations(
  confirmedNow: PlateTrack[],
  confirmScore: number,
  announced: Set<number>,
): HistoryRecord[] {
  const records: HistoryRecord[] = []
  for (const track of confirmedNow) {
    if (!displayable(track, confirmScore) || announced.has(track.id)) {
      continue
    }
    announced.add(track.id)
    records.push({
      id: nextHistoryId(),
      time: new Date(),
      plate: track.text,
      confidence: track.confidence,
      votes: track.votes,
      observations: track.observations,
      native: track.nativeLast,
      fast: track.fastLast,
      region: track.region,
      color: track.color,
      score: track.carScore,
      thumb: track.thumb,
    })
  }
  return records
}

/**
 * Feed one pipeline result into the tracker + history + analysis log
 * (shared by all modes). `time` is the video time of the frame
 * (-1 for a single image).
 */
function processResult(result: PipelineResult, time = -1, stepIndex = 0): PlateTrack[] {
  lastResult.value = result
  loopState.frames++
  const refreshed = plateTracker.update(pickReading(result), performance.now())
  tracks.value = refreshed
  const records = collectConfirmations(
    refreshed.filter(track => track.confirmed),
    options.confirmScore,
    announcedIds,
  )
  if (records.length > 0) {
    pushHistory(records)
  }
  // Keep the pass results visible even after live tracks age out
  for (const track of refreshed) {
    if (track.confirmed) {
      upsertConfirmedPlate(track)
    }
  }
  pushAnalysisLog(buildLogEntry(result, time, stepIndex, refreshed, records))
  return refreshed
}

/** Detailed per-frame entry: how everything was detected on this frame */
function buildLogEntry(
  result: PipelineResult,
  time: number,
  stepIndex: number,
  refreshed: PlateTrack[],
  records: HistoryRecord[],
): AnalysisLogEntry {
  const cars: AnalysisLogEntry['cars'] = []
  let personCount = 0
  const readings: AnalysisLogEntry['readings'] = []
  for (const object of result.objects) {
    if (object.type === 'car') {
      cars.push({ score: object.score, plate: !!object.plateBox })
    } else if (object.type === 'person') {
      personCount++
    }
    if (object.plate || object.plateFast) {
      const native = object.plate?.value
      const fast = object.plateFast?.value
      const useNative =
        object.plate && native && native.length >= 4 && /\d.*\d/.test(cleanPlate(native))
      const text = cleanPlate(useNative ? native! : fast || '')
      readings.push({
        native: object.plate?.raw,
        fast: object.plateFast?.raw,
        text,
        confidence: useNative ? object.plate!.confidence : (object.plateFast?.confidence ?? 0),
        region: object.plateRegion,
        plateScore: object.plateScore,
      })
    }
  }
  return {
    index: stepIndex,
    time,
    cars,
    persons: personCount,
    readings,
    tracks: refreshed.slice(0, 8).map(track => ({
      text: track.text,
      votes: track.votes,
      observations: track.observations,
      confirmed: track.confirmed,
    })),
    confirmed: records.map(record => record.plate),
    warnings: result.warnings,
    ms: result.timings.totalMs,
  }
}

/**
 * Seek and wait until the new frame is decoded. The `seeked` event fires
 * once the frame is ready, so canvas drawImage can capture it right away
 * (deliberately no requestAnimationFrame here: rAF is suspended while the
 * window is occluded, which would hang the whole analysis). A timeout
 * guards sources that never fire `seeked` — e.g. seeking to the current
 * time does not produce a seek event at all.
 */
function seekAndWait(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise(resolve => {
    let done = false
    const finish = () => {
      if (done) {
        return
      }
      done = true
      video.removeEventListener('seeked', finish)
      resolve()
    }
    video.addEventListener('seeked', finish)
    try {
      if (Math.abs(video.currentTime - time) < 0.005) {
        finish() // already on this exact frame — nothing to wait for
      } else {
        video.currentTime = time
        window.setTimeout(finish, 1500)
      }
    } catch {
      finish()
    }
  })
}

/** Let the UI paint between inference runs (WASM blocks the main thread) */
const yieldToUi = () => new Promise<void>(resolve => window.setTimeout(resolve, 0))

/**
 * Best-effort fps estimate: frames decoded so far / duration. A rough
 * figure is enough — the seek loop deduplicates frames the seek snaps to
 * twice, and the tracker tolerates small gaps.
 */
function estimateFps(video: HTMLVideoElement): number {
  try {
    const withLegacy = video as unknown as { webkitDecodedFrameCount?: number }
    const quality =
      typeof video.getVideoPlaybackQuality === 'function'
        ? video.getVideoPlaybackQuality()
        : null
    const decoded = quality ? quality.totalVideoFrames : (withLegacy.webkitDecodedFrameCount ?? 0)
    if (decoded > 4 && Number.isFinite(video.duration) && video.duration > 0) {
      const fps = Math.round(decoded / video.duration)
      if (fps >= 12 && fps <= 120) {
        return fps
      }
    }
  } catch {
    // fall through to the default
  }
  return 30
}

/** Reset all per-run state (analysis and history start from zero) */
function resetRunState(video: HTMLVideoElement | null) {
  plateTracker.clear()
  announcedIds.clear()
  tracks.value = []
  confirmedPlates.value = []
  analysisFrames.value = []
  clearAnalysisLog()
  analysisProgress.index = 0
  analysisProgress.total = 0
  analysisProgress.time = 0
  analysisProgress.duration = video && Number.isFinite(video.duration) ? video.duration : 0
  loopState.frames = 0
  resetOverlay()
}

/**
 * Phase 1: step through the video **frame by frame** and analyze every
 * frame. The video does not play — it is seeked while the pipeline runs,
 * then the collected marks are played back (phase 2 starts automatically).
 */
async function analyzeVideo(video: HTMLVideoElement | null): Promise<void> {
  if (
    !video ||
    phase.value === 'analyzing' ||
    !video.videoWidth ||
    !Number.isFinite(video.duration)
  ) {
    return
  }
  const token = ++runToken
  cancelRequested = false
  await pipeline.preloadModels({ ...options })
  if (token !== runToken) {
    return
  }
  // The smoothing algorithm chosen in the panel governs this pass
  applySmoothingToTracker()
  video.pause()
  stopPlayback(video)
  resetRunState(video)
  phase.value = 'analyzing'

  const fps = estimateFps(video)
  const frameDur = 1 / fps
  const totalFrames = Math.max(1, Math.floor((video.duration - 0.02) / frameDur))
  analysisProgress.total = totalFrames

  let lastFrameTime = -1
  for (let i = 0; i < totalFrames; i++) {
    if (token !== runToken) {
      // superseded (source switch / new run) — drop everything
      return
    }
    if (cancelRequested) {
      break
    }
    const target = Math.min(i * frameDur, video.duration - 0.001)
    await seekAndWait(video, target)
    if (token !== runToken) {
      return
    }
    const actual = video.currentTime
    if (Math.abs(actual - lastFrameTime) < 5e-4) {
      // fps overestimated: the seek snapped to the same decoded frame —
      // count the step but do not waste an inference run on a duplicate
      analysisProgress.index++
      analysisProgress.time = actual
      await yieldToUi()
      continue
    }
    loopState.busy = true
    let frame: PlaybackFrame
    try {
      const img = grabFrame(video)
      const result = await pipeline.recognize(img, { ...options })
      const refreshed = processResult(result, actual, analysisProgress.index)
      frame = buildPlaybackFrame(actual, result, refreshed)
    } finally {
      loopState.busy = false
    }
    if (token !== runToken) {
      return
    }
    analysisFrames.value = [...analysisFrames.value, frame]
    lastFrameTime = actual
    analysisProgress.index++
    analysisProgress.time = actual
    await yieldToUi()
  }

  if (token !== runToken) {
    return
  }
  if (analysisFrames.value.length === 0) {
    phase.value = 'idle'
    return
  }
  phase.value = 'ready'
  // Phase 2 starts right away: the annotated video plays back
  await startPlayback(video)
}

/** Soft cancel: stop the pass, keep what has been analyzed so far */
function cancelAnalysis() {
  if (phase.value === 'analyzing') {
    cancelRequested = true
  }
}

/** Phase 2: normal playback; marks come from the precomputed snapshots */
async function startPlayback(video?: HTMLVideoElement | null) {
  if (phase.value !== 'ready' && phase.value !== 'playing') {
    return
  }
  phase.value = 'playing'
  if (video) {
    video.currentTime = 0
    await video.play().catch(() => {})
  }
}

function stopPlayback(video?: HTMLVideoElement | null) {
  if (phase.value === 'playing') {
    phase.value = 'ready'
  }
  video?.pause()
}

/** Hard reset: analysis canceled for good (source switch, tab change) */
function abort() {
  runToken++ // any in-flight analyzeVideo returns without touching state
  cancelRequested = false
  if (liveTimer) {
    clearTimeout(liveTimer)
    liveTimer = null
  }
  loopState.running = false
  loopState.busy = false
  phase.value = 'idle'
  analysisFrames.value = []
  resetOverlay()
}

// ---------------------------------------------------------------------------
// Live mode (webcam only — a camera stream cannot be stepped)
// ---------------------------------------------------------------------------

async function analyzeLiveFrame(video: HTMLVideoElement | null): Promise<void> {
  if (!video || loopState.busy || !video.videoWidth) {
    return
  }
  loopState.busy = true
  try {
    const img = grabFrame(video)
    const result = await pipeline.recognize(img, { ...options })
    processResult(result, video.currentTime, loopState.frames)
  } finally {
    loopState.busy = false
  }
}

function scheduleLive(videoRef: Ref<HTMLVideoElement | null>) {
  if (liveTimer) {
    clearTimeout(liveTimer)
  }
  // Small yield between runs: WASM inference blocks the main thread
  liveTimer = setTimeout(async () => {
    if (!loopState.running) {
      return
    }
    await analyzeLiveFrame(videoRef.value)
    if (loopState.running) {
      scheduleLive(videoRef)
    }
  }, 50)
}

async function startLive(videoRef: Ref<HTMLVideoElement | null>) {
  if (loopState.running) {
    return
  }
  const token = ++runToken
  await pipeline.preloadModels({ ...options })
  if (token !== runToken) {
    return
  }
  applySmoothingToTracker()
  resetRunState(videoRef.value)
  loopState.running = true
  // "Start" is a user gesture, so play() is allowed here even under
  // strict autoplay policies.
  videoRef.value?.play().catch(() => {})
  await analyzeLiveFrame(videoRef.value)
  scheduleLive(videoRef)
}

function stopLive() {
  loopState.running = false
  if (liveTimer) {
    clearTimeout(liveTimer)
    liveTimer = null
  }
}

/** Single run over a ready frame (image test tab) */
async function analyzeImage(img: RgbImage): Promise<PipelineResult> {
  await pipeline.preloadModels({ ...options })
  const result = await pipeline.recognize(img, { ...options })
  processResult(result)
  return result
}

export function useLoop(videoEl: Ref<HTMLVideoElement | null>) {
  onBeforeUnmount(abort)

  return {
    analyzeVideo: () => analyzeVideo(videoEl.value),
    cancelAnalysis,
    startPlayback: () => startPlayback(videoEl.value),
    stopPlayback: () => stopPlayback(videoEl.value),
    startLive: () => startLive(videoEl),
    stopLive,
    abort,
    analyzeImage,
  }
}
