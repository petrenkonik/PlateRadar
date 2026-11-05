/**
 * Shared demo state (plain reactive, no Pinia — a single page is enough).
 */

import { reactive, ref, shallowRef } from 'vue'
import {
  MODEL_FILES,
  modelRegistry,
  type ModelInfo,
} from './core/model-registry'
import type { PlaybackFrame } from './core/playback'
import { PlateTracker, type PlateTrack } from './core/plate-tracker'
import {
  defaultPipelineOptions,
  RecognitionPipeline,
} from './core/pipeline'
import type { PipelineOptions, PipelineResult } from './core/types'

export interface HistoryRecord {
  id: number
  time: Date
  /** Voted plate text (confirmed by the tracker) */
  plate: string
  /** Mean confidence of the winning variant */
  confidence: number
  /** Votes of the winning variant / observations in the window */
  votes: number
  observations: number
  /** Last-frame production OCR reading (comparison) */
  native?: string
  /** Last-frame fast-plate-ocr reading (comparison) */
  fast?: string
  region?: string
  color?: string
  score: number
  thumb?: string
}

export const pipeline = new RecognitionPipeline()

export const options = reactive<PipelineOptions>(defaultPipelineOptions())

/**
 * One entry of the detailed per-frame analysis log ("how everything was
 * detected"): detections, raw OCR readings, tracker state and warnings.
 */
export interface AnalysisLogEntry {
  /** Analysis step number (0-based) */
  index: number
  /** Video time, seconds (-1 = single image) */
  time: number
  /** Vehicle detections: score + whether a plate box was found on it */
  cars: { score: number; plate: boolean }[]
  /** Person detection count */
  persons: number
  /** Raw plate readings of the frame (both engines) */
  readings: {
    native?: string
    fast?: string
    /** Text accepted into the tracker (cleaned) */
    text: string
    confidence: number
    region?: string
    plateScore?: number
  }[]
  /** Tracker state after this frame (per track) */
  tracks: { text: string; votes: number; observations: number; confirmed: boolean }[]
  /** Newly confirmed plates on this frame */
  confirmed: string[]
  /** Per-stage warnings */
  warnings: string[]
  /** Pipeline time for this frame, ms */
  ms: number
}

/** Detailed analysis log (capped — the oldest entries drop off) */
export const analysisLog = reactive<AnalysisLogEntry[]>([])
const ANALYSIS_LOG_MAX = 4000

export function pushAnalysisLog(entry: AnalysisLogEntry) {
  analysisLog.push(entry)
  if (analysisLog.length > ANALYSIS_LOG_MAX) {
    analysisLog.splice(0, analysisLog.length - ANALYSIS_LOG_MAX)
  }
}

export function clearAnalysisLog() {
  analysisLog.length = 0
}

/**
 * Video processing concept: phase 1 — the whole video is stepped through
 * and analyzed frame by frame; every analyzed frame is shown with its raw
 * detections painted on the overlay (the visual counterpart of the
 * analysis log). Phase 2 — the video plays with the collected (confirmed,
 * smoothed) marks, no inference runs. Only the webcam (a stream cannot be
 * seeked) works live: analyzed frame → shown marks.
 */
export type DemoPhase = 'idle' | 'analyzing' | 'ready' | 'playing'
export const phase = ref<DemoPhase>('idle')

/** Precomputed annotated frames for the playback phase */
export const analysisFrames = shallowRef<PlaybackFrame[]>([])

/** Progress of the analysis pass (panel + progress bar) */
export const analysisProgress = reactive({
  index: 0,
  total: 0,
  /** currentTime of the last analyzed step, seconds */
  time: 0,
  duration: 0,
})

/**
 * Smoothing algorithm applied to the raw per-frame readings. Every video
 * frame is analyzed; these parameters shape the results. Applied to the
 * tracker when an analysis pass starts (see applySmoothingToTracker).
 */
export type SmoothingAlgorithm = 'vote' | 'ema'
export const smoothing = reactive({
  /** vote = multi-frame voting (spikes outvoted), ema = instant display */
  algorithm: 'vote' as SmoothingAlgorithm,
  /** Sliding window per track, analysis frames */
  window: 10,
  /** vote mode: consistent votes needed to confirm a reading */
  minVotes: 3,
  /** vote mode: minimal consistency share (votes / window) */
  minShare: 0.5,
  /**
   * Spike filter: readings whose text differs more than this from the
   * track's current reading start a separate variant (and get outvoted).
   */
  textSimilarity: 0.75,
  /** Box exponential smoothing factor (0.1..1, higher = snappier) */
  boxEma: 0.45,
  /**
   * Keep-alive: the track is dropped after this many analysis frames
   * without detections — its marks cancel at the last seen position.
   */
  keepAlive: 3,
})

/** Push the selected algorithm + parameters into the tracker */
export function applySmoothingToTracker() {
  plateTracker.configure({
    window: smoothing.window,
    // EMA mode shows every reading right away (no vote gate)
    minVotes: smoothing.algorithm === 'ema' ? 1 : smoothing.minVotes,
    minShare: smoothing.algorithm === 'ema' ? 0 : smoothing.minShare,
    textSimilarity: smoothing.textSimilarity,
    boxEma: smoothing.boxEma,
    maxGapFrames: smoothing.keepAlive,
  })
}

/** Live camera mode state (the only non-seekable source) */
export const loopState = reactive({
  running: false,
  busy: false,
  frames: 0,
})

export const lastResult = shallowRef<PipelineResult | null>(null)

/** Confirmed plate tracks (multi-frame voting, see core/plate-tracker) */
export const tracks = ref<PlateTrack[]>([])

/**
 * Every plate confirmed during the current pass, kept even after its live
 * track ages out (the tracker holds only live tracks — a plate that left
 * the frame 12+ frames ago is still a result of the pass). Best/latest
 * state per track id.
 */
export const confirmedPlates = ref<PlateTrack[]>([])

export function upsertConfirmedPlate(track: PlateTrack) {
  const list = confirmedPlates.value
  const i = list.findIndex(item => item.id === track.id)
  if (i >= 0) {
    list[i] = track
  } else {
    list.push(track)
  }
}

export const plateTracker = new PlateTracker()

export const history = reactive<HistoryRecord[]>([])
let historySeq = 0

export function pushHistory(records: HistoryRecord[]) {
  for (const record of records) {
    history.unshift(record)
  }
  if (history.length > 100) {
    history.length = 100
  }
}

export function nextHistoryId(): number {
  return ++historySeq
}

/** Model loading statuses (reactive mirror of the ModelRegistry) */
export const modelsStatus = ref<ModelInfo[]>(modelRegistry.infoList())
modelRegistry.subscribe(infos => {
  modelsStatus.value = [...infos]
})

export { modelRegistry, MODEL_FILES }

/** Browser-console debug handle (demo/testing aid, see README) */
if (typeof window !== 'undefined') {
  ;(window as unknown as Record<string, unknown>).__vision = {
    phase,
    analysisFrames,
    analysisProgress,
    analysisLog,
    lastResult,
    smoothing,
    tracks,
    confirmedPlates,
    history,
    options,
    loopState,
    plateTracker,
  }
}
