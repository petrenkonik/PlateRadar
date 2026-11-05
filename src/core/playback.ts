/**
 * Precomputed annotations for the playback phase.
 *
 * The video is processed in two phases: first the whole file is stepped
 * through and analyzed (phase 1, see useLoop.analyzeVideo) — during that
 * pass no marks are drawn. Every analysis step produces one PlaybackFrame
 * snapshot keyed by the video time. When the pass is done, the video plays
 * normally (phase 2) and the overlay renders these snapshots — no
 * inference runs during playback.
 */

import type { PlateTrack } from './plate-tracker'
import type { PipelineResult, PixelBox, RecognizedBbox } from './types'

/** One confirmed track at one analysis step */
export interface PlaybackTrack {
  id: number
  /** Voted plate text */
  text: string
  /** Vehicle box, normalized 0..1 (absent for close-up plates) */
  carBox?: PixelBox
  /** Plate box, normalized 0..1 */
  plateBox: PixelBox
  /** Voted plate confidence 0..1 */
  confidence: number
  /** Vehicle detection score 0..1 */
  carScore: number
}

/** Secondary detection at one analysis step (people / plateless vehicles) */
export interface PlaybackRaw {
  type: 'person' | 'car'
  score: number
  bbox: RecognizedBbox
}

/** All marks belonging to one video moment */
export interface PlaybackFrame {
  /** video currentTime, seconds */
  time: number
  tracks: PlaybackTrack[]
  raw: PlaybackRaw[]
}

/**
 * Snapshot one analysis step for the playback phase. Confirmed tracks are
 * stored with their scores, so the display threshold is applied at draw
 * time (changing it after the analysis still works). Thumbnails and other
 * debug payloads are not stored here — the panels use the live result.
 */
export function buildPlaybackFrame(
  time: number,
  result: PipelineResult,
  trackList: PlateTrack[],
): PlaybackFrame {
  const tracks: PlaybackTrack[] = []
  for (const track of trackList) {
    if (!track.confirmed) {
      continue
    }
    tracks.push({
      id: track.id,
      text: track.text,
      carBox: track.carBox ? { ...track.carBox } : undefined,
      plateBox: { ...track.plateBox },
      confidence: track.confidence,
      carScore: track.carScore,
    })
  }
  const raw: PlaybackRaw[] = []
  for (const object of result.objects) {
    if (object.type === 'person') {
      raw.push({ type: 'person', score: object.score, bbox: { ...object.bbox } })
    } else if (object.type === 'car' && !object.plateBox) {
      raw.push({ type: 'car', score: object.score, bbox: { ...object.bbox } })
    }
  }
  return { time, tracks, raw }
}
