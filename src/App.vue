<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AnalysisLogPanel from './components/AnalysisLogPanel.vue'
import CropsPanel from './components/CropsPanel.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import ImageTestTab from './components/ImageTestTab.vue'
import ModelLoader from './components/ModelLoader.vue'
import ObjectsPanel from './components/ObjectsPanel.vue'
import PlatesPanel from './components/PlatesPanel.vue'
import SettingsPanel from './components/SettingsPanel.vue'
import SourcePicker from './components/SourcePicker.vue'
import { useLoop } from './composables/useLoop'
import { useSource } from './composables/useSource'
import {
  clearOverlay,
  renderAnalysisFrame,
  renderOverlay,
  renderPlayback,
  resetOverlay,
} from './overlay'
import {
  analysisFrames,
  analysisProgress,
  lastResult,
  loopState,
  options,
  phase,
  plateTracker,
  tracks,
} from './store'

const tab = ref<'video' | 'image'>('video')

const videoEl = ref<HTMLVideoElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)

const source = useSource(videoEl)
const loop = useLoop(videoEl)

/**
 * The overlay is painted on every animation frame, depending on the phase:
 *  - analyzing (video source): visual analysis — the video is paused on
 *    the frame that was just analyzed and its raw detections (vehicle /
 *    person boxes, raw OCR readings, tracker votes) are painted exactly
 *    as the analysis log records them;
 *  - playing / ready (video source): annotated playback — marks come from
 *    the precomputed analysis (phase 1), nothing is inferred. "Ready"
 *    keeps the marks on a paused frame, so inspecting a frozen moment
 *    still shows its annotations;
 *  - camera + live: marks appear as soon as a camera frame is analyzed;
 *  - otherwise (idle): no marks — the overlay stays empty until the
 *    analysis pass exists.
 */
let rafId = 0
function tick() {
  const canvas = canvasEl.value
  const video = videoEl.value
  if (canvas && video && tab.value === 'video') {
    const videoSource = source.kind.value !== 'camera'
    if (videoSource && (phase.value === 'playing' || phase.value === 'ready')) {
      renderPlayback(
        canvas,
        video,
        analysisFrames.value,
        options.confirmScore,
        options.detectPersons,
      )
    } else if (videoSource && phase.value === 'analyzing') {
      renderAnalysisFrame(
        canvas,
        video,
        lastResult.value,
        tracks.value,
        analysisProgress,
        options.detectPersons,
      )
    } else if (!videoSource && loopState.running) {
      renderOverlay(
        canvas,
        video,
        lastResult.value,
        tracks.value,
        options.confirmScore,
        options.detectPersons,
      )
    } else {
      clearOverlay(canvas)
    }
  }
  rafId = requestAnimationFrame(tick)
}

watch(
  () => loopState.running,
  running => {
    if (running) {
      resetOverlay()
    }
  },
)
watch(tab, () => setTimeout(() => { resetOverlay(); syncOverlayRect() }, 50))

/**
 * The canvas must cover exactly the *content* rectangle of the video.
 * The video element can be wider/taller than its picture (width 100% +
 * max-height clamp), and the browser shows the content letter/pillarboxed
 * inside it (object-fit: contain). Stretching the canvas over the whole
 * element would distort the marks horizontally. So the canvas rect is
 * recomputed with the same contain math the browser applies.
 */
function syncOverlayRect() {
  const video = videoEl.value
  const canvas = canvasEl.value
  if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
    return
  }
  const elW = video.clientWidth
  const elH = video.clientHeight
  if (!elW || !elH) {
    return
  }
  const scale = Math.min(elW / video.videoWidth, elH / video.videoHeight)
  const w = Math.round(video.videoWidth * scale)
  const h = Math.round(video.videoHeight * scale)
  canvas.style.left = `${Math.round((elW - w) / 2)}px`
  canvas.style.top = `${Math.round((elH - h) / 2)}px`
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
}

let overlayObserver: ResizeObserver | null = null

/** Manual play/pause through the video controls maps to the playback phase */
function onVideoPlay() {
  if (source.kind.value !== 'camera' && phase.value === 'ready') {
    phase.value = 'playing'
  }
}
function onVideoPause() {
  if (phase.value === 'playing') {
    phase.value = 'ready'
  }
}

/** Source switch: analysis, tracks and votes of the previous video must not leak into the new one */
function onSourceChange(apply: (arg?: unknown) => void) {
  loop.abort()
  plateTracker.clear()
  tracks.value = []
  lastResult.value = null
  resetOverlay()
  apply()
}

onMounted(() => {
  source.loadServerList()
  overlayObserver = new ResizeObserver(syncOverlayRect)
  if (videoEl.value) {
    overlayObserver.observe(videoEl.value)
  }
  syncOverlayRect()
  rafId = requestAnimationFrame(tick)
})
onBeforeUnmount(() => {
  cancelAnimationFrame(rafId)
  overlayObserver?.disconnect()
})
</script>

<template>
  <header class="app-header">
    <div>
      <h1>Vision Demo</h1>
      <div class="small muted">
        License plate &amp; people recognition — ONNX models running entirely in the browser, no backend
      </div>
    </div>
    <div class="row">
      <span class="chip"> yolov8n · plate · PaddleOCR · fast-plate-ocr · SCRFD · ArcFace </span>
    </div>
  </header>

  <main class="layout">
    <section class="left">
      <div class="tabs row">
        <button :class="{ primary: tab === 'video' }" @click="tab = 'video'">Video / camera</button>
        <button :class="{ primary: tab === 'image' }" @click="tab = 'image'">Image test</button>
      </div>

      <div v-show="tab === 'video'">
        <SourcePicker
          :server-videos="source.serverVideos.value"
          :selected-video="source.selectedVideo.value"
          :kind="source.kind.value"
          :camera-active="source.cameraActive.value"
          @select-server="onSourceChange(() => source.selectServer($event))"
          @select-file="onSourceChange(file => source.selectFile(file as File))"
          @start-camera="onSourceChange(() => source.startCamera())"
          @stop="source.stop"
        />

        <div class="stage panel">
          <div class="video-wrap">
            <video
              ref="videoEl"
              controls
              muted
              loop
              playsinline
              @loadedmetadata="syncOverlayRect"
              @resize="syncOverlayRect"
              @play="onVideoPlay"
              @pause="onVideoPause"
            ></video>
            <canvas ref="canvasEl" class="overlay"></canvas>
            <div v-if="loopState.busy" class="busy">analyzing frame…</div>
          </div>
          <div v-if="source.error.value" class="warn-line">{{ source.error.value }}</div>
          <div
            v-for="(warning, index) in lastResult?.warnings || []"
            :key="index"
            class="warn-line"
          >
            {{ warning }}
          </div>
        </div>

        <AnalysisLogPanel />
      </div>

      <div v-show="tab === 'image'">
        <ImageTestTab @error="message => (source.error.value = message)" />
      </div>
    </section>

    <aside class="right">
      <ModelLoader />
      <SettingsPanel
        :kind="source.kind.value"
        :running="loopState.running"
        @analyze="loop.analyzeVideo()"
        @cancel="loop.cancelAnalysis()"
        @play="loop.startPlayback()"
        @stop-play="loop.stopPlayback()"
        @start-live="loop.startLive()"
        @stop-live="loop.stopLive()"
      />
      <PlatesPanel />
      <ObjectsPanel />
      <CropsPanel />
      <HistoryPanel />
    </aside>
  </main>
</template>

<style scoped>
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}

.app-header h1 {
  margin: 0;
  font-size: 18px;
}

.layout {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 16px;
  padding: 16px 20px;
  align-items: start;
}

@media (max-width: 1100px) {
  .layout {
    grid-template-columns: 1fr;
  }
}

.tabs {
  margin-bottom: 12px;
}

.video-wrap {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  min-height: 240px;
}

.video-wrap video {
  display: block;
  width: 100%;
  max-height: 62vh;
  /* the browser pillars/letterboxes the picture inside the element —
     syncOverlayRect() mirrors this exact geometry for the overlay */
  object-fit: contain;
}

.overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.busy {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(255, 184, 77, 0.9);
  color: #000;
  font-size: 12px;
  font-weight: 600;
  border-radius: 6px;
  padding: 2px 8px;
}

.warn-line {
  margin-top: 8px;
  color: var(--amber);
  font-size: 12px;
}
</style>
