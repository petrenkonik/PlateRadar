<script setup lang="ts">
import { computed } from 'vue'
import type { SourceKind } from '../composables/useSource'
import {
  analysisFrames,
  analysisProgress,
  lastResult,
  loopState,
  options,
  phase,
  smoothing,
} from '../store'

defineProps<{ kind: SourceKind | null; running: boolean }>()

const emit = defineEmits<{
  analyze: []
  cancel: []
  play: []
  'stop-play': []
  'start-live': []
  'stop-live': []
}>()

const timings = computed(() => lastResult.value?.timings)

const pct = (value: number | undefined) =>
  value === undefined ? '—' : `${(value * 100).toFixed(0)}%`

const progressPct = computed(() =>
  analysisProgress.total > 0
    ? Math.round((analysisProgress.index / analysisProgress.total) * 100)
    : 0,
)
</script>

<template>
  <div class="panel">
    <h3>Recognition</h3>

    <!-- Video / file: two phases — analyze first, then playback with marks -->
    <template v-if="kind !== 'camera'">
      <div class="row" style="margin-bottom: 10px">
        <button
          v-if="phase === 'idle'"
          class="primary"
          :disabled="!kind"
          @click="emit('analyze')"
        >
          ▶ Analyze video
        </button>
        <button v-else-if="phase === 'analyzing'" class="danger" @click="emit('cancel')">
          ■ Stop analysis
        </button>
        <button v-else-if="phase === 'ready'" class="primary" @click="emit('play')">
          ▶ Play with marks
        </button>
        <button v-else-if="phase === 'playing'" class="danger" @click="emit('stop-play')">
          ■ Stop playback
        </button>
        <button v-if="phase === 'ready' || phase === 'playing'" @click="emit('analyze')">
          ↻ Re-analyze
        </button>
        <span class="chip" :class="{ ok: phase === 'playing', warn: phase === 'analyzing' }">
          {{ phase }}
        </span>
        <span v-if="loopState.busy" class="chip warn">analyzing frame…</span>
        <span class="chip">frames: {{ loopState.frames }}</span>
      </div>

      <div v-if="phase === 'analyzing'" class="progress">
        <div class="bar" :style="{ width: `${progressPct}%` }"></div>
      </div>
      <div v-if="phase === 'analyzing'" class="small muted" style="margin-bottom: 8px">
        {{ analysisProgress.time.toFixed(1) }}s / {{ analysisProgress.duration.toFixed(1) }}s —
        frame {{ analysisProgress.index }}/{{ analysisProgress.total }}
      </div>
      <div v-if="phase === 'ready' || phase === 'playing'" class="small muted" style="margin-bottom: 8px">
        Analysis done: {{ analysisFrames.length }} annotated frames. Playback renders the
        collected marks — no inference while the video plays.
      </div>
      <div v-if="phase === 'idle'" class="small muted" style="margin-bottom: 8px">
        Two phases: 1) every video frame is analyzed (raw detections are
        shown as the analysis log fills), 2) the video plays with the
        collected marks.
      </div>

      <label class="ctl">
        <span>Smoothing algorithm</span>
        <select v-model="smoothing.algorithm" class="algo">
          <option value="vote">Multi-frame voting (spikes outvoted)</option>
          <option value="ema">Instant + box smoothing (no voting)</option>
        </select>
      </label>

      <div class="params">
        <label class="ctl">
          <span>Window: <b>{{ smoothing.window }} frames</b></span>
          <input v-model.number="smoothing.window" type="range" min="3" max="20" step="1" />
        </label>
        <label v-if="smoothing.algorithm === 'vote'" class="ctl">
          <span>Votes to confirm: <b>{{ smoothing.minVotes }}</b></span>
          <input v-model.number="smoothing.minVotes" type="range" min="1" max="6" step="1" />
        </label>
        <label v-if="smoothing.algorithm === 'vote'" class="ctl">
          <span>Consistency share: <b>{{ smoothing.minShare.toFixed(2) }}</b></span>
          <input v-model.number="smoothing.minShare" type="range" min="0.2" max="0.9" step="0.05" />
        </label>
        <label class="ctl">
          <span>
            Spike filter (text similarity):
            <b>{{ smoothing.textSimilarity.toFixed(2) }}</b>
          </span>
          <input v-model.number="smoothing.textSimilarity" type="range" min="0.5" max="1" step="0.05" />
        </label>
        <label class="ctl">
          <span>Box smoothing (EMA): <b>{{ smoothing.boxEma.toFixed(2) }}</b></span>
          <input v-model.number="smoothing.boxEma" type="range" min="0.1" max="1" step="0.05" />
        </label>
        <label class="ctl">
          <span>Lost track: drop after <b>{{ smoothing.keepAlive }}</b> frame{{ smoothing.keepAlive > 1 ? 's' : '' }}</span>
          <input v-model.number="smoothing.keepAlive" type="range" min="1" max="12" step="1" />
        </label>
        <div class="small muted">Parameters apply when the next analysis starts.</div>
      </div>
    </template>

    <!-- Webcam: live mode (a camera stream cannot be stepped) -->
    <template v-else>
      <div class="row" style="margin-bottom: 10px">
        <button v-if="!running" class="primary" @click="emit('start-live')">▶ Start live</button>
        <button v-else class="danger" @click="emit('stop-live')">■ Stop</button>
        <span class="chip" :class="{ ok: running }">{{ running ? 'live' : 'stopped' }}</span>
        <span v-if="loopState.busy" class="chip warn">analyzing frame…</span>
        <span class="chip">frames: {{ loopState.frames }}</span>
      </div>
      <div class="small muted" style="margin-bottom: 8px">
        The camera stream cannot be stepped — analysis runs live, and a frame's marks
        appear as soon as that frame is analyzed.
      </div>
    </template>

    <label class="ctl">
      <span>YOLO threshold: <b>{{ pct(options.yoloScore) }}</b></span>
      <input v-model.number="options.yoloScore" type="range" min="0.05" max="0.95" step="0.05" />
    </label>

    <label class="ctl">
      <span>Confirm threshold (vehicle + plate): <b>{{ pct(options.confirmScore) }}</b></span>
      <input v-model.number="options.confirmScore" type="range" min="0.4" max="0.95" step="0.05" />
    </label>

    <div class="checks">
      <label><input v-model="options.detectPersons" type="checkbox" /> people</label>
      <label><input v-model="options.detectCars" type="checkbox" /> vehicles</label>
      <label><input v-model="options.detectPlates" type="checkbox" /> plates</label>
      <label><input v-model="options.detectFaces" type="checkbox" /> faces (SCRFD+ArcFace)</label>
      <label><input v-model="options.carColor" type="checkbox" /> body color</label>
      <label><input v-model="options.fastPlate" type="checkbox" /> fast-plate-ocr</label>
    </div>

    <div v-if="timings" class="timings small muted">
      <span>YOLO {{ timings.detectMs.toFixed(0) }}ms</span>
      <span>plates {{ timings.plateDetectMs.toFixed(0) }}ms</span>
      <span>OCR prod {{ timings.ocrNativeMs.toFixed(0) }}ms</span>
      <span>OCR fp {{ timings.ocrFastMs.toFixed(0) }}ms</span>
      <span>faces {{ timings.facesMs.toFixed(0) }}ms</span>
      <span>color {{ timings.colorMs.toFixed(0) }}ms</span>
      <span class="total">frame {{ timings.totalMs.toFixed(0) }}ms</span>
    </div>
  </div>
</template>

<style scoped>
.ctl {
  display: block;
  margin-bottom: 8px;
}

.ctl select.algo {
  width: 100%;
  margin-top: 2px;
}

.params {
  border-top: 1px solid var(--border);
  padding-top: 8px;
  margin-bottom: 8px;
}

.checks {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 12px;
  margin-top: 4px;
  font-size: 13px;
}

.progress {
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress .bar {
  height: 100%;
  background: #4d8dff;
  transition: width 0.2s;
}

.timings {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 8px;
}

.timings .total {
  color: var(--text);
  font-weight: 600;
}
</style>
