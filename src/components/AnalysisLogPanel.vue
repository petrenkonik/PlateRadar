<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { analysisLog, clearAnalysisLog } from '../store'

const pct = (value: number | undefined) =>
  value === undefined ? '—' : `${(value * 100).toFixed(0)}%`

const timeLabel = (entry: { time: number }) =>
  entry.time < 0 ? 'image' : `${entry.time.toFixed(2)}s`

const open = ref(false)
const listEl = ref<HTMLDivElement | null>(null)

/** Keep the newest frame in view while the log is open and filling */
watch(
  () => [analysisLog.length, open.value] as const,
  async () => {
    if (!open.value) {
      return
    }
    await nextTick()
    const el = listEl.value
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  },
)
</script>

<template>
  <div class="panel">
    <div class="row" style="justify-content: space-between; margin-bottom: 6px">
      <button :class="{ primary: open }" @click="open = !open">
        {{ open ? '▼' : '▶' }} Analysis log ({{ analysisLog.length }})
      </button>
      <div class="row">
        <span class="small muted">per-frame detections &amp; OCR readings</span>
        <button class="small" :disabled="analysisLog.length === 0" @click="clearAnalysisLog">
          Clear
        </button>
      </div>
    </div>
    <div v-if="open" ref="listEl" class="log-list">
      <div v-if="analysisLog.length === 0" class="muted small">
        Nothing yet — run the analysis and every frame lands here: detections, raw OCR
        readings of both engines, tracker votes and warnings.
      </div>
      <div v-for="entry in analysisLog" :key="entry.index" class="log-row">
        <span class="idx">#{{ entry.index }}</span>
        <span class="muted">{{ timeLabel(entry) }}</span>
        <span v-if="entry.cars.length === 0 && entry.persons === 0 && entry.readings.length === 0" class="muted">
          nothing detected
        </span>
        <span v-for="(car, i) in entry.cars" :key="'c' + i" class="det">
          car {{ pct(car.score) }}<span v-if="car.plate" class="muted">+plate</span>
        </span>
        <span v-if="entry.persons > 0" class="det person">person ×{{ entry.persons }}</span>
        <span v-for="(r, i) in entry.readings" :key="'r' + i" class="reading">
          «{{ r.native || '—' }}»<span class="muted">/</span>«{{ r.fast || '—' }}»
          → <b>{{ r.text || 'rejected' }}</b> {{ pct(r.confidence) }}
          <span v-if="r.region" class="muted">{{ r.region }}</span>
        </span>
        <span v-for="(t, i) in entry.tracks" :key="'t' + i" :class="['track', { ok: t.confirmed }]">
          {{ t.text }} {{ t.votes }}/{{ t.observations }}{{ t.confirmed ? ' ✓' : '' }}
        </span>
        <span v-for="(plate, i) in entry.confirmed" :key="'h' + i" class="chip ok">
          → history {{ plate }}
        </span>
        <span v-for="(w, i) in entry.warnings" :key="'w' + i" class="warn">{{ w }}</span>
        <span class="muted ms">{{ entry.ms.toFixed(0) }}ms</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.log-list {
  max-height: 300px;
  overflow: auto;
  border-top: 1px solid var(--border);
  padding-top: 6px;
}

.log-row {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  align-items: baseline;
  padding: 2px 0;
  border-bottom: 1px dashed var(--border);
  font-size: 12px;
}

.idx {
  color: var(--text);
  font-weight: 600;
  min-width: 34px;
}

.det {
  color: var(--blue, #4d8dff);
}

.det.person {
  color: var(--green, #34c759);
}

.reading {
  color: var(--text);
}

.reading b {
  letter-spacing: 0.06em;
}

.track {
  color: var(--amber);
}

.track.ok {
  color: var(--green, #34c759);
  font-weight: 600;
}

.warn {
  color: var(--amber);
}

.ms {
  margin-left: auto;
}
</style>
