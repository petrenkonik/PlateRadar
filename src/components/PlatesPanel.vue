<script setup lang="ts">
import { computed } from 'vue'
import { confirmedPlates, lastResult, options, smoothing, tracks } from '../store'

const pct = (value: number | undefined) =>
  value === undefined ? '—' : `${(value * 100).toFixed(0)}%`

/**
 * Everything confirmed during the pass (kept even after the live track
 * ages out), filtered by the plate confidence threshold. The vehicle
 * score gates only the vehicle box on the overlay, not the record.
 */
const confirmed = computed(() =>
  confirmedPlates.value.filter(t => t.confidence >= options.confirmScore),
)

/** Live tracks still voting (not confirmed yet) */
const pending = computed(() => tracks.value.filter(t => !t.confirmed))

/** Readings needed for confirmation under the selected algorithm */
const votesNeeded = computed(() => (smoothing.algorithm === 'ema' ? 1 : smoothing.minVotes))

/** Raw readings of the current frame (engine comparison) */
const rawPlates = computed(() =>
  (lastResult.value?.objects || []).filter(o => o.plate || o.plateFast),
)

const match = (item: (typeof rawPlates.value)[number]) =>
  item.plate && item.plateFast && item.plate.value === item.plateFast.value
</script>

<template>
  <div class="panel">
    <h3>Plates — tracking &amp; voting</h3>
    <div v-if="confirmed.length === 0" class="muted small">
      No confirmed plates yet — with the current algorithm a plate needs
      {{ votesNeeded }}+ consistent reading{{ votesNeeded > 1 ? 's' : '' }} in the last
      {{ smoothing.window }} frames (spikes that differ more than
      {{ smoothing.textSimilarity.toFixed(2) }} similarity are outvoted).
    </div>
    <div v-for="track in confirmed" :key="track.id" class="plate-card">
      <div class="plate-value">
        {{ track.text }}
        <span class="chip ok">confirmed</span>
      </div>
      <div class="small" style="margin-bottom: 6px">
        votes <b>{{ track.votes }}</b>/{{ track.observations }}
        ({{ pct(track.share) }}) · confidence <b>{{ pct(track.confidence) }}</b>
        · car {{ pct(track.carScore) }}
        <span v-if="track.region" class="muted"> · {{ track.region }}</span>
        <span v-if="track.color" class="muted"> · {{ track.color }}</span>
      </div>
      <div class="vote-bar">
        <div class="vote-fill" :style="{ width: `${Math.round(track.share * 100)}%` }"></div>
      </div>
      <table><tbody>
        <tr>
          <th>engine</th>
          <th>last raw text</th>
        </tr>
        <tr>
          <td>production (PaddleOCR)</td>
          <td><code>{{ track.nativeLast || '—' }}</code></td>
        </tr>
        <tr>
          <td>fast-plate-ocr</td>
          <td><code>{{ track.fastLast || '—' }}</code></td>
        </tr>
      </tbody></table>
    </div>

    <div v-if="pending.length > 0" class="pending small muted">
      voting: {{ pending.map(t => `${t.text} ${t.votes}/${t.observations}`).join(', ') }}
    </div>

    <h3 style="margin-top: 14px">Current frame — raw OCR comparison</h3>
    <div v-if="rawPlates.length === 0" class="muted small">No plates detected on the current frame.</div>
    <div v-for="(item, index) in rawPlates" :key="'raw' + index" class="raw-row small">
      <code>{{ item.plate?.raw || '—' }}</code> {{ pct(item.plate?.confidence) }}
      <span class="muted">|</span>
      <code>{{ item.plateFast?.raw || '—' }}</code> {{ pct(item.plateFast?.confidence) }}
      <span v-if="match(item)" class="chip ok">agree</span>
    </div>
  </div>
</template>

<style scoped>
.plate-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 8px;
  background: var(--panel-2);
}

.plate-value {
  font-family: 'Consolas', monospace;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: #fff;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.vote-bar {
  height: 4px;
  background: var(--panel);
  border-radius: 3px;
  margin: 4px 0 8px;
  overflow: hidden;
}

.vote-fill {
  height: 100%;
  background: var(--green);
}

.pending {
  margin-top: 4px;
}

.raw-row {
  margin-bottom: 6px;
}
</style>
