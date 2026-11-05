<script setup lang="ts">
import { history } from '../store'

const pct = (value: number | undefined) =>
  value === undefined ? '—' : `${(value * 100).toFixed(0)}%`

const time = (date: Date) =>
  date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

function exportCsv() {
  const header = 'time;plate;confidence;votes;observations;prod_ocr_last;fast_plate_last;region;color;car_score'
  const rows = history.map(record =>
    [
      record.time.toISOString(),
      record.plate,
      record.confidence.toFixed(3),
      record.votes,
      record.observations,
      record.native || '',
      record.fast || '',
      record.region || '',
      record.color || '',
      record.score.toFixed(3),
    ].join(';'),
  )
  const csv = '\uFEFF' + [header, ...rows].join('\r\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `vision-plates-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`
  link.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="panel">
    <div class="row" style="justify-content: space-between; margin-bottom: 8px">
      <h3 style="margin: 0">Plate history (confirmed)</h3>
      <div class="row">
        <button class="small" :disabled="history.length === 0" @click="exportCsv">CSV</button>
        <button class="small" :disabled="history.length === 0" @click="history.length = 0">Clear</button>
      </div>
    </div>
    <div v-if="history.length === 0" class="muted small">
      Nothing confirmed yet — only plates voted through the frame window get here.
    </div>
    <div v-else class="scroll">
      <table><tbody>
        <tr>
          <th>time</th>
          <th>plate</th>
          <th>conf</th>
          <th>votes</th>
          <th>region</th>
          <th>color</th>
          <th></th>
        </tr>
        <tr v-for="record in history" :key="record.id">
          <td class="muted">{{ time(record.time) }}</td>
          <td><code>{{ record.plate }}</code></td>
          <td>{{ pct(record.confidence) }} <span class="muted small">car {{ pct(record.score) }}</span></td>
          <td class="muted">{{ record.votes }}/{{ record.observations }}</td>
          <td class="muted small">{{ record.region || '' }}</td>
          <td class="muted small">{{ record.color || '' }}</td>
          <td>
            <img v-if="record.thumb" :src="record.thumb" alt="crop" class="thumb" />
          </td>
        </tr>
      </tbody></table>
    </div>
  </div>
</template>

<style scoped>
.scroll {
  max-height: 260px;
  overflow: auto;
}

.thumb {
  max-height: 28px;
  border-radius: 4px;
  display: block;
}
</style>
