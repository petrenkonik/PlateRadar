<script setup lang="ts">
import { computed } from 'vue'
import { lastResult } from '../store'

const objects = computed(() => lastResult.value?.objects || [])
const persons = computed(() => objects.value.filter(o => o.type === 'person'))
const cars = computed(() => objects.value.filter(o => o.type === 'car'))

const pct = (value: number) => `${(value * 100).toFixed(0)}%`
</script>

<template>
  <div class="panel">
    <h3>Frame objects</h3>
    <div class="row" style="margin-bottom: 8px">
      <span class="chip">👤 people: {{ persons.length }}</span>
      <span class="chip">🚗 vehicles: {{ cars.length }}</span>
    </div>
    <div v-if="objects.length === 0" class="muted small">No objects found.</div>
    <table v-else><tbody>
      <tr>
        <th>type</th>
        <th>conf.</th>
        <th>details</th>
      </tr>
      <tr v-for="(object, index) in objects" :key="index">
        <td>{{ object.type === 'person' ? 'person' : 'vehicle' }}</td>
        <td>{{ pct(object.score) }}</td>
        <td class="small muted">
          <template v-if="object.type === 'car'">
            {{ object.color || '' }}
            {{ object.plate ? `#${object.plate.value}` : '' }}
            {{ object.plateFast && !object.plate ? `#${object.plateFast.value} (fp)` : '' }}
          </template>
          <template v-else-if="object.face">
            face {{ pct(object.face.detScore) }}, embedding {{ object.face.dims }}d
          </template>
        </td>
      </tr>
    </tbody></table>
  </div>
</template>
