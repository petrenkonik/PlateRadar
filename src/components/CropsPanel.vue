<script setup lang="ts">
import { computed } from 'vue'
import { lastResult } from '../store'

/** Crops of the last frame: vehicles and plates (detection stage debug) */
const crops = computed(() => {
  const result = lastResult.value
  if (!result) {
    return []
  }
  const list: { label: string; url: string }[] = []
  for (const object of result.objects) {
    if (object.crops?.plate) {
      list.push({ label: object.plate?.value || object.plateFast?.value || 'plate', url: object.crops.plate })
    }
  }
  return list
})
</script>

<template>
  <div class="panel">
    <h3>Plate crops (OCR debug)</h3>
    <div v-if="crops.length === 0" class="muted small">No crops — no plates detected on the frame.</div>
    <div v-else class="grid">
      <figure v-for="(crop, index) in crops" :key="index">
        <img :src="crop.url" alt="plate crop" />
        <figcaption class="small muted">{{ crop.label }}</figcaption>
      </figure>
    </div>
  </div>
</template>

<style scoped>
.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

figure {
  margin: 0;
}

img {
  max-width: 160px;
  max-height: 60px;
  border-radius: 6px;
  border: 1px solid var(--border);
  display: block;
}
</style>
