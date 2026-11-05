<script setup lang="ts">
import { ref } from 'vue'
import { imageFileToRgb } from '../core/tensor.util'
import { useLoop } from '../composables/useLoop'
import { lastResult } from '../store'

const emit = defineEmits<{ error: [message: string] }>()

const videoEl = ref<HTMLVideoElement | null>(null)
const { analyzeImage } = useLoop(videoEl)
const busy = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const previewUrl = ref<string | null>(null)
const summary = ref<string | null>(null)

async function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file) {
    await analyzeFile(file)
  }
}

/** Single pipeline run over an image file */
async function analyzeFile(file: File) {
  busy.value = true
  summary.value = null
  try {
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
    }
    previewUrl.value = URL.createObjectURL(file)
    const img = await imageFileToRgb(file)
    const result = await analyzeImage(img)
    const cars = result.objects.filter(o => o.type === 'car')
    const persons = result.objects.filter(o => o.type === 'person')
    const platesCount = result.objects.filter(o => o.plate || o.plateFast).length
    summary.value = `${persons.length} people, ${cars.length} vehicles, plates: ${platesCount}, frame ${result.timings.totalMs.toFixed(0)} ms`
  } catch (e) {
    emit('error', `Analysis error: ${e instanceof Error ? e.message : e}`)
  } finally {
    busy.value = false
  }
}

/** Built-in sample: photo of a car with a Ukrainian plate (public/test-plate.jpg) */
async function runSample() {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}test-plate.jpg`)
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`)
    }
    await analyzeFile(new File([await res.blob()], 'test-plate.jpg', { type: 'image/jpeg' }))
  } catch (e) {
    emit('error', `Sample is unavailable: ${e instanceof Error ? e.message : e}`)
  }
}
</script>

<template>
  <div class="panel">
    <h3>Single image test</h3>
    <div class="row">
      <button class="primary" :disabled="busy" @click="fileInput?.click()">
        {{ busy ? 'Analyzing…' : 'Choose image' }}
      </button>
      <button :disabled="busy" @click="runSample">Sample: car with plate</button>
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        style="display: none"
        @change="onFileChange"
      />
      <span v-if="summary" class="chip ok">{{ summary }}</span>
      <span v-else-if="busy" class="chip warn">pipeline run…</span>
    </div>
    <div v-if="previewUrl" class="preview">
      <img :src="previewUrl" alt="test image" />
      <div v-if="lastResult" class="small muted" style="margin-top: 6px">
        Result in the panels on the right (plates, objects, crops). Last run:
        YOLO {{ lastResult.timings.detectMs.toFixed(0) }}ms, OCR
        {{ lastResult.timings.ocrNativeMs.toFixed(0) }}/{{ lastResult.timings.ocrFastMs.toFixed(0) }}ms
      </div>
    </div>
    <div class="small muted" style="margin-top: 6px">
      A single frame run through the whole recognition mechanism, no video needed —
      useful for quick checks with a photo.
    </div>
  </div>
</template>

<style scoped>
.preview img {
  max-width: 100%;
  max-height: 320px;
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-top: 10px;
}
</style>
