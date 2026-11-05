<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { MODEL_FILES, modelsStatus, options, pipeline } from '../store'

const loading = ref(false)
const list = computed(() => modelsStatus.value)
const readyCount = computed(() => list.value.filter(m => m.status === 'ready').length)

const NICE_NAMES: Record<string, string> = {
  [MODEL_FILES.yolo]: 'YOLO v8n — people/vehicles',
  [MODEL_FILES.plate]: 'Plate — plate detector',
  [MODEL_FILES.ocr]: 'Production OCR — PaddleOCR plate text',
  [MODEL_FILES.faceDet]: 'SCRFD — faces',
  [MODEL_FILES.faceEmb]: 'ArcFace — face embedding',
  [MODEL_FILES.fastPlate]: 'fast-plate-ocr — cct-xs-v2',
}

const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(1)

async function loadAll() {
  loading.value = true
  try {
    await pipeline.preloadModels({ ...options })
  } finally {
    loading.value = false
  }
}

onMounted(loadAll)
</script>

<template>
  <div class="panel">
    <h3>Models (loaded into the browser)</h3>
    <div v-for="model in list" :key="model.name" class="model-row">
      <div class="row model-head">
        <span class="model-name">{{ NICE_NAMES[model.name] || model.name }}</span>
        <span class="chip" :class="{ ok: model.status === 'ready', warn: model.status === 'loading', err: model.status === 'error' }">
          {{ model.status === 'ready' ? 'ready' : model.status === 'loading' ? 'loading…' : model.status === 'error' ? 'error' : 'not loaded' }}
        </span>
        <span v-if="model.totalBytes" class="small muted">
          {{ mb(model.loadedBytes) }} / {{ mb(model.totalBytes) }} MB
        </span>
      </div>
      <div v-if="model.status === 'loading' || (model.progress > 0 && model.progress < 1)" class="bar">
        <div class="bar-fill" :style="{ width: `${Math.round(model.progress * 100)}%` }"></div>
      </div>
      <div v-if="model.error" class="small err-msg">{{ model.error }}</div>
    </div>
    <div class="row" style="margin-top: 8px">
      <button class="primary" :disabled="loading" @click="loadAll">
        {{ loading ? 'Loading…' : 'Load missing' }}
      </button>
      <span class="small muted">ready {{ readyCount }} / {{ list.length }}</span>
    </div>
  </div>
</template>

<style scoped>
.model-row {
  margin-bottom: 8px;
}

.model-head {
  justify-content: space-between;
}

.model-name {
  font-size: 13px;
}

.bar {
  height: 4px;
  background: var(--panel-2);
  border-radius: 3px;
  margin-top: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.15s;
}

.err-msg {
  color: var(--red);
  margin-top: 2px;
}
</style>
