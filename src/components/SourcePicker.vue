<script setup lang="ts">
import { ref } from 'vue'
import type { SourceKind } from '../composables/useSource'

defineProps<{
  serverVideos: { file: string; title: string }[]
  selectedVideo: string | null
  kind: SourceKind | null
  cameraActive: boolean
}>()

const emit = defineEmits<{
  'select-server': [file: string]
  'select-file': [file: File]
  'start-camera': []
  stop: []
}>()

const fileInput = ref<HTMLInputElement | null>(null)

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    emit('select-file', file)
  }
  input.value = ''
}
</script>

<template>
  <div class="panel">
    <h3>Frame source</h3>
    <div class="row wrap">
      <select
        :value="selectedVideo || ''"
        :disabled="serverVideos.length === 0"
        @change="emit('select-server', ($event.target as HTMLSelectElement).value)"
      >
        <option value="" disabled>Server video…</option>
        <option v-for="video in serverVideos" :key="video.file" :value="video.file">
          {{ video.title }}
        </option>
      </select>
      <button @click="fileInput?.click()">File from disk…</button>
      <button :class="{ danger: cameraActive }" @click="cameraActive ? emit('stop') : emit('start-camera')">
        {{ cameraActive ? 'Stop camera' : 'Webcam' }}
      </button>
      <input ref="fileInput" type="file" accept="video/*" style="display: none" @change="onFileChange" />
    </div>
    <div class="small muted" style="margin-top: 6px">
      Server video: put mp4/webm files into <code>vision-demo/public/videos/</code> and run
      <code>npm run videos:index</code>
    </div>
  </div>
</template>

<style scoped>
.wrap {
  flex-wrap: wrap;
}
</style>
