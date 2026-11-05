/**
 * Frame sources: server videos (public/videos + index.json), a local file,
 * a webcam (getUserMedia).
 */

import { onBeforeUnmount, ref, type Ref } from 'vue'

export type SourceKind = 'server' | 'file' | 'camera'

interface VideoManifest {
  videos: { file: string; title?: string }[]
}

export function useSource(videoEl: Ref<HTMLVideoElement | null>) {
  const kind = ref<SourceKind | null>(null)
  const serverVideos = ref<{ file: string; title: string }[]>([])
  const selectedVideo = ref<string | null>(null)
  const fileUrl = ref<string | null>(null)
  const cameraActive = ref(false)
  const error = ref<string | null>(null)

  let cameraStream: MediaStream | null = null

  async function loadServerList(): Promise<void> {
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}videos/index.json`, { cache: 'no-store' })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const manifest = (await res.json()) as VideoManifest
      serverVideos.value = (manifest.videos || []).map(v => ({
        file: v.file,
        title: v.title || v.file,
      }))
    } catch (e) {
      error.value = `Server video list is unavailable (${e instanceof Error ? e.message : e}). Put files into public/videos and run npm run videos:index`
    }
  }

  function applySource(src: string | null, stream: MediaStream | null) {
    const el = videoEl.value
    if (!el) {
      return
    }
    if (cameraStream && cameraStream !== stream) {
      cameraStream.getTracks().forEach(t => t.stop())
    }
    cameraStream = stream
    if (stream) {
      el.srcObject = stream
      el.src = ''
      // A camera stream shows nothing until it is played
      el.muted = true
      el.play().catch(() => {})
    } else {
      el.srcObject = null
      el.src = src || ''
      el.muted = true
      // Stay paused on the first frame: the video must not play before
      // (or during) the analysis — playback starts only as phase 2
      el.pause()
    }
  }

  function selectServer(file: string) {
    kind.value = 'server'
    selectedVideo.value = file
    fileUrl.value = null
    error.value = null
    applySource(`${import.meta.env.BASE_URL}videos/${file}`, null)
  }

  function selectFile(file: File) {
    kind.value = 'file'
    selectedVideo.value = null
    if (fileUrl.value) {
      URL.revokeObjectURL(fileUrl.value)
    }
    fileUrl.value = URL.createObjectURL(file)
    error.value = null
    applySource(fileUrl.value, null)
  }

  async function startCamera() {
    error.value = null
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      kind.value = 'camera'
      cameraActive.value = true
      selectedVideo.value = null
      applySource(null, stream)
    } catch (e) {
      error.value = `Webcam is unavailable: ${e instanceof Error ? e.message : e}`
      cameraActive.value = false
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop())
      cameraStream = null
    }
    cameraActive.value = false
  }

  function stop() {
    stopCamera()
    const el = videoEl.value
    if (el) {
      el.pause()
      el.removeAttribute('src')
      el.srcObject = null
      el.load()
    }
    kind.value = null
    selectedVideo.value = null
  }

  onBeforeUnmount(() => {
    stopCamera()
    if (fileUrl.value) {
      URL.revokeObjectURL(fileUrl.value)
    }
  })

  return {
    kind,
    serverVideos,
    selectedVideo,
    fileUrl,
    cameraActive,
    error,
    loadServerList,
    selectServer,
    selectFile,
    startCamera,
    stop,
  }
}
