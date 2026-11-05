import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    port: 5178,
  },
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      output: {
        // ONNX Runtime Web тащит свой wasm-загрузчик — держим его отдельным чанком
        manualChunks: {
          onnxruntime: ['onnxruntime-web'],
        },
      },
    },
  },
})
