/**
 * Downloads the fast-plate-ocr ONNX model (cct-xs-v2-global-model) + config
 * from GitHub releases into public/models/fastplate/.
 * Run: npm run models:fastplate
 */
import { createWriteStream, existsSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = join(root, 'public', 'models', 'fastplate')

const BASE_URL = 'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates'
const FILES = [
  { url: `${BASE_URL}/cct_xs_v2_global.onnx`, name: 'model.onnx' },
  { url: `${BASE_URL}/cct_xs_v2_global_plate_config.yaml`, name: 'plate_config.yaml' },
]

mkdirSync(outDir, { recursive: true })

for (const file of FILES) {
  const outPath = join(outDir, file.name)
  if (existsSync(outPath)) {
    console.log(`already present: ${file.name}`)
    continue
  }
  console.log(`downloading ${file.url} …`)
  const res = await fetch(file.url, { redirect: 'follow' })
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}: ${file.url}`)
  }
  await pipeline(Readable.fromWeb(res.body), createWriteStream(outPath))
  console.log(`done: ${outPath}`)
}

console.log('\nfast-plate-ocr model is in place. Browser config: config.json (already in the repo).')
