/**
 * Generates public/videos/index.json — the server video list for the
 * "Server video" select.
 * Run: npm run videos:index
 */
import { readdirSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const videosDir = join(root, 'public', 'videos')
const manifestPath = join(videosDir, 'index.json')

const VIDEO_EXT = new Set(['.mp4', '.webm', '.mov', '.m4v', '.mkv'])

const files = readdirSync(videosDir)
  .filter(name => VIDEO_EXT.has(name.toLowerCase().slice(name.lastIndexOf('.'))))
  .sort()

writeFileSync(
  manifestPath,
  JSON.stringify(
    { generated: new Date().toISOString(), videos: files.map(file => ({ file, title: file })) },
    null,
    2,
  ),
)

console.log(`index.json: ${files.length} video(s) -> ${manifestPath}`)
for (const file of files) {
  console.log(`  - ${file}`)
}
