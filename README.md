# PlateRadar

Real-time license plate & people recognition running **entirely in the
browser**
(Vue 3 + onnxruntime-web, no backend). The same ONNX models power every
stage of the pipeline:

| Model | Role |
|---|---|
| `yolov8n.onnx` | people (COCO 0) and vehicle (car/bus/truck) detection |
| `plate.onnx` | license plate detector |
| `ocr_rec_en.onnx` + `.dict.txt` | plate text OCR (PaddleOCR, CTC + ±40° deskew) |
| `det_500m.onnx` | SCRFD — face detection |
| `w600k_mbf.onnx` | ArcFace — 512-d face embedding |
| `fastplate/model.onnx` | **fast-plate-ocr** `cct-xs-v2-global-model` — second OCR for comparison |

The recognition core (`src/core/`) is a self-contained TypeScript pipeline:
tensor utils, YOLO decoding, dual OCR, SCRFD/ArcFace, body-color k-means
and orchestration.

## Run

```bash
cd plateradar
npm install
npm run dev        # http://localhost:5178
```

Production build with type-check: `npm run build`.

## Two-phase video processing

Videos are processed in two distinct phases — playback and inference never
run at the same time:

1. **Analysis** ("▶ Analyze video") — the video is stepped through
   **frame by frame** with seeks (it stays paused, nothing plays); every
   frame runs the recognition pipeline. Every analyzed frame is shown
   **visually**: its raw detections are painted on the overlay exactly as
   the analysis log records them — vehicle/person boxes with scores, raw
   OCR readings of both engines («native» | «fast») and the tracker's
   accepted variant with its votes, plus a `analysis #i/N · time` caption.
   The panels fill with data (plate voting, objects, crops, history) while
   the pass progresses.
2. **Annotated playback** ("▶ Play with marks", starts automatically when
   the analysis finishes) — the video plays normally and the overlay
   renders the *precomputed* annotations: confirmed tracks are interpolated
   between neighboring frames, so the boxes sit exactly on the moving
   video (no extra smoothing — marks never trail the picture). Pausing
   keeps the marks on the frozen frame. **No inference runs during
   playback.**

Selecting a video leaves it **paused on the first frame** — nothing plays
until the playback phase (or the user presses play).

**Analysis log** ("▶ Analysis log" under the video) — a detailed per-frame
record of how everything was detected: vehicle/person detections with
scores, raw OCR readings of both engines with the accepted text, tracker
votes per frame, newly confirmed plates and per-stage warnings. The
overlay during the analysis shows the same information visually on each
frame (see above).

The **webcam** is the only non-seekable source, so it works live: each
camera frame is analyzed once and its marks appear as soon as that frame
is done.

## Smoothing algorithms (Recognition panel)

Raw per-frame OCR readings are noisy; instead of a sampling step, you pick
the **algorithm and its parameters** (applied when the next analysis
starts):

- **Multi-frame voting** (default) — readings are associated into tracks
  and vote over a sliding window; a track is shown only when the winning
  variant gathers enough votes. Parameters: **Window** (frames),
  **Votes to confirm**, **Consistency share** (votes/window),
  **Spike filter** (text similarity — readings that differ more start a
  separate variant and get outvoted).
- **Instant + box smoothing** — every reading is displayed right away
  (no vote gate); noise is handled by box EMA and the confidence
  threshold. Good for a quick look at what the OCR reads per frame.

Common parameters: **Box smoothing (EMA)** — the exponential factor the
track boxes follow the detections with (higher = snappier, lower =
smoother), the spike filter shared by both algorithms, and **Lost track:
drop after N frames** — a track without detections for this many analysis
frames is dropped, and its marks cancel at the last seen position instead
of being carried through the rest of the pass.

Per-mark display rule: the **vehicle box** is drawn only when the vehicle
score ≥ the confirm threshold, the **plate marks** only when the plate
confidence ≥ the confirm threshold (a confident close-up plate shows its
plate box even without a vehicle box).

## Frame sources

1. **Server video** — put mp4/webm files into `public/videos/` and run
   `npm run videos:index` (generates `index.json` for the select). The files
   stream over HTTP (range requests), so nothing has to be uploaded.
2. **Local file** — the "File from disk…" button.
3. **Webcam** — the "Webcam" button (asks for browser permission) — live mode.
4. **Single image test** — the "Image test" tab: your own image or the
   "Sample: car with plate" button (`public/test-plate.jpg`).

YOLO threshold, confirm threshold and stage toggles (people/vehicles/plates/
faces/color/fast-plate) are in the "Recognition" panel. The confirm
threshold is applied when the annotated snapshots are drawn, so changing it
after the analysis re-filters the marks without a re-run.

## License

[MIT](LICENSE)

## Bundled demo videos (public/videos/)

- `royal-mail-van.webm` — main demo, 51 s 1080p: both engines read the UK
  plates `LM07RUV` and `WV13HVX`, plus a cyclist/people for person detection
- `honda-civic.webm` — 6 s 1080p: a driving Honda Civic with the UK plate
  `V598 FHO` readable around 1.7–3.7 s (portrait picture pillarboxed in a
  16:9 frame — also exercises the overlay geometry)
- `sinaloa-plates.webm` — 1 s close-up clip: both engines read `VNF967D`
  (a quick end-to-end check: ~30 frames, a couple of minutes on CPU)

## Plate tracking & display policy (core/plate-tracker.ts)

Defaults detect **vehicles + plates** (people/faces/color are off; enable
them in "Recognition").

- **Association + voting**: raw per-frame OCR readings go into a tracker.
  Readings are associated into tracks (text similarity + box overlap, or
  motion continuity; a strong text match merges even without box overlap).
  In the default voting mode each track keeps a sliding window and votes
  for the most frequent text variant — one-frame false readings are
  outvoted and never shown or recorded (all window/vote/spike parameters
  are configurable — see "Smoothing algorithms").
- **Confirm threshold** (default 70%): each mark is gated by its own
  confidence — the vehicle box only above the vehicle score, the plate
  marks only above the plate confidence. Raw vehicles below the threshold
  get no box.
- **Tracking / interpolation**: a track stays alive a small keep-alive
  window of analysis frames without updates (configurable "Lost track"
  parameter), its box is smoothed with an EMA (configurable factor) and
  persists across frames where the detector fails; after the window the
  track is dropped and its marks cancel at the last seen position. During
  playback the boxes are additionally interpolated between neighboring
  snapshots (≤ 1.2 s apart), so the marks glide with the video — a track
  missing from the next snapshot ends exactly at its last snapshot.
- **Duplicate detections merged**: when the detector fires twice on one
  object (a smaller box mostly inside the bigger one, containment ≥ 0.7,
  same class), the boxes are merged into one before tracking — the
  tracker never sees two objects for one car.
- History records only confirmed plates (once per track) with
  `votes/observations` evidence; CSV export includes them.

## fast-plate-ocr

The model is downloaded once from GitHub releases
(`ankandrew/cnn-ocr-lp`, `cct_xs_v2_global.onnx`):

```bash
npm run models:fastplate
```

(already done — files are in `public/models/fastplate/`). In the "Plates"
panel the production OCR and fast-plate-ocr results are shown side by side
with an "engines agree" marker when the texts match. The v2 model also
predicts the plate region/country.

## Performance

Inference runs on WebGPU (when available) with a WASM fallback — the
indicator is visible in the console during model loading. Every video
frame is analyzed, so on CPU (WASM) at 0.5–2 s per frame a pass over the
51 s royal-mail clip (≈1530 frames at 30 fps) takes a while — use short
clips for quick checks (the 1 s sinaloa clip is ≈30 frames). Because the
playback phase does no inference at all, the annotated video then plays at
full frame rate. Faces/color can be turned off to speed the pass up.

## Structure

```
src/core/           recognition engine (tensor.util, yolo, ocr, faces, color, pipeline,
                    playback — precomputed annotations, plate-tracker — voting)
src/composables/    useSource (video/file/camera), useLoop (analyze → playback, live camera)
src/components/     panels: models, settings, plates, objects, history+CSV, crops
src/overlay.ts      playback rendering (snapshot interpolation) + live camera rendering
scripts/            list-videos.mjs, download-fastplate.mjs
```

## Test assets (Wikimedia Commons)

- `public/videos/royal-mail-van.webm` — "Royal Mail van - Near collision with a cyclist"
- `public/videos/honda-civic.webm` — "1999-2000 Honda Civic LS Coupe, 4May2017"
- `public/videos/sinaloa-plates.webm` — "Coche con placas de Sinaloa"
- `public/test-plate.jpg` — "ZAZ-1103 Slavuta rear view and license plate"
  (used to verify the OCR comparison: production `AE6133CT 0.99`,
  fast-plate-ocr `AE6133CT 1.00`, region Ukraine)
