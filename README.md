# Depth from focus

Sweep a phone camera's focus from a web page, find where each pixel is sharpest, and turn that
into a 3D point cloud on the desktop. Built for Chrome on a Pixel 9.

- **`web/index.html`** is the desktop viewer. It shows a QR code; scan it with the phone to open
  `web/capture.html`, and the two pair over WebRTC (pairing code from
  [stopmotion](https://github.com/Demborg/stopmotion)). Press *Sweep focus*: points stream in
  while the lens moves, then a refined cloud replaces them, levelled with the phone's
  orientation sensor.
- **`web/capture.html`** on the phone highlights what is in focus during a sweep.
- **`web/dff-gpu.js`** runs depth from focus on the GPU (WebGPU compute, in a worker): breathing
  alignment, sum-modified-Laplacian focus measure, per-pixel curve normalisation, guided-filter
  aggregation of the focus volume, centroid peak. Each frame is processed as it arrives, so the
  cloud is ready a fraction of a second after the sweep, at the full 1080x1920 by default or
  540x960 with `?halvings=1` (on either page). Needs WebGPU (Chrome on Android, desktop Chrome).
- **`crates/dff-core`** is the same pipeline in dependency-free Rust, the reference the GPU
  version is checked against. `crates/dff-cli` runs it on stored captures.

## Develop

```sh
python3 server.py            # serves web/ on :8000 and stores captures in data/stacks/
adb reverse tcp:8000 tcp:8000   # phone reaches the dev server as localhost (camera needs a secure origin)
node tools/cdp.mjs open      # open the capture page on the phone; `eval` runs JS there
```

`index.html?stack=<id>&realtime` replays a capture from `data/stacks/` at its original pacing.
`cargo run --release -p dff-cli -- depth data/stacks/<id> out/<id>` runs the native pipeline;
`tools/compare.py` scores configurations against a flat cutting mat.

## Notes

- `focusDistance` on the Pixel 9 (Chrome 158) is in metres.
- The lens trails focus requests by about 271 ms during a fast sweep; frames are relabelled.
- Focal length calibrated from the mat's 1 cm grid: about 57° across the long side of the
  1080x1920 stream.
