#!/usr/bin/env node
// Score the GPU pipeline on a stored capture of the flat cutting mat (data/stacks/lag_up or similar):
// fit a plane to the points in the middle of the frame and report how far they are from it.
//   node tools/accuracy.mjs [stack] ['{"aggR": 8}' ...]     one line per option set (default: defaults)
// Needs the dev server (python3 server.py) and a headless Chrome with WebGPU on $DESK_PORT (9336), e.g.
//   google-chrome --headless=new --remote-debugging-port=9336 --user-data-dir=/tmp/chrome-gpu \
//     --enable-unsafe-webgpu --enable-features=Vulkan,WebGPUService --use-angle=vulkan --ignore-gpu-blocklist
import { execFileSync } from "node:child_process";

const [stack = "lag_up", ...sets] = process.argv.slice(2);
const desk = (...a) => execFileSync("node", [new URL("./desk.mjs", import.meta.url).pathname, ...a], {
  encoding: "utf8", env: { DESK_PORT: "9336", ...process.env },
});

const run = (opts) => `(async () => {
  const { createDff } = await import("/dff-gpu.js?" + Date.now());
  const { laggedDiopters } = await import("/lag.js");
  const { cloudMask, focalPx } = await import("/cloud.js?" + Date.now());
  const meta = await fetch("/stacks/${stack}/meta.json").then((r) => r.json());
  const lag = meta.params.settle === 0 && meta.frames.every((f) => f.t_req_ms != null);
  const labels = lag ? laggedDiopters(meta.frames) : meta.frames.map((f) => 1 / f.focus_requested);
  const s = (await createDff()).begin(${JSON.stringify(opts)});
  for (const f of meta.frames) {
    const b = await fetch("/stacks/${stack}/" + f.file).then((r) => r.blob()).then(createImageBitmap);
    await s.push(b); b.close();
  }
  const r = await s.finish(labels);
  const { w, h } = r, f = focalPx(w, h), { Z, keep } = cloudMask(r, r.diopter);
  let points = 0; for (const k of keep) points += k;
  // Middle of the frame (the mat): least-squares plane z = a x + b y + c, distances in mm.
  const P = [];
  for (let y = Math.floor(h * 0.3); y < h * 0.8; y += 2) for (let x = Math.floor(w * 0.2); x < w * 0.8; x += 2) {
    const p = y * w + x; if (!keep[p]) continue;
    const z = Z[p]; P.push([((x + 0.5 - w / 2) / f) * z, ((y + 0.5 - h / 2) / f) * z, z]);
  }
  const S = Array.from({ length: 3 }, () => [0, 0, 0]), B = [0, 0, 0];
  for (const [x, y, z] of P) { const v = [x, y, 1]; for (let i = 0; i < 3; i++) { B[i] += v[i] * z; for (let j = 0; j < 3; j++) S[i][j] += v[i] * v[j]; } }
  const det = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(S), c = [0, 1, 2].map((i) => det(S.map((row, k) => row.map((v, j) => (j === i ? B[k] : v)))) / D);
  const e = P.map(([x, y, z]) => Math.abs(z - (c[0] * x + c[1] * y + c[2])) * 1000).sort((a, b) => a - b);
  const rms = Math.sqrt(e.reduce((a, v) => a + v * v, 0) / e.length);
  return { points, medianMm: +e[e.length >> 1].toFixed(1), rmsMm: +rms.toFixed(1), p95Mm: +e[Math.floor(e.length * 0.95)].toFixed(1),
    msPerFrame: +(r.pushMs / meta.frames.length).toFixed(1), finishMs: Math.round(r.finishMs) };
})()`;

desk("open", "http://localhost:8000/index.html?stack=none");
for (const set of sets.length ? sets : ["{}"]) {
  const out = JSON.parse(desk("eval", run(JSON.parse(set))));
  console.log(set.padEnd(24), Object.entries(out).map(([k, v]) => `${k} ${v}`).join("  "));
}
