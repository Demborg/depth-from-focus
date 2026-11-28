// depth-from-focus capture page. Everything is exposed on window.dff so it can be
// driven remotely over Chrome DevTools Protocol (tools/cdp.mjs) as well as by buttons.
import { CameraLink } from "./net/link.js";
import { pairFromUrl } from "./net/signal.js";
import { laggedDiopters } from "./lag.js";
import { createPhone3D } from "./phone3d.js";

const out = document.getElementById("out");
if (new URLSearchParams(location.search).has("debug")) out.hidden = false;
const status = (t) => { document.getElementById("status").textContent = t; };
const video = document.getElementById("v");
const state = { stream: null, track: null };

function log(...args) {
  const msg = args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
  out.textContent = (msg + "\n" + out.textContent).slice(0, 8000);
  console.log("[dff]", msg);
}

async function post(path, body, type = "application/json") {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": type }, body });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
}

const remoteLog = (obj) => (["localhost", "127.0.0.1"].includes(location.hostname) ? post("/api/log", JSON.stringify(obj)).catch(() => {}) : undefined);

function nextFrame() {
  return new Promise((res) => video.requestVideoFrameCallback((now, meta) => res(meta)));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The highlight canvas follows the stream's aspect (it turns with the screen).
function fitStage() {
  const s = state.track?.getSettings();
  if (!s?.width) return;
  const hl = document.getElementById("hl"), long = 480, short = 270;
  [hl.width, hl.height] = s.width > s.height ? [long, short] : [short, long];
}
screen.orientation?.addEventListener("change", () => setTimeout(fitStage, 300));


async function start({ width = 1920, height = 1080 } = {}) {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { exact: "environment" }, width: { ideal: width }, height: { ideal: height } },
  });
  state.track = state.stream.getVideoTracks()[0];
  state.track.addEventListener("ended", () => { syncSweepBtn(); });
  video.srcObject = state.stream;
  await video.play();
  await nextFrame();
  const info = caps();
  log("camera started", info.settings.width + "x" + info.settings.height);
  fitStage();
  syncSweepBtn();
  status("ready");
  remoteLog({ kind: "caps", ...info });
  return info;
}

function caps() {
  const t = state.track;
  return { label: t.label, capabilities: t.getCapabilities(), settings: t.getSettings() };
}

// Lock exposure + white balance at their current auto values so brightness doesn't drift mid-sweep.
// Exposure and white balance are applied in separate constraint calls because Android Chrome
// reports colorTemperature = 0 in auto mode, which can reject a bundled manual-mode constraint.
async function lockAuto() {
  const c = state.track.getCapabilities();
  const s = state.track.getSettings();
  const clampStep = (v, r) => {
    const cl = Math.min(r.max, Math.max(r.min, v));
    return r.step ? Math.round((cl - r.min) / r.step) * r.step + r.min : cl;
  };
  if (c.exposureMode?.includes("manual")) {
    const exp = { exposureMode: "manual" };
    if (s.exposureTime && c.exposureTime) exp.exposureTime = clampStep(s.exposureTime, c.exposureTime);
    if (s.iso && c.iso) exp.iso = clampStep(s.iso, c.iso);
    try { await state.track.applyConstraints({ advanced: [exp] }); } catch (e) { log("lockAuto exp failed", String(e)); }
  }
  if (c.whiteBalanceMode?.includes("manual") && c.colorTemperature && s.colorTemperature >= c.colorTemperature.min && s.colorTemperature <= c.colorTemperature.max) {
    try {
      await state.track.applyConstraints({ advanced: [{ whiteBalanceMode: "manual", colorTemperature: clampStep(s.colorTemperature, c.colorTemperature) }] });
    } catch (e) {
      log("lockAuto wb failed", String(e));
    }
  }
  return state.track.getSettings();
}

async function setFocus(d) {
  await state.track.applyConstraints({ advanced: [{ focusMode: "manual", focusDistance: d }] });
}

async function grab(canvas, quality) {
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.convertToBlob({ type: "image/jpeg", quality });
}

// Discrete sweep: for each focus distance, set, wait `settle` ms + `skip` fresh frames, grab.
// In-focus highlight during a sweep: a cheap focus measure (modified Laplacian, 5x5 box) on a
// 270x480 copy of each grabbed frame, compared with each pixel's best so far. Pixels at their
// sharpest glow green; pixels whose peak has clearly passed keep a faint teal tint.
function makeHighlighter(canvasEl) {
  const w = canvasEl.width, h = canvasEl.height, n = w * h;
  const src = new OffscreenCanvas(w, h);
  const sctx = src.getContext("2d", { willReadFrequently: true });
  const octx = canvasEl.getContext("2d");
  const out = octx.createImageData(w, h);
  const gray = new Float32Array(n), ml = new Float32Array(n), tmp = new Float32Array(n), m = new Float32Array(n);
  const best = new Float32Array(n);
  let frames = 0;
  octx.clearRect(0, 0, w, h);
  canvasEl.style.opacity = 1;
  const boxH = (a, b, r) => { for (let y = 0; y < h; y++) { let acc = 0; const o = y * w; for (let x = -r; x <= r; x++) acc += a[o + Math.min(w - 1, Math.max(0, x))]; for (let x = 0; x < w; x++) { b[o + x] = acc; acc += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)]; } } };
  const boxV = (a, b, r) => { for (let x = 0; x < w; x++) { let acc = 0; for (let y = -r; y <= r; y++) acc += a[Math.min(h - 1, Math.max(0, y)) * w + x]; for (let y = 0; y < h; y++) { b[y * w + x] = acc; acc += a[Math.min(h - 1, y + r + 1) * w + x] - a[Math.max(0, y - r) * w + x]; } } };
  return {
    frame(image) {
      sctx.drawImage(image, 0, 0, w, h);
      const d = sctx.getImageData(0, 0, w, h).data;
      for (let p = 0, i = 0; p < n; p++, i += 4) gray[p] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        const p = y * w + x, c = 2 * gray[p];
        ml[p] = Math.abs(c - gray[p - 1] - gray[p + 1]) + Math.abs(c - gray[p - w] - gray[p + w]);
      }
      boxH(ml, tmp, 2); boxV(tmp, m, 2);
      let mean = 0;
      for (let p = 0; p < n; p++) mean += m[p];
      mean /= n;
      const tex = 1.5 * mean; // ignore flat regions (sensor noise only)
      frames++;
      const o = out.data;
      for (let p = 0, i = 0; p < n; p++, i += 4) {
        const v = m[p];
        if (v > best[p]) best[p] = v;
        let a = 0, g = 0;
        if (frames > 1 && best[p] > tex) {
          if (v >= 0.88 * best[p]) { a = 150; g = 1; }      // at (or within 12% of) its sharpest
          else if (v < 0.6 * best[p]) { a = 45; g = 0; }    // peak has passed: captured
        }
        o[i] = g ? 60 : 40; o[i + 1] = g ? 255 : 190; o[i + 2] = g ? 120 : 200; o[i + 3] = a;
      }
      octx.putImageData(out, 0, 0);
    },
    done() { canvasEl.style.opacity = 0; },
  };
}

// Where sweep frames go: the dev server (python3 server.py, reached over adb reverse on
// localhost), a paired viewer over WebRTC, or nowhere (standalone demo: just the highlight).
const devServer = ["localhost", "127.0.0.1"].includes(location.hostname);
const serverSink = (id) => devServer
  ? {
      frame: (i, blob) => post(`/api/frame?stack=${id}&i=${i}`, blob, "image/jpeg"),
      meta: (obj) => post(`/api/meta?stack=${id}`, JSON.stringify(obj)),
    }
  : { frame: async () => {}, meta: async () => {} };

async function sweep({ n = 30, settle = 150, skip = 2, quality = 0.95, lo, hi, id, lock = true, spacing = "diopter", sink, onProgress } = {}) {
  if (!state.track) await start();
  const c = state.track.getCapabilities();
  if (!c.focusDistance) throw new Error("no focusDistance capability");
  lo ??= c.focusDistance.min;
  hi ??= c.focusDistance.max;
  id ??= new Date().toISOString().replace(/[:.]/g, "-");
  sink ??= serverSink(id);
  view3d.hide();
  const highlight = document.getElementById("hl") ? makeHighlighter(document.getElementById("hl")) : null;
  const local = localDepth.begin(n);
  const locked = lock ? await lockAuto() : null;
  const s0 = state.track.getSettings();
  const canvas = new OffscreenCanvas(s0.width, s0.height);
  const frames = [];
  const uploads = [];
  const t0 = performance.now();
  // Park at the start and let it settle properly before the first frame.
  await setFocus(lo);
  await sleep(500);
  for (let i = 0; i < n; i++) {
    const u = i / Math.max(1, n - 1);
    // diopter spacing: uniform in 1/d, which is roughly uniform in defocus blur.
    const req = spacing === "diopter" ? 1 / (1 / lo + (1 / hi - 1 / lo) * u) : lo + (hi - lo) * u;
    const ta = performance.now();
    await setFocus(req);
    const tb = performance.now();
    await sleep(settle);
    let meta;
    for (let k = 0; k <= skip; k++) meta = await nextFrame();
    const blob = await grab(canvas, quality);
    highlight?.frame(canvas);
    const bitmapP = createImageBitmap(canvas);
    const st = state.track.getSettings();
    frames.push({
      file: `${String(i).padStart(3, "0")}.jpg`,
      focus_requested: req,
      focus_reported: st.focusDistance,
      t_req_ms: ta - t0,
      orientation: i === 0 ? orientation.snapshot() : undefined,
      t_apply_ms: tb - ta,
      t_ms: performance.now() - t0,
      media_time: meta?.mediaTime,
    });
    uploads.push(sink.frame(i, blob, frames[i]));
    local.frame(i, await bitmapP, frames[i]);
    onProgress?.(i + 1, n);
    status(`sweeping ${i + 1}/${n}`);
    log(`frame ${i + 1}/${n} req=${req.toFixed(3)} rep=${st.focusDistance}`);
  }
  await Promise.all(uploads);
  const metaObj = {
    id, width: canvas.width, height: canvas.height, params: { n, settle, skip, quality, lo, hi, spacing },
    total_ms: performance.now() - t0, locked, caps: caps(), frames, orientation: orientation.snapshot(),
  };
  await sink.meta(metaObj);
  highlight?.done();
  log("sweep done", id, `${metaObj.total_ms.toFixed(0)} ms`);
  status("computing depth…");
  local.finish(metaObj).then((r) => showResult(r, metaObj)).catch((e) => { log("depth failed", String(e)); status(`depth failed: ${e.message}`); });
  return { id, total_ms: metaObj.total_ms };
}

// On-device depth: the same WebGPU worker as the desktop viewer, fed each sweep frame as it is
// grabbed, so the per-frame work overlaps the sweep and the phone can show the result in 3D.
// Pipeline overrides from the URL, as in the viewer: e.g. ?halvings=1 for half resolution.
function urlOpts() {
  const q = new URLSearchParams(location.search), opts = {};
  for (const k of ["win", "halvings", "peak", "agg", "eps", "norm", "sharp"])
    if (q.has(k)) opts[{ peak: "peakR", agg: "aggR", eps: "aggEps" }[k] ?? k] = +q.get(k);
  return opts;
}

const localDepth = (() => {
  let worker = null, onMsg = () => {}, seq = 0;
  const get = () => {
    if (!worker) {
      worker = new Worker(new URL("./dff-worker.js", import.meta.url), { type: "module" });
      worker.onmessage = ({ data }) => onMsg(data);
      worker.onerror = (e) => log("worker error:", e.message);
    }
    return worker;
  };
  return {
    begin(frames = 40) {
      const w = get();
      let pushMs = 0;
      const mySeq = ++seq;
      w.postMessage({ type: "begin", opts: { frames, ...urlOpts() }, lag: true });
      const result = new Promise((res, rej) => {
        onMsg = (m) => {
          if (m.type === "pushed") pushMs += m.ms;
          if (m.type === "result" && m.seq === mySeq) res({ ...m, pushMs });
          if (m.type === "error") rej(new Error(m.message));
        };
      });
      return {
        frame(i, bitmap, meta) { w.postMessage({ type: "frame", i, bitmap, meta }, [bitmap]); },
        // Lens-lag relabelling only for continuous (settle = 0) sweeps, as in the viewer.
        async finish(meta) {
          const lag = meta.params.settle === 0 && meta.frames.every((f) => f.t_req_ms != null);
          const t = performance.now();
          w.postMessage({ type: "finish", diopters: lag ? laggedDiopters(meta.frames) : meta.frames.map((f) => 1 / f.focus_requested) });
          const r = await result;
          log(`depth ${r.result.w}x${r.result.h}: ${(r.pushMs / 1000).toFixed(1)} s per-frame work, +${((performance.now() - t) / 1000).toFixed(2)} s after the sweep`);
          return r;
        },
      };
    },
  };
})();

const stage = document.getElementById("stage");
const view3d = createPhone3D(stage, { get q() { return orientation.q; } }, { labelsEl: document.getElementById("labels3d") });
let lastPointsText = "";
function updateHint3d() {
  const h = document.getElementById("hint3d");
  if (!view3d.visible) { h.hidden = true; return; }
  h.hidden = false;
  if (view3d.measuring) h.textContent = view3d.top ? "tap 2 points · drag handles · pan/pinch" : "tap 2 points · drag handles · orbit/pinch";
  else if (view3d.top) h.textContent = "drag to pan · pinch to zoom";
  else if (view3d.tilt) h.textContent = "tilt to look around · touch to orbit";
  else h.textContent = "drag to orbit · pinch to zoom · double-tap to reset";
}
function showResult({ result, colours }, meta) {
  const capOri = meta?.frames?.[0]?.orientation ?? meta?.orientation ?? null;
  const { points } = view3d.show(result, colours, { screenAngle: capOri?.angle ?? 0, orientation: capOri });
  stage.classList.add("in3d");
  document.getElementById("legend").hidden = true;
  document.getElementById("bar3d").hidden = false;
  document.getElementById("bar").hidden = true;
  lastPointsText = `${Math.round(points / 1000)}k points`;
  status(lastPointsText);
  updateHint3d();
  log(`3D view: ${points.toLocaleString()} points`);
}
function showCamera() {
  setTimeout(fitStage);
  status("ready");
  view3d.hide();
  stage.classList.remove("in3d");
  document.getElementById("legend").hidden = false;
  document.getElementById("bar3d").hidden = true;
  document.getElementById("bar").hidden = false;
  document.getElementById("hint3d").hidden = true;
  document.getElementById("b-mclear").hidden = true;
}
const setPressed = (id, on) => {
  const b = document.getElementById(id);
  b.setAttribute("aria-pressed", on);
  b.classList.toggle("outline", !on);
};
view3d.onTilt = () => updateHint3d();
view3d.onTop = (on) => { setPressed("b-top", on); updateHint3d(); };
view3d.onMeasure = ({ on, count, pending, summary }) => {
  setPressed("b-measure", on);
  document.getElementById("b-mclear").hidden = count === 0 && !pending;
  if (summary) {
    status(summary.levelled ? `${summary.total} (↔${summary.horiz} ↕${summary.vert})` : summary.total);
  } else if (pending) {
    status("tap 2nd point");
  } else if (view3d.visible) {
    status(lastPointsText);
  }
  updateHint3d();
};
document.getElementById("b-camera").onclick = showCamera;

// Dev: run a stored stack (dev server /stacks/<id>) through the on-device depth and 3D view.
async function replay(id) {
  const meta = await fetch(`stacks/${id}/meta.json`).then((r) => r.json());
  const local = localDepth.begin(meta.frames.length);
  for (const [i, f] of meta.frames.entries()) {
    const bitmap = await createImageBitmap(await fetch(`stacks/${id}/${f.file}`).then((r) => r.blob()));
    local.frame(i, bitmap, f);
  }
  showResult(await local.finish(meta), meta);
}
document.getElementById("b-top").onclick = () => view3d.setTop(!view3d.top);
document.getElementById("b-measure").onclick = () => view3d.setMeasure(!view3d.measuring);
document.getElementById("b-mclear").onclick = () => view3d.clearMeasure();

// Probe lens settling: jump between two focus values and record per-frame sharpness of a
// center crop, so we can see how many frames/ms the lens needs to settle.
async function settleProbe({ a, b, frames = 20, crop = 256 } = {}) {
  if (!state.track) await start();
  const c = state.track.getCapabilities().focusDistance;
  a ??= c.min;
  b ??= c.max;
  const s = state.track.getSettings();
  const cv = new OffscreenCanvas(crop, crop);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  const sharp = () => {
    ctx.drawImage(video, (s.width - crop) / 2, (s.height - crop) / 2, crop, crop, 0, 0, crop, crop);
    const d = ctx.getImageData(0, 0, crop, crop).data;
    let acc = 0;
    for (let y = 1; y < crop - 1; y++)
      for (let x = 1; x < crop - 1; x++) {
        const i = (y * crop + x) * 4;
        const lap = 4 * d[i + 1] - d[i + 1 - 4] - d[i + 1 + 4] - d[i + 1 - crop * 4] - d[i + 1 + crop * 4];
        acc += lap * lap;
      }
    return acc / (crop * crop);
  };
  const result = {};
  for (const [from, to] of [[a, b], [b, a]]) {
    await setFocus(from);
    await sleep(800);
    const t0 = performance.now();
    await setFocus(to);
    const series = [];
    for (let k = 0; k < frames; k++) {
      const m = await nextFrame();
      series.push({ t: performance.now() - t0, media_time: m.mediaTime, sharp: sharp(), rep: state.track.getSettings().focusDistance });
    }
    result[`${from}->${to}`] = series;
  }
  remoteLog({ kind: "settle", result });
  return result;
}

// Phone orientation in world space. AbsoluteOrientationSensor gives a quaternion [x, y, z, w]
// rotating device coordinates (x right, y up, z out of the screen) into Earth coordinates
// (x east, y north, z up). Falls back to the deviceorientationabsolute event (Euler angles).
const orientation = {
  q: null, source: null, samples: 0,
  start() {
    try {
      const s = new AbsoluteOrientationSensor({ frequency: 30, referenceFrame: "device" });
      s.addEventListener("reading", () => { this.q = [...s.quaternion]; this.source = "AbsoluteOrientationSensor"; this.samples++; });
      s.addEventListener("error", (e) => { log("orientation sensor:", e.error?.name); this.fallback(); });
      s.start();
    } catch (e) {
      this.fallback();
    }
  },
  fallback() {
    addEventListener("deviceorientationabsolute", (e) => {
      if (e.alpha == null) return;
      this.q = eulerToQuat(e.alpha, e.beta, e.gamma); this.source = "deviceorientationabsolute"; this.samples++;
    });
  },
  snapshot() { return this.q ? { quaternion: this.q, source: this.source, screen: screen.orientation?.type, angle: screen.orientation?.angle ?? 0 } : null; },
};

// W3C DeviceOrientation Z-X'-Y'' intrinsic angles (degrees) -> quaternion [x, y, z, w].
function eulerToQuat(alpha, beta, gamma) {
  const r = Math.PI / 180, [a, b, g] = [alpha * r / 2, beta * r / 2, gamma * r / 2];
  const [ca, sa, cb, sb, cg, sg] = [Math.cos(a), Math.sin(a), Math.cos(b), Math.sin(b), Math.cos(g), Math.sin(g)];
  return [
    cg * sb * ca - sg * cb * sa,
    sg * cb * ca + cg * sb * sa,
    cg * cb * sa + sg * sb * ca,
    cg * cb * ca - sg * sb * sa,
  ];
}
orientation.start();

// Paired mode: opened from the viewer's QR (#pair=...). Streams the live preview to the viewer
// and runs sweeps on its command, sending frames over the WebRTC data channel.
async function startPaired(pairing) {
  const info = await start();
  log("pairing with viewer…");
  const link = new CameraLink(pairing, state.stream);
  link.addEventListener("state", () => log("link:", link.state));
  link.addEventListener("wire", (e) => {
    const wire = e.detail;
    const send = (m) => wire.send(m);
    send({ type: "hello", caps: info.capabilities, settings: info.settings, orientation: orientation.snapshot() });
    wire.addEventListener("message", async ({ detail: m }) => {
      if (m.type === "sweep") {
        try {
          const sink = {
            frame: async (i, blob, meta) => wire.send({ type: "frame", i, meta }, blob),
            meta: async (obj) => send({ type: "meta", meta: obj }),
          };
          await sweep({ ...m.params, sink, onProgress: (k, n) => send({ type: "progress", k, n }) });
        } catch (err) {
          send({ type: "error", message: String(err) });
        }
      }
      if (m.type === "orientation") send({ type: "orientation", orientation: orientation.snapshot() });
    });
  });
  await link.start();
  window.dffLink = link;
}

window.dff = { start, caps, lockAuto, setFocus, sweep, settleProbe, log, orientation, view3d, showCamera, replay };
const sweepBtn = document.getElementById("sweep");
const cameraLive = () => state.track?.readyState === "live";
function syncSweepBtn() { sweepBtn.textContent = cameraLive() ? "Sweep" : "Start camera"; }
const pairing = pairFromUrl(location.href);
const replayId = new URLSearchParams(location.search).get("stack");
if (pairing) {
  startPaired(pairing).catch((e) => { syncSweepBtn(); log("ERR", String(e)); });
} else if (replayId) {
  replay(replayId).catch((e) => log("replay failed", String(e)));
} else {
  start().catch((e) => { syncSweepBtn(); status("tap to start camera"); log("auto-start:", String(e)); });
}
// The same fast continuous sweep the viewer asks for (lens lag is corrected afterwards).
sweepBtn.onclick = () => (cameraLive() ? sweep({ n: 60, lo: 0.12, hi: 1.6, settle: 0, skip: 0 }) : start()).catch((e) => { syncSweepBtn(); log("ERR", String(e)); });
window.addEventListener("error", (e) => remoteLog({ kind: "error", msg: String(e.message) }));
remoteLog({ kind: "loaded", ua: navigator.userAgent });
