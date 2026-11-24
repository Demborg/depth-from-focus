// depth-from-focus capture page. Everything is exposed on window.dff so it can be
// driven remotely over Chrome DevTools Protocol (tools/cdp.mjs) as well as by buttons.
import { CameraLink } from "./net/link.js";
import { pairFromUrl } from "./net/signal.js";

const out = document.getElementById("out");
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

async function start({ width = 1920, height = 1080 } = {}) {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { exact: "environment" }, width: { ideal: width }, height: { ideal: height } },
  });
  state.track = state.stream.getVideoTracks()[0];
  video.srcObject = state.stream;
  await video.play();
  await nextFrame();
  const info = caps();
  log("camera started", info.settings.width + "x" + info.settings.height);
  remoteLog({ kind: "caps", ...info });
  return info;
}

function caps() {
  const t = state.track;
  return { label: t.label, capabilities: t.getCapabilities(), settings: t.getSettings() };
}

// Lock exposure + white balance at their current auto values so brightness doesn't drift mid-sweep.
async function lockAuto() {
  const c = state.track.getCapabilities();
  const s = state.track.getSettings();
  const adv = {};
  if (c.exposureMode?.includes("manual")) {
    adv.exposureMode = "manual";
    if (s.exposureTime && c.exposureTime) adv.exposureTime = s.exposureTime;
    if (s.iso && c.iso) adv.iso = s.iso;
  }
  if (c.whiteBalanceMode?.includes("manual")) {
    adv.whiteBalanceMode = "manual";
    if (s.colorTemperature && c.colorTemperature) adv.colorTemperature = s.colorTemperature;
  }
  try {
    await state.track.applyConstraints({ advanced: [adv] });
  } catch (e) {
    log("lockAuto failed", String(e));
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
  ? { frame: (i, blob) => post(`/api/frame?stack=${id}&i=${i}`, blob, "image/jpeg"), meta: (obj) => post(`/api/meta?stack=${id}`, JSON.stringify(obj)) }
  : { frame: async () => {}, meta: async () => {} };

async function sweep({ n = 30, settle = 150, skip = 2, quality = 0.95, lo, hi, id, lock = true, spacing = "diopter", sink, onProgress } = {}) {
  if (!state.track) await start();
  const c = state.track.getCapabilities();
  if (!c.focusDistance) throw new Error("no focusDistance capability");
  lo ??= c.focusDistance.min;
  hi ??= c.focusDistance.max;
  id ??= new Date().toISOString().replace(/[:.]/g, "-");
  sink ??= serverSink(id);
  const highlight = document.getElementById("hl") ? makeHighlighter(document.getElementById("hl")) : null;
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
    onProgress?.(i + 1, n);
    log(`frame ${i + 1}/${n} req=${req.toFixed(3)} rep=${st.focusDistance}`);
  }
  await Promise.all(uploads);
  const metaObj = {
    id, width: canvas.width, height: canvas.height, params: { n, settle, skip, quality, lo, hi, spacing },
    total_ms: performance.now() - t0, locked, caps: caps(), frames, orientation: orientation.snapshot(),
  };
  await sink.meta(metaObj);
  setTimeout(() => highlight?.done(), 600);
  log("sweep done", id, `${metaObj.total_ms.toFixed(0)} ms`);
  return { id, total_ms: metaObj.total_ms };
}

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
  snapshot() { return this.q ? { quaternion: this.q, source: this.source, screen: screen.orientation?.type } : null; },
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

window.dff = { start, caps, lockAuto, setFocus, sweep, settleProbe, log, orientation };
const pairing = pairFromUrl(location.href);
if (pairing) startPaired(pairing).catch((e) => log("ERR", String(e)));
document.getElementById("start").onclick = () => start().catch((e) => log("ERR", String(e)));
document.getElementById("sweep").onclick = () => sweep().catch((e) => log("ERR", String(e)));
window.addEventListener("error", (e) => remoteLog({ kind: "error", msg: String(e.message) }));
remoteLog({ kind: "loaded", ua: navigator.userAgent });
