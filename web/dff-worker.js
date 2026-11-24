// Depth worker: owns the wasm session so decoding, per-frame focus measures and the volume stage
// never block the page. Frames are processed as they arrive (overlapping the sweep).
//
// in:  { type: "begin", opts, lag }                 start a capture (lag: apply lens-lag relabelling)
//      { type: "frame", i, bitmap, meta }           ImageBitmap (transferred) + per-frame metadata
//      { type: "finish", diopters }
// out: { type: "pushed", n, ms }                    per frame
//      { type: "points", w, h, px, z, rgb }         newly decided pixels while the sweep runs
//      { type: "result", result, colours }          Float32 arrays + Uint8 RGB per depth pixel
//      { type: "error", message }
import { loadDff, laggedDiopters } from "./dff-wasm.js";

const ready = loadDff();
let session = null, opts = {}, lag = false, metas = [], colours = [], queue = Promise.resolve(), cw = 540, ch = 960;

// Early points: a pixel is sent once its focus peak is >= R frames behind the sweep and its
// sharpness has dropped below DROP x peak. MIN_CONF is stricter than the final threshold, since
// these come from unaggregated curves; finish() replaces them with the refined cloud.
const EARLY = { r: 3, drop: 0.6, minConf: 2.0 };

function streamDecided() {
  const n = metas.length;
  const labels = lag && metas.every((f) => f?.t_req_ms != null) ? laggedDiopters(metas) : metas.map((f) => 1 / f.focus_requested);
  const d = session.takeDecided(EARLY.r, EARLY.drop, EARLY.minConf);
  const count = d.px.length;
  if (!count) return;
  const z = new Float32Array(count), rgb = new Uint8Array(count * 3);
  const sx = cw / d.w, cx = (cw - 1) / 2, cy = (ch - 1) / 2;
  for (let i = 0; i < count; i++) {
    const t = d.index[i], k0 = Math.min(n - 1, Math.floor(t)), k1 = Math.min(n - 1, k0 + 1), a = t - k0;
    z[i] = 1 / (labels[k0] * (1 - a) + labels[k1] * a);
    const k = Math.min(n - 1, Math.max(0, Math.round(t))), s = session.scale(k), p = d.px[i];
    const x = p % d.w, y = (p - x) / d.w;
    const ix = Math.min(cw - 1, Math.max(0, Math.round(cx + ((x + 0.5) * sx - cx) * s)));
    const iy = Math.min(ch - 1, Math.max(0, Math.round(cy + ((y + 0.5) * sx - cy) * s)));
    const j = (iy * cw + ix) * 4, f = colours[k];
    rgb[i * 3] = f[j]; rgb[i * 3 + 1] = f[j + 1]; rgb[i * 3 + 2] = f[j + 2];
  }
  postMessage({ type: "points", w: d.w, h: d.h, px: d.px, z, rgb }, [d.px.buffer, z.buffer, rgb.buffer]);
}

function rgbaOf(bitmap, w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

// Colour of each depth pixel from the frame where it is sharpest (breathing-aligned sample).
function allInFocus(r) {
  const out = new Uint8Array(r.w * r.h * 3);
  const sx = cw / r.w, cx = (cw - 1) / 2, cy = (ch - 1) / 2, n = colours.length;
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const p = y * r.w + x;
    const k = Math.min(n - 1, Math.max(0, Math.round(r.index[p])));
    const s = r.scales[k];
    const ix = Math.min(cw - 1, Math.max(0, Math.round(cx + ((x + 0.5) * sx - cx) * s)));
    const iy = Math.min(ch - 1, Math.max(0, Math.round(cy + ((y + 0.5) * sx - cy) * s)));
    const i = (iy * cw + ix) * 4, f = colours[k];
    out[p * 3] = f[i]; out[p * 3 + 1] = f[i + 1]; out[p * 3 + 2] = f[i + 2];
  }
  return out;
}

const step = (fn) => { queue = queue.then(fn).catch((e) => postMessage({ type: "error", message: String(e?.stack ?? e) })); };

onmessage = ({ data: m }) => {
  if (m.type === "begin") step(async () => { await ready; session = null; colours = []; metas = []; opts = m.opts ?? {}; lag = !!m.lag; });
  if (m.type === "frame") step(async () => {
    const dff = await ready;
    const t = performance.now();
    session ??= dff.begin(m.bitmap.width, m.bitmap.height, opts);
    const n = session.push(rgbaOf(m.bitmap, m.bitmap.width, m.bitmap.height));
    colours[m.i] = rgbaOf(m.bitmap, cw, ch);
    metas[m.i] = m.meta;
    m.bitmap.close();
    if (m.meta) streamDecided();
    postMessage({ type: "pushed", n, ms: performance.now() - t });
  });
  if (m.type === "finish") step(async () => {
    const r = session.finish(m.diopters);
    session = null;
    const c = allInFocus(r);
    colours = []; metas = [];
    const bufs = [r.diopter, r.diopterSmooth, r.conf, r.index, r.scales, r.spread, r.aif].map((a) => a.buffer);
    postMessage({ type: "result", result: r, colours: c }, [...bufs, c.buffer]);
  });
};
