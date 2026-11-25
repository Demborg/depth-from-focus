// Depth worker: owns the WebGPU session (dff-gpu.js) so frame uploads, readbacks and result
// unpacking never block the page. Frames are processed as they arrive (overlapping the sweep).
//
// in:  { type: "begin", opts, lag }                 start a capture (lag: apply lens-lag relabelling)
//      { type: "frame", i, bitmap, meta }           ImageBitmap (transferred) + per-frame metadata
//      { type: "finish", diopters }                 aggregation + peak, posts the result
// out: { type: "pushed", n, ms }                    per frame
//      { type: "points", w, h, px, z, rgb }         newly decided pixels while the sweep runs
//      { type: "result", seq, result, colours }     Float32 arrays + Uint8 RGB per depth pixel;
//                                                   seq counts begins, to drop stale results
//      { type: "error", message }
import { createDff } from "./dff-gpu.js";
import { laggedDiopters } from "./lag.js";

const ready = createDff();
let seq = 0; // capture number (counted begins), echoed on results
let session = null, opts = {}, lag = false, metas = [], queue = Promise.resolve();

// Early points: a pixel is sent once its focus peak is >= R frames behind the sweep and its
// sharpness has dropped below DROP x peak. MIN_CONF is stricter than the final threshold, since
// these come from unaggregated curves; finish() replaces them with the refined cloud.
const EARLY = { r: 3, drop: 0.6, minConf: 2.0 };

async function streamDecided() {
  const labels = lag && metas.every((f) => f?.t_req_ms != null) ? laggedDiopters(metas) : metas.map((f) => 1 / f.focus_requested);
  const d = await session.takeDecided(labels, EARLY.r, EARLY.drop, EARLY.minConf);
  if (d.px.length) postMessage({ type: "points", ...d }, [d.px.buffer, d.z.buffer, d.rgb.buffer]);
}

const step = (fn) => { queue = queue.then(fn).catch((e) => postMessage({ type: "error", message: String(e?.message ?? e) })); };

onmessage = ({ data: m }) => {
  if (m.type === "begin") step(async () => { await ready; seq++; session?.free(); session = null; metas = []; opts = m.opts ?? {}; lag = !!m.lag; });
  if (m.type === "frame") step(async () => {
    const dff = await ready;
    const t = performance.now();
    session ??= dff.begin(opts);
    const n = await session.push(m.bitmap);
    m.bitmap.close();
    metas[m.i] = m.meta;
    if (m.meta) await streamDecided();
    postMessage({ type: "pushed", n, ms: performance.now() - t });
  });
  if (m.type === "finish") step(async () => {
    await ready; // rethrows "WebGPU unavailable" rather than failing on the missing session
    if (!session) throw new Error("no frames to compute depth from");
    const r = await session.finish(m.diopters);
    session = null; metas = [];
    const { colours } = r;
    delete r.colours;
    postMessage({ type: "result", seq, result: r, colours }, [r.diopter.buffer, r.conf.buffer, colours.buffer]);
  });
};
