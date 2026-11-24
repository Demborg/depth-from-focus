// JS glue for dff.wasm (crates/dff-wasm): raw C ABI, no wasm-bindgen.
export const PIXEL9_LAG_MS = 271;

// Defaults from tools/compare.py on the flat-mat stack: full-res SML focus measure, per-pixel
// curve normalisation, guided aggregation of the focus volume, centroid peak, no post-smoothing.
// halvings=2 gives a 270x480 depth map from 1080x1920 frames.
export const DEFAULTS = { win: 6, halvings: 2, smoothR: 0, measure: 0, peakR: 6, aggR: 8, aggEps: 1e-3, norm: 1 };

export async function loadDff(url = new URL("./dff.wasm", import.meta.url)) {
  const res = typeof url === "string" || url instanceof URL ? await fetch(url) : url;
  const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
  return new Dff(instance.exports);
}

export class Dff {
  constructor(x) { this.x = x; }

  /** Start a streaming stack of w x h frames: push() each frame as it arrives, then finish(). */
  begin(w, h, opts = {}) {
    const x = this.x, o = { ...DEFAULTS, ...opts };
    const ctx = x.dff_new(w, h, o.win, o.halvings, o.smoothR, o.measure, o.peakR, o.aggR, o.aggEps, o.norm);
    let n = 0, pushMs = 0;
    return {
      /** Align + focus-measure one RGBA frame now. */
      push(rgba) {
        const t = performance.now();
        new Uint8Array(x.memory.buffer, x.dff_rgba_ptr(ctx), w * h * 4).set(rgba);
        n = x.dff_push(ctx);
        pushMs += performance.now() - t;
        return n;
      },
      /** Pixels whose focus peak is now clearly behind the sweep: { w, h, px, index, conf }. */
      takeDecided(r = 3, drop = 0.6, minConf = 2.0) {
        const count = x.dff_take_decided(ctx, r, drop, minConf);
        const dims = x.dff_dims(ctx);
        const get = (k, T) => new T(x.memory.buffer, x.dff_early_ptr(ctx, k), count).slice();
        return { w: dims >>> 16, h: dims & 0xffff, px: get(0, Uint32Array), index: get(1, Float32Array), conf: get(2, Float32Array) };
      },
      scale(k) { return x.dff_scale(ctx, k); },
      /** Volume stage with the final (e.g. lag-corrected) per-frame focus values in 1/m. */
      finish(diopters) {
        const t0 = performance.now();
        try {
          new Float32Array(x.memory.buffer, x.dff_diopters_ptr(ctx, n), n).set(diopters);
          x.dff_finish(ctx);
          const ow = x.dff_out_w(ctx), oh = x.dff_out_h(ctx), len = ow * oh;
          const view = (k, l = len) => new Float32Array(x.memory.buffer, x.dff_out_ptr(ctx, k), l).slice();
          return {
            w: ow, h: oh, diopter: view(0), diopterSmooth: view(1), conf: view(2), index: view(3),
            scales: view(4, n), spread: view(5), aif: view(6), pushMs, finishMs: performance.now() - t0,
          };
        } finally {
          x.dff_free(ctx);
        }
      },
    };
  }

  /** One-shot convenience: frames = [{ rgba, diopter }]. */
  depth(frames, w, h, opts = {}) {
    const s = this.begin(w, h, opts);
    for (const f of frames) s.push(f.rgba);
    return s.finish(frames.map((f) => f.diopter));
  }
}

/** Focus (1/m) of each frame at grab time, given request times; the lens trails requests by lagMs. */
export function laggedDiopters(frames, lagMs = PIXEL9_LAG_MS) {
  const req = frames.map((f) => [f.t_req_ms, 1 / f.focus_requested]);
  return frames.map((f) => {
    const t = f.t_ms - lagMs;
    if (t <= req[0][0]) return req[0][1];
    for (let i = 1; i < req.length; i++) {
      if (t <= req[i][0]) {
        const a = (t - req[i - 1][0]) / Math.max(1e-6, req[i][0] - req[i - 1][0]);
        return req[i - 1][1] * (1 - a) + req[i][1] * a;
      }
    }
    return req[req.length - 1][1];
  });
}
