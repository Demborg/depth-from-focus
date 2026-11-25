// Lens-lag relabelling: during a continuous sweep the Pixel 9's lens trails focus requests by
// about 271 ms (up and down sweeps agree), so each frame is labelled with the focus that was
// requested lagMs before it was grabbed.
export const PIXEL9_LAG_MS = 271;

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
