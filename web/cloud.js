// Geometry shared by the desktop viewer and the phone's 3D view.

// Pixel 9 main camera, 1080x1920 preview stream. Calibrated from the cutting mat's 1 cm grid
// (tools/calib_fov.py): f = 890 px at 540x960, i.e. ~57 deg across the long side.
export const F_LONG_PX = 1780;

// Flying pixels: drop points whose 3x3 neighbourhood spans more than `rel` of their depth.
// Separable 3x3 min/max on depth (computed once), so it is a few linear passes.
export function discontinuities(Z, w, h, rel) {
  const out = new Uint8Array(w * h);
  if (!(rel > 0)) return out;
  const lo = new Float32Array(w * h), hi = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x, a = Z[p - 1], b = Z[p], c = Z[p + 1];
    lo[p] = Math.min(a, b, c); hi[p] = Math.max(a, b, c);
  }
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x;
    const l = Math.min(lo[p - w], lo[p], lo[p + w]), u = Math.max(hi[p - w], hi[p], hi[p + w]);
    out[p] = u - l > rel * l ? 1 : 0;
  }
  return out;
}
