// Geometry shared by the desktop viewer and the phone's 3D view.

// Pixel 9 main camera, 1080x1920 preview stream. Calibrated from the cutting mat's 1 cm grid
// (tools/calib_fov.py): f = 890 px at 540x960, i.e. ~57 deg across the long side.
export const F_LONG_PX = 1780;
/** Focal length in pixels for a w x h image of that stream, in either orientation. */
export const focalPx = (w, h) => F_LONG_PX * (Math.max(w, h) / 1920);

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

// Floaters: drop connected blobs of points smaller than `minPx` pixels. Neighbours (4-connected)
// belong to the same blob when their depths differ by less than `rel`, so a patch that is cut off
// from every surface by depth jumps (typically noise in a textureless region) goes away as a whole.
// `keep` is updated in place.
export function dropBlobs(Z, keep, w, h, rel, minPx) {
  if (!(minPx > 1)) return keep;
  const n = w * h, parent = new Int32Array(n);
  for (let p = 0; p < n; p++) parent[p] = p;
  const find = (p) => { while (parent[p] !== p) { parent[p] = parent[parent[p]]; p = parent[p]; } return p; };
  const join = (a, b) => {
    if (Math.abs(Z[a] - Z[b]) > rel * Math.min(Z[a], Z[b])) return;
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra < rb ? rb : ra] = ra < rb ? ra : rb;
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    if (!keep[p]) continue;
    if (x > 0 && keep[p - 1]) join(p, p - 1);
    if (y > 0 && keep[p - w]) join(p, p - w);
  }
  const size = new Int32Array(n);
  for (let p = 0; p < n; p++) if (keep[p]) size[find(p)]++;
  for (let p = 0; p < n; p++) if (keep[p] && size[find(p)] < minPx) keep[p] = 0;
  return keep;
}

/** Which depth pixels become points: confident, in range, not on a depth jump, not in a small
 *  blob. `blob` is the smallest kept blob as a fraction of the image. Returns { Z, keep }. */
export function cloudMask(r, d, { conf = 1.6, edge = 0.06, blob = 0.002 } = {}) {
  const n = r.w * r.h, Z = new Float32Array(n), keep = new Uint8Array(n);
  for (let p = 0; p < n; p++) Z[p] = 1 / Math.max(1e-3, d[p]);
  const flying = discontinuities(Z, r.w, r.h, edge);
  for (let p = 0; p < n; p++) keep[p] = r.conf[p] >= conf && d[p] > 0.2 && !flying[p] && Z[p] <= 6 ? 1 : 0;
  dropBlobs(Z, keep, r.w, r.h, edge > 0 ? edge : 0.06, blob * n);
  return { Z, keep };
}
