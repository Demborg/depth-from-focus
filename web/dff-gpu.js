// Depth from focus on the GPU (WebGPU compute). tools/accuracy.mjs scores it on stored captures.
//
//   per frame, as it arrives
//     luma, modified Laplacian on the full-res frame, block-averaged to the measure resolution
//     breathing: blurred gradient image at quarter res, NCC against the previous frame over a
//       range of scales (read back, a few floats) -> cumulative magnification
//     focus measure rescaled about the centre, box-summed (SML), stored as one layer of the
//       focus volume; running per-pixel max / argmax / min / sum
//     all-in-focus image (colours and aggregation guide) from a 3x3 focus measure
//     early points: pixels whose peak is clearly behind the sweep (streamed to the viewer)
//   at the end
//     per-pixel curve normalisation, guided-filter aggregation of every slice (He et al.),
//     centroid peak, lens-lag-corrected diopters, all-in-focus colours
//
// The focus volume and the per-frame colours live in texture arrays (one layer per frame), so
// only the depth result is read back.

// Full resolution by default (halvings = 0: depth per pixel of the 1080x1920 stream). Radii are
// given at 540x960 (halvings = 1) and scaled with the measure resolution, so the aggregation
// covers the same part of the scene at any resolution.
// aggR 8 (16 px at full res): with a larger radius the guided filter spreads each depth jump at an
// object's outline into a ramp, which shows as sheets of points along the view rays.
export const DEFAULTS = { win: 6, halvings: 0, peakR: 6, aggR: 8, aggEps: 1e-3, norm: 1, sharp: 1 };

const SEG = 32; // outputs per thread in the sliding-window box filters

// ------------------------------------------------------------------------------------ shaders
const bil = (name, buf, w = "u.w", h = "u.h") => /* wgsl */ `
fn ${name}(x: f32, y: f32) -> f32 {
  let fx = clamp(x, 0.0, f32(${w} - 1u)); let fy = clamp(y, 0.0, f32(${h} - 1u));
  let x0 = u32(fx); let y0 = u32(fy);
  let x1 = min(x0 + 1u, ${w} - 1u); let y1 = min(y0 + 1u, ${h} - 1u);
  let ax = fx - f32(x0); let ay = fy - f32(y0);
  let a = ${buf}[y0 * ${w} + x0] * (1.0 - ax) + ${buf}[y0 * ${w} + x1] * ax;
  let b = ${buf}[y1 * ${w} + x0] * (1.0 - ax) + ${buf}[y1 * ${w} + x1] * ax;
  return a * (1.0 - ay) + b * ay;
}`;

const LUMA = /* wgsl */ `
fn luma(c: vec4f) -> f32 { return dot(c.rgb, vec3f(0.299, 0.587, 0.114)); }
fn L(x: i32, y: i32) -> f32 { return luma(textureLoad(src, vec2i(x, y), 0)); }`;

// Colour of measure pixel (x, y) in frame round(t), at its breathing-aligned position in the
// half-res colour copy.
const COLOUR_AT = /* wgsl */ `
fn colourAt(x: u32, y: u32, t: f32) -> u32 {
  let k = min(u32(max(round(t), 0.0)), u.n - 1u); let s = scales[k];
  let sx = f32(u.cw) / f32(u.w);
  let cx = (f32(u.cw) - 1.0) * 0.5; let cy = (f32(u.ch) - 1.0) * 0.5;
  let X = (f32(x) + 0.5) * sx - 0.5; let Y = (f32(y) + 0.5) * sx - 0.5;
  let ix = clamp(i32(round(cx + (X - cx) * s)), 0, i32(u.cw) - 1);
  let iy = clamp(i32(round(cy + (Y - cy) * s)), 0, i32(u.ch) - 1);
  return pack4x8unorm(textureLoad(col, vec2i(ix, iy), k, 0));
}`;

const SHADERS = {
  // Modified Laplacian |2I - I(x-1) - I(x+1)| + |2I - I(y-1) - I(y+1)| on the full-res frame,
  // averaged over BxB blocks (the measure resolution), plus the block-averaged luma.
  measure: {
    layout: ["tex", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> m0: array<f32>;
@group(0) @binding(2) var<storage, read_write> a0: array<f32>;
struct U { w: u32, h: u32, W: u32, H: u32, B: u32 }
@group(0) @binding(3) var<uniform> u: U;
${LUMA}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  var ms = 0.0; var asum = 0.0;
  for (var j = 0u; j < u.B; j++) {
    for (var i = 0u; i < u.B; i++) {
      let x = i32(id.x * u.B + i); let y = i32(id.y * u.B + j);
      let c = L(x, y);
      asum += c;
      if (x > 0 && y > 0 && x < i32(u.W) - 1 && y < i32(u.H) - 1) {
        ms += abs(2.0 * c - L(x - 1, y) - L(x + 1, y)) + abs(2.0 * c - L(x, y - 1) - L(x, y + 1));
      }
    }
  }
  let p = id.y * u.w + id.x; let inv = 1.0 / f32(u.B * u.B);
  m0[p] = ms * inv; a0[p] = asum * inv;
}`,
  },
  // Quarter-res luma (4x4 blocks) for the breathing alignment.
  quarter: {
    layout: ["tex", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
struct U { w: u32, h: u32 }
@group(0) @binding(2) var<uniform> u: U;
${LUMA}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  var s = 0.0;
  for (var j = 0; j < 4; j++) { for (var i = 0; i < 4; i++) { s += L(i32(id.x) * 4 + i, i32(id.y) * 4 + j); } }
  q[id.y * u.w + id.x] = s / 16.0;
}`,
  },
  // Half-res colour copy of the frame (2x2 blocks), one layer per frame.
  colour: {
    layout: ["tex", "st-rgba8"],
    code: /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = textureDimensions(dst);
  if (id.x >= d.x || id.y >= d.y) { return; }
  let x = i32(id.x) * 2; let y = i32(id.y) * 2;
  let c = textureLoad(src, vec2i(x, y), 0) + textureLoad(src, vec2i(x + 1, y), 0)
        + textureLoad(src, vec2i(x, y + 1), 0) + textureLoad(src, vec2i(x + 1, y + 1), 0);
  textureStore(dst, vec2i(id.xy), c * 0.25);
}`,
  },
  // Approximate Gaussian: three box passes of radius 2 = one 13-tap kernel, along (dx, dy).
  blur13: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct U { w: u32, h: u32, dx: i32, dy: i32 }
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  var k = array<f32, 13>(1., 3., 6., 10., 15., 18., 19., 18., 15., 10., 6., 3., 1.);
  var s = 0.0;
  for (var i = 0; i < 13; i++) {
    let x = clamp(i32(id.x) + (i - 6) * u.dx, 0, i32(u.w) - 1);
    let y = clamp(i32(id.y) + (i - 6) * u.dy, 0, i32(u.h) - 1);
    s += k[i] * src[u32(y) * u.w + u32(x)];
  }
  dst[id.y * u.w + id.x] = s / 125.0;
}`,
  },
  // Gradient magnitude (central differences): blur-tolerant structure for alignment.
  grad: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> b: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
struct U { w: u32, h: u32 }
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = id.y * u.w + id.x;
  if (id.x == 0u || id.y == 0u || id.x == u.w - 1u || id.y == u.h - 1u) { o[p] = 0.0; return; }
  let gx = b[p + 1u] - b[p - 1u]; let gy = b[p + u.w] - b[p - u.w];
  o[p] = sqrt(gx * gx + gy * gy);
}`,
  },
  // NCC between the previous structure image a(p) and the current one sampled at c + (p - c) s,
  // one workgroup per candidate scale, over the central region (every second row).
  ncc: {
    layout: ["ro", "ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> score: array<f32>;
struct U { w: u32, h: u32, x0: u32, x1: u32, y0: u32, y1: u32 }
@group(0) @binding(4) var<uniform> u: U;
${bil("bs", "b")}
var<workgroup> s0: array<f32, 256>; var<workgroup> s1: array<f32, 256>; var<workgroup> s2: array<f32, 256>;
var<workgroup> s3: array<f32, 256>; var<workgroup> s4: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let s = scales[wg.x];
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let nx = u.x1 - u.x0; let ny = (u.y1 - u.y0 + 1u) / 2u;
  var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0;
  for (var i = li; i < nx * ny; i += 256u) {
    let x = u.x0 + i % nx; let y = u.y0 + 2u * (i / nx);
    let va = a[y * u.w + x];
    let vb = bs(cx + (f32(x) - cx) * s, cy + (f32(y) - cy) * s);
    sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb;
  }
  s0[li] = sa; s1[li] = sb; s2[li] = saa; s3[li] = sbb; s4[li] = sab;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { s0[li] += s0[li + st]; s1[li] += s1[li + st]; s2[li] += s2[li + st]; s3[li] += s3[li + st]; s4[li] += s4[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let n = f32(nx * ny);
    let ma = s0[0] / n; let mb = s1[0] / n;
    let cov = s4[0] / n - ma * mb;
    let va = s2[0] / n - ma * ma; let vb = s3[0] / n - mb * mb;
    score[wg.x] = cov / max(sqrt(va * vb), 1e-12);
  }
}`,
  },
  // Focus measure rescaled about the centre (breathing), horizontal box sum.
  rescaleH: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> m0: array<f32>;
@group(0) @binding(1) var<storage, read_write> tmp: array<f32>;
struct U { w: u32, h: u32, r: i32, k: u32, s: f32 }
@group(0) @binding(2) var<uniform> u: U;
${bil("ms", "m0")}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let sy = cy + (f32(id.y) - cy) * u.s;
  var acc = 0.0;
  for (var d = -u.r; d <= u.r; d++) {
    let x = clamp(i32(id.x) + d, 0, i32(u.w) - 1);
    acc += ms(cx + (f32(x) - cx) * u.s, sy);
  }
  tmp[id.y * u.w + id.x] = acc / f32(2 * u.r + 1);
}`,
  },
  // Vertical box sum -> focus volume layer k; running per-pixel stats and the all-in-focus guide.
  // stats = (sum, min, max, guide value at the max), bestk = frame of the max.
  storeV: {
    layout: ["ro", "ro", "st-r32f", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmp: array<f32>;
@group(0) @binding(1) var<storage, read> a0: array<f32>;
@group(0) @binding(2) var vol: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<storage, read_write> stats: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> bestk: array<u32>;
struct U { w: u32, h: u32, r: i32, k: u32, s: f32, sharp: u32 }
@group(0) @binding(5) var<uniform> u: U;
${bil("ga", "a0")}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  var acc = 0.0;
  for (var d = -u.r; d <= u.r; d++) {
    let y = clamp(i32(id.y) + d, 0, i32(u.h) - 1);
    acc += tmp[u32(y) * u.w + id.x];
  }
  let v = acc / f32(2 * u.r + 1);
  textureStore(vol, vec2i(id.xy), vec4f(v, 0.0, 0.0, 0.0));
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let g = ga(cx + (f32(id.x) - cx) * u.s, cy + (f32(id.y) - cy) * u.s);
  let p = id.y * u.w + id.x;
  if (u.k == 0u) { stats[p] = vec4f(v, v, v, g); bestk[p] = 0u; return; }
  var st = stats[p];
  st.x += v; st.y = min(st.y, v);
  if (v > st.z) { st.z = v; if (u.sharp == 0u) { st.w = g; } bestk[p] = u.k; }
  stats[p] = st;
}`,
  },
  // Early points: a pixel is decided once its peak is >= r frames behind the sweep, its sharpness
  // has dropped below drop x peak, and its curve is peaked enough (peak / mean >= minConf).
  decide: {
    layout: ["tex-uf-a", "ro", "ro", "rw", "ro", "ro", "tex-a", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var vol: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<vec4f>;
@group(0) @binding(2) var<storage, read> bestk: array<u32>;
@group(0) @binding(3) var<storage, read_write> decided: array<u32>;
@group(0) @binding(4) var<storage, read> labels: array<f32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var col: texture_2d_array<f32>;
struct Out { count: atomic<u32>, pad0: u32, pad1: u32, pad2: u32, data: array<vec4u> }
@group(0) @binding(7) var<storage, read_write> out: Out;
struct U { w: u32, h: u32, n: u32, r: u32, drop: f32, minConf: f32, cw: u32, ch: u32 }
@group(0) @binding(8) var<uniform> u: U;
fn V(x: u32, y: u32, k: u32) -> f32 { return textureLoad(vol, vec2u(x, y), k, 0).x; }
${COLOUR_AT}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = id.y * u.w + id.x;
  if (decided[p] != 0u) { return; }
  let k = bestk[p]; let st = stats[p];
  if (k + u.r > u.n - 1u || V(id.x, id.y, u.n - 1u) > u.drop * st.z) { return; }
  let conf = st.z / max(st.x / f32(u.n), 1e-9);
  if (conf < u.minConf) { return; }
  let lo = select(0u, k - u.r, k >= u.r); let hi = k + u.r;
  var base = 3.4e38;
  for (var j = lo; j <= hi; j++) { base = min(base, V(id.x, id.y, j)); }
  var sw = 0.0; var sk = 0.0;
  for (var j = lo; j <= hi; j++) { let w = V(id.x, id.y, j) - base; sw += w; sk += w * f32(j); }
  let t = select(f32(k), sk / sw, sw > 0.0);
  decided[p] = 1u;
  let k0 = min(u.n - 1u, u32(t)); let k1 = min(u.n - 1u, k0 + 1u); let a = t - f32(k0);
  let z = 1.0 / (labels[k0] * (1.0 - a) + labels[k1] * a);
  let i = atomicAdd(&out.count, 1u);
  out.data[i] = vec4u(p, bitcast<u32>(z), colourAt(id.x, id.y, t), bitcast<u32>(conf));
}`,
  },
  // Per-pixel curve normalisation: (m - min) * w with w = clamp(conf - 1, 0, 4) / sum(m - min), so
  // every pixel votes with a peak shape and flat (textureless) curves contribute little.
  // nrm = (min, w, guide, conf).
  norm: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> stats: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> nrm: array<vec4f>;
struct U { w: u32, h: u32, n: u32, norm: u32 }
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = id.y * u.w + id.x; let st = stats[p]; let n = f32(u.n);
  let conf = st.z / max(st.x / n, 1e-9);
  if (u.norm == 0u) { nrm[p] = vec4f(0.0, 1.0, st.w, conf); return; }
  nrm[p] = vec4f(st.y, clamp(conf - 1.0, 0.0, 4.0) / max(st.x - n * st.y, 1e-12), st.w, conf);
}`,
  },
  // Peak per pixel over the (aggregated) volume: argmax, then a baseline-subtracted centroid over
  // +-R frames (or a 3-point Gaussian fit for R = 0); diopters from the per-frame labels and the
  // colour from the sharpest frame.
  // With sharp = 1 the colour is the full-res all-in-focus image (pass "sharpest") instead.
  peak: {
    layout: ["tex-uf-a", "ro", "ro", "ro", "tex-a", "rw", "rw", "rw", "uni", "ro"],
    code: /* wgsl */ `
@group(0) @binding(0) var vol: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> nrm: array<vec4f>;
@group(0) @binding(2) var<storage, read> labels: array<f32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var col: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> dio: array<f32>;
@group(0) @binding(6) var<storage, read_write> conf: array<f32>;
@group(0) @binding(7) var<storage, read_write> rgb: array<u32>;
struct U { w: u32, h: u32, n: u32, r: u32, cw: u32, ch: u32, sharp: u32 }
@group(0) @binding(8) var<uniform> u: U;
@group(0) @binding(9) var<storage, read> aif: array<vec2f>;
fn V(x: u32, y: u32, k: u32) -> f32 { return textureLoad(vol, vec2u(x, y), k, 0).x; }
${COLOUR_AT}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = id.y * u.w + id.x;
  var best = -3.4e38; var k = 0u;
  for (var j = 0u; j < u.n; j++) { let v = V(id.x, id.y, j); if (v > best) { best = v; k = j; } }
  var t = f32(k);
  if (u.r == 0u) {
    if (k > 0u && k < u.n - 1u) {
      let l = log(V(id.x, id.y, k - 1u) + 1e-6); let c = log(best + 1e-6); let r = log(V(id.x, id.y, k + 1u) + 1e-6);
      let den = l - 2.0 * c + r;
      if (den < 0.0) { t += clamp(0.5 * (l - r) / den, -0.5, 0.5); }
    }
  } else {
    let lo = select(0u, k - u.r, k >= u.r); let hi = min(k + u.r, u.n - 1u);
    var base = 3.4e38;
    for (var j = lo; j <= hi; j++) { base = min(base, V(id.x, id.y, j)); }
    var sw = 0.0; var sk = 0.0;
    for (var j = lo; j <= hi; j++) { let w = V(id.x, id.y, j) - base; sw += w; sk += w * f32(j); }
    if (sw > 0.0) { t = sk / sw; }
  }
  let i0 = min(u32(max(t, 0.0)), u.n - 1u); let i1 = min(i0 + 1u, u.n - 1u); let f = t - f32(i0);
  dio[p] = labels[i0] * (1.0 - f) + labels[i1] * f;
  conf[p] = nrm[p].w;
  rgb[p] = select(colourAt(id.x, id.y, t), bitcast<u32>(aif[p].y), u.sharp != 0u);
}`,
  },
  // All-in-focus image, per frame: each pixel keeps the colour (full-res, from the source frame)
  // and luma of the frame where a 3x3 mean of the modified Laplacian is highest so far. The focus
  // volume uses a much larger window (needed for a stable depth), and choosing colours or the
  // aggregation guide with that window blurs them at occlusion edges: pixels next to a sharp near
  // object pick the frame where the object is in focus and they themselves are blurred.
  // aif = (best 3x3 measure, packed RGBA bits); the luma goes to stats.w as the guide.
  sharpest: {
    layout: ["tex", "ro", "ro", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> m0: array<f32>;
@group(0) @binding(2) var<storage, read> a0: array<f32>;
@group(0) @binding(3) var<storage, read_write> stats: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> aif: array<vec2f>;
struct U { w: u32, h: u32, W: u32, H: u32, k: u32, s: f32, B: u32 }
@group(0) @binding(5) var<uniform> u: U;
${bil("ms", "m0")}
${bil("ga", "a0")}
fn texel(x: f32, y: f32) -> vec4f {
  let fx = clamp(x, 0.0, f32(u.W - 1u)); let fy = clamp(y, 0.0, f32(u.H - 1u));
  let x0 = i32(fx); let y0 = i32(fy);
  let x1 = min(x0 + 1, i32(u.W) - 1); let y1 = min(y0 + 1, i32(u.H) - 1);
  let ax = fx - f32(x0); let ay = fy - f32(y0);
  let a = mix(textureLoad(src, vec2i(x0, y0), 0), textureLoad(src, vec2i(x1, y0), 0), ax);
  let b = mix(textureLoad(src, vec2i(x0, y1), 0), textureLoad(src, vec2i(x1, y1), 0), ax);
  return mix(a, b, ay);
}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let X = cx + (f32(id.x) - cx) * u.s; let Y = cy + (f32(id.y) - cy) * u.s;
  var f = 0.0;
  for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) { f += ms(X + f32(dx), Y + f32(dy)); } }
  let p = id.y * u.w + id.x;
  if (u.k > 0u && f <= aif[p].x) { return; }
  let B = f32(u.B); let CX = (f32(u.W) - 1.0) * 0.5; let CY = (f32(u.H) - 1.0) * 0.5;
  let c = texel(CX + ((f32(id.x) + 0.5) * B - 0.5 - CX) * u.s, CY + ((f32(id.y) + 0.5) * B - 0.5 - CY) * u.s);
  aif[p] = vec2f(f, bitcast<f32>(pack4x8unorm(c)));
  stats[p].w = ga(X, Y);
}`,
  },
};


// Guided filter box sums with sliding windows: each thread produces SEG outputs along a row (H) or
// a column (V). The source of the H pass and the output of the V pass depend on the stage:
//   G: guide statistics      src (I, I^2)         -> gst = (mean I, var I)
//   A: slice k               src (p, I p)         -> ab = (a, b)
//   Q: coefficients          src ab               -> q = mean(a) I + mean(b), into volume layer k
function boxH(stage) {
  const bind = {
    G: [["ro", "var<storage, read> nrm: array<vec4f>"]],
    A: [["tex-uf", "var vol: texture_2d<f32>"], ["ro", "var<storage, read> nrm: array<vec4f>"]],
    Q: [["ro", "var<storage, read> ab: array<vec2f>"]],
  }[stage];
  const src = {
    G: "let I = nrm[y * u.w + u32(x)].z; return vec2f(I, I * I);",
    A: "let n = nrm[y * u.w + u32(x)]; let pv = (textureLoad(vol, vec2i(x, i32(y)), 0).x - n.x) * n.y; return vec2f(pv, n.z * pv);",
    Q: "return ab[y * u.w + u32(x)];",
  }[stage];
  const b = bind.map(([, d], i) => `@group(0) @binding(${i}) ${d};`).join("\n");
  const o = bind.length;
  return {
    layout: [...bind.map(([t]) => t), "rw", "uni"],
    code: /* wgsl */ `
${b}
@group(0) @binding(${o}) var<storage, read_write> tmp: array<vec2f>;
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(${o + 1}) var<uniform> u: U;
fn src(x: i32, y: u32) -> vec2f { ${src} }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.w + ${SEG}u - 1u) / ${SEG}u;
  if (id.x >= segs * u.h) { return; }
  let y = id.x / segs; let x0 = (id.x % segs) * ${SEG}u; let x1 = min(x0 + ${SEG}u, u.w);
  let w = i32(u.w);
  var acc = vec2f(0.0);
  for (var d = -u.r; d <= u.r; d++) { acc += src(clamp(i32(x0) + d, 0, w - 1), y); }
  let norm = 1.0 / f32(2 * u.r + 1);
  for (var x = x0; x < x1; x++) {
    tmp[y * u.w + x] = acc * norm;
    acc += src(min(i32(x) + u.r + 1, w - 1), y) - src(max(i32(x) - u.r, 0), y);
  }
}`,
  };
}

function boxV(stage) {
  const bind = {
    G: [["rw", "var<storage, read_write> gst: array<vec2f>"]],
    A: [["ro", "var<storage, read> gst: array<vec2f>"], ["rw", "var<storage, read_write> ab: array<vec2f>"]],
    Q: [["ro", "var<storage, read> nrm: array<vec4f>"], ["st-r32f", "var vol: texture_storage_2d<r32float, write>"]],
  }[stage];
  const out = {
    G: "gst[p] = vec2f(m.x, m.y - m.x * m.x);",
    A: "let g = gst[p]; let a = (m.y - g.x * m.x) / (g.y + u.eps); ab[p] = vec2f(a, m.x - a * g.x);",
    Q: "textureStore(vol, vec2i(i32(x), i32(y)), vec4f(m.x * nrm[p].z + m.y, 0.0, 0.0, 0.0));",
  }[stage];
  const b = bind.map(([, d], i) => `@group(0) @binding(${i + 1}) ${d};`).join("\n");
  return {
    layout: ["ro", ...bind.map(([t]) => t), "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmp: array<vec2f>;
${b}
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(${bind.length + 1}) var<uniform> u: U;
fn emit(x: u32, y: u32, m: vec2f) { let p = y * u.w + x; ${out} }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.h + ${SEG}u - 1u) / ${SEG}u;
  if (id.x >= segs * u.w) { return; }
  let x = id.x % u.w; let y0 = (id.x / u.w) * ${SEG}u; let y1 = min(y0 + ${SEG}u, u.h);
  let h = i32(u.h);
  var acc = vec2f(0.0);
  for (var d = -u.r; d <= u.r; d++) { acc += tmp[u32(clamp(i32(y0) + d, 0, h - 1)) * u.w + x]; }
  let norm = 1.0 / f32(2 * u.r + 1);
  for (var y = y0; y < y1; y++) {
    emit(x, y, acc * norm);
    acc += tmp[u32(min(i32(y) + u.r + 1, h - 1)) * u.w + x] - tmp[u32(max(i32(y) - u.r, 0)) * u.w + x];
  }
}`,
  };
}
for (const s of ["G", "A", "Q"]) { SHADERS["boxH" + s] = boxH(s); SHADERS["boxV" + s] = boxV(s); }

// ------------------------------------------------------------------------------------ engine
const LAYOUT = {
  tex: { texture: { sampleType: "float" } },
  "tex-uf": { texture: { sampleType: "unfilterable-float" } },
  "tex-a": { texture: { sampleType: "float", viewDimension: "2d-array" } },
  "tex-uf-a": { texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" } },
  "st-r32f": { storageTexture: { access: "write-only", format: "r32float" } },
  "st-rgba8": { storageTexture: { access: "write-only", format: "rgba8unorm" } },
  ro: { buffer: { type: "read-only-storage" } },
  rw: { buffer: { type: "storage" } },
  uni: { buffer: { type: "uniform" } },
};

export async function createDff() {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  device.lost.then((i) => console.error("WebGPU device lost:", i.message));
  device.onuncapturederror = (e) => console.error("WebGPU:", e.error.message);
  const pipes = {};
  for (const [name, s] of Object.entries(SHADERS)) {
    const layout = device.createBindGroupLayout({
      entries: s.layout.map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, ...LAYOUT[t] })),
    });
    const module = device.createShaderModule({ code: s.code, label: name });
    pipes[name] = { layout, pipeline: device.createComputePipeline({ label: name, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } }) };
  }
  return { device, info: adapter.info, begin: (opts) => new Session(device, pipes, { ...DEFAULTS, ...opts }) };
}

const U = GPUBufferUsage;

class Session {
  constructor(device, pipes, o) {
    Object.assign(this, { device, pipes, o, n: 0, cum: 1, scales: [], pushMs: 0, cap: 0, res: [] });
  }

  buf(size, usage = U.STORAGE | U.COPY_SRC | U.COPY_DST) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.res.push(b);
    return b;
  }

  /** Uniform buffer from a format string ("u" u32, "i" i32, "f" f32) and values. */
  uni(fmt, vals, b = null) {
    const ab = new ArrayBuffer(Math.ceil((fmt.length * 4) / 16) * 16), dv = new DataView(ab);
    [...fmt].forEach((c, i) => (c === "f" ? dv.setFloat32 : c === "i" ? dv.setInt32 : dv.setUint32).call(dv, i * 4, vals[i], true));
    b ??= this.buf(ab.byteLength, U.UNIFORM | U.COPY_DST);
    this.device.queue.writeBuffer(b, 0, ab);
    return b;
  }

  run(enc, name, resources, x, y = 1) {
    const { layout, pipeline } = this.pipes[name];
    const group = this.device.createBindGroup({
      layout,
      entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  async read(src, size, offset = 0) {
    const b = this.device.createBuffer({ size, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, b, 0, size);
    this.device.queue.submit([enc.finish()]);
    await b.mapAsync(GPUMapMode.READ);
    const out = b.getMappedRange().slice(0);
    b.destroy();
    return out;
  }

  alloc(W, H) {
    const o = this.o, B = 1 << o.halvings;
    const w = Math.floor(W / B), h = Math.floor(H / B), qw = Math.floor(W / 4), qh = Math.floor(H / 4);
    const N = w * h, scale = 2 ** (1 - o.halvings);
    Object.assign(this, { W, H, w, h, qw, qh, N, cw: Math.floor(W / 2), ch: Math.floor(H / 2) });
    this.src = this.device.createTexture({ size: [W, H], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    this.m0 = this.buf(N * 4); this.a0 = this.buf(N * 4); this.tmp1 = this.buf(N * 4);
    this.q = this.buf(qw * qh * 4); this.qt = this.buf(qw * qh * 4);
    this.st = [this.buf(qw * qh * 4), this.buf(qw * qh * 4)];
    this.stats = this.buf(N * 16); this.bestk = this.buf(N * 4); this.decided = this.buf(N * 4); this.aif = this.buf(N * 8);
    this.nccScales = this.buf(64 * 4); this.score = this.buf(64 * 4);
    this.scalesBuf = this.buf(1024 * 4); this.labels = this.buf(1024 * 4);
    this.out = this.buf(16 + N * 16);
    const r = Math.max(1, o.win >> o.halvings), m = 0.12;
    this.agg = Math.round(o.aggR * scale);
    this.u = {
      measure: this.uni("uuuuu", [w, h, W, H, B]),
      quarter: this.uni("uu", [qw, qh]),
      blurX: this.uni("uuii", [qw, qh, 1, 0]), blurY: this.uni("uuii", [qw, qh, 0, 1]),
      grad: this.uni("uu", [qw, qh]),
      ncc: this.uni("uuuuuu", [qw, qh, Math.floor(qw * m), Math.floor(qw * (1 - m)), Math.floor(qh * m), Math.floor(qh * (1 - m))]),
      frame: this.uni("uuiufu", [w, h, r, 0, 1, o.sharp ? 1 : 0]),
      sharpest: this.uni("uuuuufu", [w, h, W, H, 0, 1, B]),
      box: this.uni("uuif", [w, h, this.agg, o.aggEps]),
    };
    this.grow(40);
  }

  /** Room for at least n frames in the focus volume and colour arrays (copying what is there). */
  grow(n) {
    if (n <= this.cap) return;
    const cap = Math.max(n, Math.ceil(this.cap * 1.5));
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const vol = this.device.createTexture({ size: [this.w, this.h, cap], format: "r32float", usage });
    const col = this.device.createTexture({ size: [this.cw, this.ch, cap], format: "rgba8unorm", usage });
    if (this.cap) {
      const enc = this.device.createCommandEncoder();
      enc.copyTextureToTexture({ texture: this.vol }, { texture: vol }, [this.w, this.h, this.n]);
      enc.copyTextureToTexture({ texture: this.col }, { texture: col }, [this.cw, this.ch, this.n]);
      this.device.queue.submit([enc.finish()]);
      this.vol.destroy(); this.col.destroy();
    }
    Object.assign(this, { vol, col, cap });
  }

  layer(tex, k) { return tex.createView({ dimension: "2d", baseArrayLayer: k, arrayLayerCount: 1 }); }

  /** Scores of the current frame's structure image against the previous one at each scale. */
  async nccScores(list, enc = this.device.createCommandEncoder()) {
    this.device.queue.writeBuffer(this.nccScales, 0, new Float32Array(list));
    const k = this.n;
    this.run(enc, "ncc", [this.st[(k - 1) & 1], this.st[k & 1], this.nccScales, this.score, this.u.ncc], list.length);
    this.device.queue.submit([enc.finish()]);
    return new Float32Array(await this.read(this.score, list.length * 4));
  }

  /** Align and focus-measure one frame (ImageBitmap, VideoFrame, canvas...). Returns the count. */
  async push(image) {
    const t0 = performance.now();
    const W = image.width ?? image.displayWidth, H = image.height ?? image.displayHeight;
    if (!this.src) this.alloc(W, H);
    const k = this.n, d = this.device;
    this.grow(k + 1);
    d.queue.copyExternalImageToTexture({ source: image }, { texture: this.src }, [this.W, this.H]);
    const g = (n, s = 16) => Math.ceil(n / s);
    let enc = d.createCommandEncoder();
    const srcView = this.src.createView();
    this.run(enc, "measure", [srcView, this.m0, this.a0, this.u.measure], g(this.w), g(this.h));
    this.run(enc, "colour", [srcView, this.layer(this.col, k)], g(this.cw), g(this.ch));
    this.run(enc, "quarter", [srcView, this.q, this.u.quarter], g(this.qw), g(this.qh));
    this.run(enc, "blur13", [this.q, this.qt, this.u.blurX], g(this.qw), g(this.qh));
    this.run(enc, "blur13", [this.qt, this.q, this.u.blurY], g(this.qw), g(this.qh));
    this.run(enc, "grad", [this.q, this.st[k & 1], this.u.grad], g(this.qw), g(this.qh));
    if (k > 0) {
      // Adjacent frames differ by ~0.02% in magnification: search +-0.4% (fine), and +-3% (wide)
      // in case the best lands at the edge, then refine around the wide estimate.
      const fine = steps(1, 0.004, 20), wide = steps(1, 0.03, 24);
      const sc = await this.nccScores([...fine, ...wide], enc);
      let [s] = pick(sc.subarray(0, 21), 1, 0.004, 20);
      if (Math.abs(s - 1) >= 0.0038) {
        const [s0] = pick(sc.subarray(21), 1, 0.03, 24);
        [s] = pick(await this.nccScores(steps(s0, 0.003, 20)), s0, 0.003, 20);
      }
      this.cum *= s;
      enc = d.createCommandEncoder();
    }
    this.scales.push(this.cum);
    d.queue.writeBuffer(this.scalesBuf, k * 4, new Float32Array([this.cum]));
    this.uni("uuiufu", [this.w, this.h, Math.max(1, this.o.win >> this.o.halvings), k, this.cum, this.o.sharp ? 1 : 0], this.u.frame);
    this.run(enc, "rescaleH", [this.m0, this.tmp1, this.u.frame], g(this.w), g(this.h));
    this.run(enc, "storeV", [this.tmp1, this.a0, this.layer(this.vol, k), this.stats, this.bestk, this.u.frame], g(this.w), g(this.h));
    if (this.o.sharp) {
      this.uni("uuuuufu", [this.w, this.h, this.W, this.H, k, this.cum, 1 << this.o.halvings], this.u.sharpest);
      this.run(enc, "sharpest", [srcView, this.m0, this.a0, this.stats, this.aif, this.u.sharpest], g(this.w), g(this.h));
    }
    d.queue.submit([enc.finish()]);
    this.n++;
    this.pushMs += performance.now() - t0;
    return this.n;
  }

  /** Pixels decided so far (each returned once), with depth from `labels` (1/m per frame so far)
   *  and all-in-focus colour: { w, h, px, z, rgb, conf }. */
  async takeDecided(labels, r = 3, drop = 0.6, minConf = 2.0) {
    const empty = { w: this.w, h: this.h, px: new Uint32Array(0), z: new Float32Array(0), rgb: new Uint8Array(0) };
    if (this.n <= r) return empty;
    const d = this.device;
    d.queue.writeBuffer(this.labels, 0, new Float32Array(labels));
    const u = this.uni("uuuuffuu", [this.w, this.h, this.n, r, drop, minConf, this.cw, this.ch]);
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.out, 0, 16);
    this.run(enc, "decide", [this.vol.createView({ dimension: "2d-array" }), this.stats, this.bestk, this.decided, this.labels, this.scalesBuf, this.col.createView({ dimension: "2d-array" }), this.out, u], Math.ceil(this.w / 16), Math.ceil(this.h / 16));
    d.queue.submit([enc.finish()]);
    const count = new Uint32Array(await this.read(this.out, 4))[0];
    u.destroy();
    if (!count) return empty;
    const raw = new Uint32Array(await this.read(this.out, count * 16, 16));
    const px = new Uint32Array(count), z = new Float32Array(count), rgb = new Uint8Array(count * 3);
    const f = new Float32Array(raw.buffer);
    for (let i = 0; i < count; i++) {
      px[i] = raw[i * 4]; z[i] = f[i * 4 + 1];
      const c = raw[i * 4 + 2];
      rgb[i * 3] = c & 255; rgb[i * 3 + 1] = (c >> 8) & 255; rgb[i * 3 + 2] = (c >> 16) & 255;
    }
    return { w: this.w, h: this.h, px, z, rgb };
  }

  /** Aggregation and peak with the final (lag-corrected) per-frame focus in 1/m. Frees the GPU
   *  memory. diopter and conf are per measure pixel; colours are RGB bytes. */
  async finish(diopters) {
    const t0 = performance.now(), d = this.device, n = this.n, { w, h } = this;
    try {
      d.queue.writeBuffer(this.labels, 0, new Float32Array(diopters));
      const nrm = this.buf(this.N * 16), tmp = this.buf(this.N * 8), gst = this.buf(this.N * 8), ab = this.buf(this.N * 8);
      const dio = this.buf(this.N * 4), conf = this.buf(this.N * 4), rgb = this.buf(this.N * 4);
      const enc = d.createCommandEncoder();
      const g16 = (v) => Math.ceil(v / 16);
      const gH = Math.ceil((Math.ceil(w / SEG) * h) / 64), gV = Math.ceil((Math.ceil(h / SEG) * w) / 64);
      this.run(enc, "norm", [this.stats, nrm, this.uni("uuuu", [w, h, n, this.o.norm ? 1 : 0])], g16(w), g16(h));
      if (this.agg > 0) {
        this.run(enc, "boxHG", [nrm, tmp, this.u.box], gH);
        this.run(enc, "boxVG", [tmp, gst, this.u.box], gV);
        for (let k = 0; k < n; k++) {
          this.run(enc, "boxHA", [this.layer(this.vol, k), nrm, tmp, this.u.box], gH);
          this.run(enc, "boxVA", [tmp, gst, ab, this.u.box], gV);
          this.run(enc, "boxHQ", [ab, tmp, this.u.box], gH);
          this.run(enc, "boxVQ", [tmp, nrm, this.layer(this.vol, k), this.u.box], gV);
        }
      }
      const u = this.uni("uuuuuuu", [w, h, n, this.o.peakR, this.cw, this.ch, this.o.sharp ? 1 : 0]);
      this.run(enc, "peak", [this.vol.createView({ dimension: "2d-array" }), nrm, this.labels, this.scalesBuf, this.col.createView({ dimension: "2d-array" }), dio, conf, rgb, u, this.aif], g16(w), g16(h));
      d.queue.submit([enc.finish()]);
      const [dv, fv, cv] = await Promise.all([this.read(dio, this.N * 4), this.read(conf, this.N * 4), this.read(rgb, this.N * 4)]);
      const diopter = new Float32Array(dv), C = new Uint8Array(cv), colours = new Uint8Array(this.N * 3);
      for (let p = 0, q = 0; p < this.N * 4; p += 4, q += 3) { colours[q] = C[p]; colours[q + 1] = C[p + 1]; colours[q + 2] = C[p + 2]; }
      return {
        w, h, diopter, diopterSmooth: diopter, conf: new Float32Array(fv), colours,
        scales: new Float32Array(this.scales), pushMs: this.pushMs, finishMs: performance.now() - t0,
      };
    } finally {
      this.free();
    }
  }

  free() {
    for (const b of this.res) b.destroy();
    this.res = [];
    this.src?.destroy(); this.vol?.destroy(); this.col?.destroy();
  }
}

const steps = (c, range, n) => Array.from({ length: n + 1 }, (_, i) => c - range + (2 * range * i) / n);

/** Best scale from scores at steps(c, range, n), with parabolic refinement. */
function pick(scores, c, range, n) {
  let k = 0;
  for (let i = 1; i <= n; i++) if (scores[i] > scores[k]) k = i;
  let s = c - range + (2 * range * k) / n;
  if (k > 0 && k < n) {
    const [l, m, r] = [scores[k - 1], scores[k], scores[k + 1]], den = l - 2 * m + r;
    if (den < 0) s += ((0.5 * (l - r)) / den) * ((2 * range) / n);
  }
  return [s, scores[k]];
}
