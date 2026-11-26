// Depth from focus on the GPU (WebGPU compute). tools/accuracy.mjs scores it on stored captures.
//
//   per frame, as it arrives
//     luma, modified Laplacian on the full-res frame (tiled, fused with quarter-res downsample)
//     breathing + optical-center shift: blurred gradient image at quarter res, 3-DOF Huber-IRLS
//       Gauss-Newton alignment (scale, dx, dy) entirely on the GPU (no CPU readback stall)
//     focus measure warped and box-summed (SML), stored as one layer of the focus volume;
//       running per-pixel max / argmax / min / sum
//     all-in-focus image (colours and aggregation guide) from a 3x3 focus measure
//     early points: pixels whose peak is clearly behind the sweep (streamed to the viewer)
//   at the end
//     per-pixel curve normalisation with denoised guide, 4-slice vectorised guided-filter
//     aggregation (He et al.), diopter-space trapezoidal centroid peak, separable colour-guided
//     weighted median depth refinement, GPU-packed RGB readback
//
// The focus volume lives in a texture array (one layer per frame), so only the depth result is
// read back. Core WebGPU 1.0 only (<= 16 KB workgroup memory, <= 8 storage buffers/stage,
// <= 4 storage textures/stage, <= 128 MB per buffer) so it runs identically on Pixel 9 / Mali.

// Full resolution by default (halvings = 0: depth per pixel of the 1080x1920 stream). Radii are
// given at 540x960 (halvings = 1) and scaled with the measure resolution, so the aggregation
// covers the same part of the scene at any resolution.
// aggR 8 (16 px at full res): with a larger radius the guided filter spreads each depth jump at an
// object's outline into a ramp, which shows as sheets of points along the view rays.
export const DEFAULTS = { win: 6, halvings: 0, peakR: 6, aggR: 8, aggEps: 1e-3, norm: 1, sharp: 1 };

const SEG = 32; // outputs per thread in the sliding-window vertical box filters

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
fn luma(c: vec4f) -> f32 { return dot(c.rgb, vec3f(0.299, 0.587, 0.114)); }`;

// Source-texture helpers shared between the fast texture_external (VideoFrame) path and the
// texture_2d<f32> (copyExternalImageToTexture) fallback.
const SRC_DECL = (ext) => (ext ? "var src: texture_external;" : "var src: texture_2d<f32>;");
const SRC_LOAD = (ext, p) => (ext ? `textureLoad(src, ${p})` : `textureLoad(src, ${p}, 0)`);
const SRC_SAMPLE = (ext, uv) => (ext ? `textureSampleBaseClampToEdge(src, samp, ${uv})` : `textureSampleLevel(src, samp, ${uv}, 0.0)`);

// Colour of measure pixel (x, y) in frame round(t), at its aligned position in the half-res copy
// (only used when sharp = 0).
const COLOUR_AT = /* wgsl */ `
fn colourAt(x: u32, y: u32, t: f32) -> u32 {
  let k = min(u32(max(round(t), 0.0)), u.n - 1u); let tr = xform[k]; let s = tr.x;
  let sx = f32(u.cw) / f32(u.w);
  let cx = (f32(u.cw) - 1.0) * 0.5; let cy = (f32(u.ch) - 1.0) * 0.5;
  let X = (f32(x) + 0.5) * sx - 0.5; let Y = (f32(y) + 0.5) * sx - 0.5;
  let ix = clamp(i32(round(cx + (X - cx) * s + tr.y * 2.0)), 0, i32(u.cw) - 1);
  let iy = clamp(i32(round(cy + (Y - cy) * s + tr.z * 2.0)), 0, i32(u.ch) - 1);
  return pack4x8unorm(textureLoad(col, vec2i(ix, iy), k, 0));
}`;

// Build the per-source-type shaders (ext = true for texture_external, false for texture_2d<f32>).
function srcShaders(ext) {
  const srcT = ext ? "tex-ext" : "tex";
  const sfx = ext ? "" : "2d";
  return {
    // Full-res (B = 1): shared-memory tiled Modified Laplacian + fused 4x4 quarter-res downsample.
    ["measureQ1" + sfx]: {
      layout: [srcT, "rw", "rw", "rw", "uni"],
      code: /* wgsl */ `
@group(0) @binding(0) ${SRC_DECL(ext)}
@group(0) @binding(1) var<storage, read_write> m0: array<f32>;
@group(0) @binding(2) var<storage, read_write> a0: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
struct U { w: u32, h: u32, qw: u32, qh: u32, writeA0: u32 }
@group(0) @binding(4) var<uniform> u: U;
${LUMA}
var<workgroup> tile: array<f32, 324>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let bx = i32(wg.x * 16u) - 1; let by = i32(wg.y * 16u) - 1;
  let W = i32(u.w); let H = i32(u.h);
  for (var t = li; t < 324u; t += 256u) {
    let x = clamp(bx + i32(t % 18u), 0, W - 1); let y = clamp(by + i32(t / 18u), 0, H - 1);
    tile[t] = luma(${SRC_LOAD(ext, "vec2i(x, y)")});
  }
  workgroupBarrier();
  let gx = wg.x * 16u + lid.x; let gy = wg.y * 16u + lid.y;
  if (gx < u.w && gy < u.h) {
    let idx = (lid.y + 1u) * 18u + (lid.x + 1u);
    let c = tile[idx];
    var ml = 0.0;
    if (gx > 0u && gy > 0u && gx < u.w - 1u && gy < u.h - 1u) {
      ml = abs(2.0 * c - tile[idx - 1u] - tile[idx + 1u]) + abs(2.0 * c - tile[idx - 18u] - tile[idx + 18u]);
    }
    let p = gy * u.w + gx;
    m0[p] = ml;
    if (u.writeA0 != 0u) { a0[p] = c; }
  }
  if ((lid.x & 3u) == 0u && (lid.y & 3u) == 0u) {
    let qx = wg.x * 4u + (lid.x >> 2u); let qy = wg.y * 4u + (lid.y >> 2u);
    if (qx < u.qw && qy < u.qh) {
      var s = 0.0;
      let base = (lid.y + 1u) * 18u + (lid.x + 1u);
      for (var j = 0u; j < 4u; j++) { for (var i = 0u; i < 4u; i++) { s += tile[base + j * 18u + i]; } }
      q[qy * u.qw + qx] = s * 0.0625;
    }
  }
}`,
    },
    // Downsampled measure (halvings >= 1, B >= 2): Modified Laplacian averaged over BxB blocks.
    ["measure" + sfx]: {
      layout: [srcT, "rw", "rw", "uni"],
      code: /* wgsl */ `
@group(0) @binding(0) ${SRC_DECL(ext)}
@group(0) @binding(1) var<storage, read_write> m0: array<f32>;
@group(0) @binding(2) var<storage, read_write> a0: array<f32>;
struct U { w: u32, h: u32, W: u32, H: u32, B: u32 }
@group(0) @binding(3) var<uniform> u: U;
${LUMA}
fn L(x: i32, y: i32) -> f32 { return luma(${SRC_LOAD(ext, "vec2i(x, y)")}); }
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
    // Quarter-res luma (4x4 blocks) for breathing alignment when halvings >= 1.
    ["quarter" + sfx]: {
      layout: [srcT, "rw", "uni"],
      code: /* wgsl */ `
@group(0) @binding(0) ${SRC_DECL(ext)}
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
struct U { w: u32, h: u32 }
@group(0) @binding(2) var<uniform> u: U;
${LUMA}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  var s = 0.0;
  for (var j = 0; j < 4; j++) { for (var i = 0; i < 4; i++) { s += luma(${SRC_LOAD(ext, "vec2i(i32(id.x) * 4 + i, i32(id.y) * 4 + j)")}); } }
  q[id.y * u.w + id.x] = s * 0.0625;
}`,
    },
    // Half-res colour copy (2x2 blocks), only run when sharp = 0.
    ["colour" + sfx]: {
      layout: [srcT, "st-rgba8"],
      code: /* wgsl */ `
@group(0) @binding(0) ${SRC_DECL(ext)}
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = textureDimensions(dst);
  if (id.x >= d.x || id.y >= d.y) { return; }
  let x = i32(id.x) * 2; let y = i32(id.y) * 2;
  let c = ${SRC_LOAD(ext, "vec2i(x, y)")} + ${SRC_LOAD(ext, "vec2i(x + 1, y)")}
        + ${SRC_LOAD(ext, "vec2i(x, y + 1)")} + ${SRC_LOAD(ext, "vec2i(x + 1, y + 1)")};
  textureStore(dst, vec2i(id.xy), c * 0.25);
}`,
    },
    // All-in-focus image, per frame: each pixel keeps the colour (full-res, hardware-filtered
    // from the source frame) and luma of the frame where a 3x3 mean of the modified Laplacian is
    // highest so far. Tiled 18x18 shared memory avoids redundant bilinear m0 lookups.
    ["sharpest" + sfx]: {
      layout: [srcT, "samp", "ro", "ro", "rw", "rw", "uni"],
      code: /* wgsl */ `
@group(0) @binding(0) ${SRC_DECL(ext)}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> m0: array<f32>;
@group(0) @binding(3) var<storage, read> xform: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> stats: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> aif: array<vec2f>;
struct U { w: u32, h: u32, W: u32, H: u32, k: u32, B: u32, qscale: f32 }
@group(0) @binding(6) var<uniform> u: U;
${LUMA}
${bil("ms", "m0")}
var<workgroup> tile: array<f32, 324>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let tr = xform[u.k];
  let s = tr.x; let tx = tr.y * u.qscale; let ty = tr.z * u.qscale;
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let bx = i32(wg.x * 16u) - 1; let by = i32(wg.y * 16u) - 1;
  for (var t = li; t < 324u; t += 256u) {
    let sx = f32(bx + i32(t % 18u)); let sy = f32(by + i32(t / 18u));
    tile[t] = ms(cx + (sx - cx) * s + tx, cy + (sy - cy) * s + ty);
  }
  workgroupBarrier();
  let gx = wg.x * 16u + lid.x; let gy = wg.y * 16u + lid.y;
  if (gx >= u.w || gy >= u.h) { return; }
  let idx = (lid.y + 1u) * 18u + (lid.x + 1u);
  let f = tile[idx - 19u] + tile[idx - 18u] + tile[idx - 17u]
        + tile[idx - 1u]  + tile[idx]       + tile[idx + 1u]
        + tile[idx + 17u] + tile[idx + 18u] + tile[idx + 19u];
  let p = gy * u.w + gx;
  if (u.k > 0u && f <= aif[p].x) { return; }
  let B = f32(u.B); let CX = (f32(u.W) - 1.0) * 0.5; let CY = (f32(u.H) - 1.0) * 0.5;
  let fx = CX + ((f32(gx) + 0.5) * B - 0.5 - CX) * s + tr.y * 4.0;
  let fy = CY + ((f32(gy) + 0.5) * B - 0.5 - CY) * s + tr.z * 4.0;
  let uv = vec2f((fx + 0.5) / f32(u.W), (fy + 0.5) / f32(u.H));
  let c = ${SRC_SAMPLE(ext, "uv")};
  aif[p] = vec2f(f, bitcast<f32>(pack4x8unorm(c)));
  stats[p].w = luma(c);
}`,
    },
  };
}

const SHADERS = {
  ...srcShaders(true),
  ...srcShaders(false),
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
  // 3-DOF Huber-IRLS Gauss-Newton alignment (scale, dx, dy) between adjacent quarter-res structure
  // frames. Captures both focus breathing and reversible lens/VCM optical-center translation while
  // staying robust to moving foreground objects and avoiding any CPU-GPU readback stall.
  gnAccum: {
    layout: ["ro", "ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> rel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> acc: array<f32>;
struct U { w: u32, h: u32, x0: u32, x1: u32, y0: u32, y1: u32 }
@group(0) @binding(4) var<uniform> u: U;
${bil("bs", "b")}
var<workgroup> sh: array<f32, 2304>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li: u32) {
  let tr = rel[0]; let s = tr.x; let tx = tr.y; let ty = tr.z;
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let nx = (u.x1 - u.x0 + 1u) / 2u; let ny = (u.y1 - u.y0 + 1u) / 2u;
  let total = nx * ny;
  var h00 = 0.0; var h01 = 0.0; var h02 = 0.0; var h11 = 0.0; var h12 = 0.0; var h22 = 0.0;
  var g0 = 0.0; var g1 = 0.0; var g2 = 0.0;
  for (var i = li; i < total; i += 256u) {
    let x = u.x0 + 2u * (i % nx); let y = u.y0 + 2u * (i / nx);
    let dx = f32(x) - cx; let dy = f32(y) - cy;
    let wx = cx + dx * s + tx; let wy = cy + dy * s + ty;
    let va = a[y * u.w + x]; let vb = bs(wx, wy);
    let gx = 0.5 * (bs(wx + 1.0, wy) - bs(wx - 1.0, wy));
    let gy = 0.5 * (bs(wx, wy + 1.0) - bs(wx, wy - 1.0));
    let r = vb - va; let ar = abs(r);
    let wt = select(1.0, 0.02 / ar, ar > 0.02);
    let j0 = gx * dx + gy * dy; let j1 = gx; let j2 = gy;
    h00 += wt * j0 * j0; h01 += wt * j0 * j1; h02 += wt * j0 * j2;
    h11 += wt * j1 * j1; h12 += wt * j1 * j2; h22 += wt * j2 * j2;
    g0 += wt * j0 * r; g1 += wt * j1 * r; g2 += wt * j2 * r;
  }
  sh[li] = h00; sh[256u + li] = h01; sh[512u + li] = h02;
  sh[768u + li] = h11; sh[1024u + li] = h12; sh[1280u + li] = h22;
  sh[1536u + li] = g0; sh[1792u + li] = g1; sh[2048u + li] = g2;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { for (var c = 0u; c < 9u; c++) { sh[c * 256u + li] += sh[c * 256u + li + st]; } }
    workgroupBarrier();
  }
  if (li < 9u) { acc[li] = sh[li * 256u]; }
}`,
  },
  // Solve the 3x3 normal equations and compose into cumulative per-frame transform xform[k].
  gnStep: {
    layout: ["ro", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> acc: array<f32>;
@group(0) @binding(1) var<storage, read_write> rel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> xform: array<vec4f>;
struct U { k: u32, last: u32 }
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(1) fn main() {
  if (u.k == 0u) {
    rel[0] = vec4f(1.0, 0.0, 0.0, 0.0);
    xform[0] = vec4f(1.0, 0.0, 0.0, 0.0);
    return;
  }
  let h00 = acc[0] + 1e-6; let h01 = acc[1]; let h02 = acc[2];
  let h11 = acc[3] + 1e-4; let h12 = acc[4]; let h22 = acc[5] + 1e-4;
  let g0 = acc[6]; let g1 = acc[7]; let g2 = acc[8];
  let c00 = h11 * h22 - h12 * h12; let c01 = h02 * h12 - h01 * h22; let c02 = h01 * h12 - h02 * h11;
  let c11 = h00 * h22 - h02 * h02; let c12 = h01 * h02 - h00 * h12; let c22 = h00 * h11 - h01 * h01;
  let det = h00 * c00 + h01 * c01 + h02 * c02;
  var tr = rel[0];
  if (abs(det) > 1e-12) {
    let inv = -1.0 / det;
    tr.x = clamp(tr.x + clamp((c00 * g0 + c01 * g1 + c02 * g2) * inv, -0.02, 0.02), 0.96, 1.04);
    tr.y = clamp(tr.y + clamp((c01 * g0 + c11 * g1 + c12 * g2) * inv, -0.75, 0.75), -0.75, 0.75);
    tr.z = clamp(tr.z + clamp((c02 * g0 + c12 * g1 + c22 * g2) * inv, -0.75, 0.75), -0.75, 0.75);
  }
  if (u.last != 0u) {
    let prev = xform[u.k - 1u];
    xform[u.k] = vec4f(prev.x * tr.x, prev.y * tr.x + tr.y, prev.z * tr.x + tr.z, 0.0);
    rel[0] = vec4f(1.0, 0.0, 0.0, 0.0);
  } else {
    rel[0] = tr;
  }
}`,
  },
  // Focus measure warped by xform[k], horizontal box sum using a 128 + 2r shared-memory tile.
  rescaleH: {
    layout: ["ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> m0: array<f32>;
@group(0) @binding(1) var<storage, read> xform: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tmp: array<f32>;
struct U { w: u32, h: u32, r: i32, k: u32, qscale: f32, sharp: u32 }
@group(0) @binding(3) var<uniform> u: U;
${bil("ms", "m0")}
var<workgroup> row: array<f32, 256>;
@compute @workgroup_size(128, 1) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let y = wg.y; if (y >= u.h) { return; }
  let tr = xform[u.k]; let s = tr.x; let tx = tr.y * u.qscale; let ty = tr.z * u.qscale;
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let sy = cy + (f32(y) - cy) * s + ty;
  let r = min(u.r, 64); let x0 = i32(wg.x * 128u); let W = i32(u.w);
  for (var t = i32(lid.x); t < 128 + 2 * r; t += 128) {
    let sx = clamp(x0 - r + t, 0, W - 1);
    row[u32(t)] = ms(cx + (f32(sx) - cx) * s + tx, sy);
  }
  workgroupBarrier();
  let gx = wg.x * 128u + lid.x;
  if (gx < u.w) {
    var acc = 0.0; let c = i32(lid.x) + r;
    for (var d = -r; d <= r; d++) { acc += row[u32(c + d)]; }
    tmp[y * u.w + gx] = acc / f32(2 * r + 1);
  }
}`,
  },
  // Vertical sliding-window box sum -> focus volume layer k; running per-pixel stats and bestk.
  // stats = (sum, min, max, guide value at the max). Skips a0 reads when sharp = 1.
  storeV: {
    layout: ["ro", "ro", "ro", "st-r32f", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmp: array<f32>;
@group(0) @binding(1) var<storage, read> a0: array<f32>;
@group(0) @binding(2) var<storage, read> xform: array<vec4f>;
@group(0) @binding(3) var vol: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<storage, read_write> stats: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> bestk: array<u32>;
struct U { w: u32, h: u32, r: i32, k: u32, qscale: f32, sharp: u32 }
@group(0) @binding(6) var<uniform> u: U;
${bil("ga", "a0")}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.h + ${SEG - 1}u) / ${SEG}u;
  if (id.x >= segs * u.w) { return; }
  let x = id.x % u.w; let y0 = (id.x / u.w) * ${SEG}u; let y1 = min(y0 + ${SEG}u, u.h);
  let H = i32(u.h); let r = u.r; let inv = 1.0 / f32(2 * r + 1);
  let tr = xform[u.k]; let s = tr.x; let tx = tr.y * u.qscale; let ty = tr.z * u.qscale;
  let cx = (f32(u.w) - 1.0) * 0.5; let cy = (f32(u.h) - 1.0) * 0.5;
  let gx = cx + (f32(x) - cx) * s + tx;
  var acc = 0.0;
  for (var d = -r; d <= r; d++) { acc += tmp[u32(clamp(i32(y0) + d, 0, H - 1)) * u.w + x]; }
  for (var y = y0; y < y1; y++) {
    let v = acc * inv;
    textureStore(vol, vec2i(i32(x), i32(y)), vec4f(v, 0.0, 0.0, 0.0));
    let p = y * u.w + x;
    if (u.k == 0u) {
      let g = select(ga(gx, cy + (f32(y) - cy) * s + ty), 0.0, u.sharp != 0u);
      stats[p] = vec4f(v, v, v, g);
      bestk[p] = 0u;
    } else {
      var st = stats[p];
      st.x += v; st.y = min(st.y, v);
      if (v > st.z) {
        st.z = v;
        if (u.sharp == 0u) { st.w = ga(gx, cy + (f32(y) - cy) * s + ty); }
        bestk[p] = u.k;
      }
      stats[p] = st;
    }
    acc += tmp[u32(min(i32(y) + r + 1, H - 1)) * u.w + x] - tmp[u32(max(i32(y) - r, 0)) * u.w + x];
  }
}`,
  },
  // Early points: a pixel is decided once its peak is >= r frames behind the sweep, its sharpness
  // has dropped below drop x peak, and its curve is peaked enough (peak / mean >= minConf).
  // Uses diopter-space trapezoidal quadrature and full-res aif colour when sharp = 1.
  decide: {
    layout: ["tex-uf-a", "ro", "ro", "rw", "ro", "ro", "tex-a", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var vol: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<vec4f>;
@group(0) @binding(2) var<storage, read> bestk: array<u32>;
@group(0) @binding(3) var<storage, read_write> decided: array<u32>;
@group(0) @binding(4) var<storage, read> labels: array<f32>;
@group(0) @binding(5) var<storage, read> xform: array<vec4f>;
@group(0) @binding(6) var col: texture_2d_array<f32>;
@group(0) @binding(7) var<storage, read> aif: array<vec2f>;
struct Out { count: atomic<u32>, pad0: u32, pad1: u32, pad2: u32, data: array<vec4u> }
@group(0) @binding(8) var<storage, read_write> out: Out;
struct U { w: u32, h: u32, n: u32, r: u32, drop: f32, minConf: f32, cw: u32, ch: u32, sharp: u32 }
@group(0) @binding(9) var<uniform> u: U;
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
  var sw = 0.0; var sd = 0.0; var sk = 0.0;
  for (var j = lo; j <= hi; j++) {
    let jL = select(0u, j - 1u, j > 0u); let jR = min(j + 1u, u.n - 1u);
    let dd = 0.5 * abs(labels[jR] - labels[jL]);
    let w = max(V(id.x, id.y, j) - base, 0.0) * dd;
    sw += w; sd += w * labels[j]; sk += w * f32(j);
  }
  let d = select(labels[k], sd / sw, sw > 0.0);
  let t = select(f32(k), sk / sw, sw > 0.0);
  decided[p] = 1u;
  let z = 1.0 / max(d, 1e-6);
  let c = select(colourAt(id.x, id.y, t), bitcast<u32>(aif[p].y), u.sharp != 0u);
  let i = atomicAdd(&out.count, 1u);
  out.data[i] = vec4u(p, bitcast<u32>(z), c, bitcast<u32>(conf));
}`,
  },
  // Per-pixel curve normalisation + light guide denoising (box of radius smR = 2 at full res,
  // 1 at half res) to remove 3x3 sharpest-frame salt-and-pepper speckle from the guided-filter
  // guide while keeping the rendered all-in-focus RGB colours untouched.
  // nrm = (min, w, smoothGuide, conf).
  normSmooth: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> stats: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> nrm: array<vec4f>;
struct U { w: u32, h: u32, n: u32, norm: u32, smR: u32 }
@group(0) @binding(2) var<uniform> u: U;
var<workgroup> tile: array<f32, 400>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let bx = i32(wg.x * 16u) - 2; let by = i32(wg.y * 16u) - 2;
  let W = i32(u.w); let H = i32(u.h);
  for (var t = li; t < 400u; t += 256u) {
    let x = clamp(bx + i32(t % 20u), 0, W - 1); let y = clamp(by + i32(t / 20u), 0, H - 1);
    tile[t] = stats[u32(y) * u.w + u32(x)].w;
  }
  workgroupBarrier();
  let gx = wg.x * 16u + lid.x; let gy = wg.y * 16u + lid.y;
  if (gx >= u.w || gy >= u.h) { return; }
  let p = gy * u.w + gx; let st = stats[p]; let n = f32(u.n);
  let conf = st.z / max(st.x / n, 1e-9);
  let r = i32(min(u.smR, 2u));
  var g = tile[(lid.y + 2u) * 20u + (lid.x + 2u)];
  if (r > 0) {
    var acc = 0.0;
    for (var dy = -r; dy <= r; dy++) {
      let row = u32(i32(lid.y) + 2 + dy) * 20u;
      for (var dx = -r; dx <= r; dx++) { acc += tile[row + u32(i32(lid.x) + 2 + dx)]; }
    }
    let side = f32(2 * r + 1);
    g = acc / (side * side);
  }
  if (u.norm == 0u) { nrm[p] = vec4f(0.0, 1.0, g, conf); return; }
  nrm[p] = vec4f(st.y, clamp(conf - 1.0, 0.0, 4.0) / max(st.x - n * st.y, 1e-12), g, conf);
}`,
  },
  // Guide statistics (mean I, var I) for the guided filter.
  boxHG: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> nrm: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> tmp: array<vec2f>;
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.w + ${SEG - 1}u) / ${SEG}u;
  if (id.x >= segs * u.h) { return; }
  let y = id.x / segs; let x0 = (id.x % segs) * ${SEG}u; let x1 = min(x0 + ${SEG}u, u.w);
  let W = i32(u.w);
  var acc = vec2f(0.0);
  for (var d = -u.r; d <= u.r; d++) { let I = nrm[y * u.w + u32(clamp(i32(x0) + d, 0, W - 1))].z; acc += vec2f(I, I * I); }
  let inv = 1.0 / f32(2 * u.r + 1);
  for (var x = x0; x < x1; x++) {
    tmp[y * u.w + x] = acc * inv;
    let Ia = nrm[y * u.w + u32(min(i32(x) + u.r + 1, W - 1))].z;
    let Ib = nrm[y * u.w + u32(max(i32(x) - u.r, 0))].z;
    acc += vec2f(Ia - Ib, Ia * Ia - Ib * Ib);
  }
}`,
  },
  boxVG: {
    layout: ["ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmp: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> gst: array<vec2f>;
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.h + ${SEG - 1}u) / ${SEG}u;
  if (id.x >= segs * u.w) { return; }
  let x = id.x % u.w; let y0 = (id.x / u.w) * ${SEG}u; let y1 = min(y0 + ${SEG}u, u.h);
  let H = i32(u.h);
  var acc = vec2f(0.0);
  for (var d = -u.r; d <= u.r; d++) { acc += tmp[u32(clamp(i32(y0) + d, 0, H - 1)) * u.w + x]; }
  let inv = 1.0 / f32(2 * u.r + 1);
  for (var y = y0; y < y1; y++) {
    let m = acc * inv;
    gst[y * u.w + x] = vec2f(m.x, m.y - m.x * m.x);
    acc += tmp[u32(min(i32(y) + u.r + 1, H - 1)) * u.w + x] - tmp[u32(max(i32(y) - u.r, 0)) * u.w + x];
  }
}`,
  },
  // 4-slice vectorised guided filter (exact full-resolution He et al.): coalesced 128-wide
  // shared-memory horizontal passes + sliding-window vertical passes cut memory passes 4x.
  boxHA4: {
    layout: ["tex-uf-a", "ro", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var vol: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> nrm: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tmpP: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> tmpIP: array<vec4f>;
struct U { w: u32, h: u32, r: i32, k0: u32, n: u32 }
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> shP: array<vec4f, 256>;
var<workgroup> shIP: array<vec4f, 256>;
@compute @workgroup_size(128, 1) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let y = wg.y; if (y >= u.h) { return; }
  let r = min(u.r, 64); let W = i32(u.w); let x0 = i32(wg.x * 128u);
  let k0 = u.k0; let k1 = min(k0 + 1u, u.n - 1u); let k2 = min(k0 + 2u, u.n - 1u); let k3 = min(k0 + 3u, u.n - 1u);
  for (var t = i32(lid.x); t < 128 + 2 * r; t += 128) {
    let sx = clamp(x0 - r + t, 0, W - 1);
    let nv = nrm[y * u.w + u32(sx)];
    let v = vec4f(
      textureLoad(vol, vec2i(sx, i32(y)), k0, 0).x,
      textureLoad(vol, vec2i(sx, i32(y)), k1, 0).x,
      textureLoad(vol, vec2i(sx, i32(y)), k2, 0).x,
      textureLoad(vol, vec2i(sx, i32(y)), k3, 0).x
    );
    let pv = (v - vec4f(nv.x)) * nv.y;
    shP[u32(t)] = pv; shIP[u32(t)] = pv * nv.z;
  }
  workgroupBarrier();
  let gx = wg.x * 128u + lid.x;
  if (gx < u.w) {
    var ap = vec4f(0.0); var aip = vec4f(0.0); let c = i32(lid.x) + r;
    for (var d = -r; d <= r; d++) { ap += shP[u32(c + d)]; aip += shIP[u32(c + d)]; }
    let inv = 1.0 / f32(2 * r + 1);
    let idx = y * u.w + gx; tmpP[idx] = ap * inv; tmpIP[idx] = aip * inv;
  }
}`,
  },
  boxVA4: {
    layout: ["ro", "ro", "ro", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmpP: array<vec4f>;
@group(0) @binding(1) var<storage, read> tmpIP: array<vec4f>;
@group(0) @binding(2) var<storage, read> gst: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> abA: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> abB: array<vec4f>;
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(5) var<uniform> u: U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.h + ${SEG - 1}u) / ${SEG}u;
  if (id.x >= segs * u.w) { return; }
  let x = id.x % u.w; let y0 = (id.x / u.w) * ${SEG}u; let y1 = min(y0 + ${SEG}u, u.h);
  let H = i32(u.h);
  var ap = vec4f(0.0); var aip = vec4f(0.0);
  for (var d = -u.r; d <= u.r; d++) {
    let idx = u32(clamp(i32(y0) + d, 0, H - 1)) * u.w + x;
    ap += tmpP[idx]; aip += tmpIP[idx];
  }
  let inv = 1.0 / f32(2 * u.r + 1);
  for (var y = y0; y < y1; y++) {
    let p = y * u.w + x; let mp = ap * inv; let mip = aip * inv; let g = gst[p];
    let a = (mip - g.x * mp) / (g.y + u.eps);
    abA[p] = a; abB[p] = mp - a * g.x;
    let iA = u32(min(i32(y) + u.r + 1, H - 1)) * u.w + x;
    let iS = u32(max(i32(y) - u.r, 0)) * u.w + x;
    ap += tmpP[iA] - tmpP[iS]; aip += tmpIP[iA] - tmpIP[iS];
  }
}`,
  },
  boxHQ4: {
    layout: ["ro", "ro", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> abA: array<vec4f>;
@group(0) @binding(1) var<storage, read> abB: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tmpP: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> tmpIP: array<vec4f>;
struct U { w: u32, h: u32, r: i32, eps: f32 }
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> shA: array<vec4f, 256>;
var<workgroup> shB: array<vec4f, 256>;
@compute @workgroup_size(128, 1) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let y = wg.y; if (y >= u.h) { return; }
  let r = min(u.r, 64); let W = i32(u.w); let x0 = i32(wg.x * 128u);
  for (var t = i32(lid.x); t < 128 + 2 * r; t += 128) {
    let idx = y * u.w + u32(clamp(x0 - r + t, 0, W - 1));
    shA[u32(t)] = abA[idx]; shB[u32(t)] = abB[idx];
  }
  workgroupBarrier();
  let gx = wg.x * 128u + lid.x;
  if (gx < u.w) {
    var aa = vec4f(0.0); var ab = vec4f(0.0); let c = i32(lid.x) + r;
    for (var d = -r; d <= r; d++) { aa += shA[u32(c + d)]; ab += shB[u32(c + d)]; }
    let inv = 1.0 / f32(2 * r + 1);
    let idx = y * u.w + gx; tmpP[idx] = aa * inv; tmpIP[idx] = ab * inv;
  }
}`,
  },
  boxVQ4: {
    layout: ["ro", "ro", "ro", "st-r32f", "st-r32f", "st-r32f", "st-r32f", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> tmpP: array<vec4f>;
@group(0) @binding(1) var<storage, read> tmpIP: array<vec4f>;
@group(0) @binding(2) var<storage, read> nrm: array<vec4f>;
@group(0) @binding(3) var v0: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var v1: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var v2: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var v3: texture_storage_2d<r32float, write>;
struct U { w: u32, h: u32, r: i32, cnt: u32 }
@group(0) @binding(7) var<uniform> u: U;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let segs = (u.h + ${SEG - 1}u) / ${SEG}u;
  if (id.x >= segs * u.w) { return; }
  let x = id.x % u.w; let y0 = (id.x / u.w) * ${SEG}u; let y1 = min(y0 + ${SEG}u, u.h);
  let H = i32(u.h);
  var aa = vec4f(0.0); var ab = vec4f(0.0);
  for (var d = -u.r; d <= u.r; d++) {
    let idx = u32(clamp(i32(y0) + d, 0, H - 1)) * u.w + x;
    aa += tmpP[idx]; ab += tmpIP[idx];
  }
  let inv = 1.0 / f32(2 * u.r + 1);
  for (var y = y0; y < y1; y++) {
    let p = y * u.w + x;
    let q = (aa * inv) * nrm[p].z + (ab * inv);
    let pt = vec2i(i32(x), i32(y));
    textureStore(v0, pt, vec4f(q.x, 0.0, 0.0, 0.0));
    if (u.cnt > 1u) { textureStore(v1, pt, vec4f(q.y, 0.0, 0.0, 0.0)); }
    if (u.cnt > 2u) { textureStore(v2, pt, vec4f(q.z, 0.0, 0.0, 0.0)); }
    if (u.cnt > 3u) { textureStore(v3, pt, vec4f(q.w, 0.0, 0.0, 0.0)); }
    let iA = u32(min(i32(y) + u.r + 1, H - 1)) * u.w + x;
    let iS = u32(max(i32(y) - u.r, 0)) * u.w + x;
    aa += tmpP[iA] - tmpP[iS]; ab += tmpIP[iA] - tmpIP[iS];
  }
}`,
  },
  // Peak per pixel over the (aggregated) volume: argmax, then a diopter-space trapezoidal
  // quadrature centroid over +-R frames (eliminates bias from non-uniform diopter spacing and
  // lag-clamped frames at sweep start), or a 3-point log-Gaussian fit for R = 0.
  peak: {
    layout: ["tex-uf-a", "ro", "ro", "ro", "tex-a", "rw", "rw", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var vol: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> nrm: array<vec4f>;
@group(0) @binding(2) var<storage, read> labels: array<f32>;
@group(0) @binding(3) var<storage, read> xform: array<vec4f>;
@group(0) @binding(4) var col: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> dio: array<f32>;
@group(0) @binding(6) var<storage, read_write> conf: array<f32>;
@group(0) @binding(7) var<storage, read_write> rgb: array<u32>;
struct U { w: u32, h: u32, n: u32, r: u32, cw: u32, ch: u32, sharp: u32 }
@group(0) @binding(8) var<uniform> u: U;
fn V(x: u32, y: u32, k: u32) -> f32 { return textureLoad(vol, vec2u(x, y), k, 0).x; }
${COLOUR_AT}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.w || id.y >= u.h) { return; }
  let p = id.y * u.w + id.x;
  var best = -3.4e38; var k = 0u;
  for (var j = 0u; j < u.n; j++) { let v = V(id.x, id.y, j); if (v > best) { best = v; k = j; } }
  var d = labels[k];
  var t = f32(k);
  if (u.r == 0u) {
    if (k > 0u && k < u.n - 1u) {
      let l = log(V(id.x, id.y, k - 1u) + 1e-6); let c = log(best + 1e-6); let r = log(V(id.x, id.y, k + 1u) + 1e-6);
      let den = l - 2.0 * c + r;
      if (den < 0.0) { t += clamp(0.5 * (l - r) / den, -0.5, 0.5); }
    }
    let i0 = min(u32(max(t, 0.0)), u.n - 1u); let i1 = min(i0 + 1u, u.n - 1u); let f = t - f32(i0);
    d = labels[i0] * (1.0 - f) + labels[i1] * f;
  } else {
    let lo = select(0u, k - u.r, k >= u.r); let hi = min(k + u.r, u.n - 1u);
    var base = 3.4e38;
    for (var j = lo; j <= hi; j++) { base = min(base, V(id.x, id.y, j)); }
    var sw = 0.0; var sd = 0.0; var sk = 0.0;
    for (var j = lo; j <= hi; j++) {
      let jL = select(0u, j - 1u, j > 0u); let jR = min(j + 1u, u.n - 1u);
      let dd = 0.5 * abs(labels[jR] - labels[jL]);
      let w = max(V(id.x, id.y, j) - base, 0.0) * dd;
      sw += w; sd += w * labels[j]; sk += w * f32(j);
    }
    if (sw > 0.0) { d = sd / sw; t = sk / sw; }
  }
  dio[p] = d;
  conf[p] = nrm[p].w;
  if (u.sharp == 0u) { rgb[p] = colourAt(id.x, id.y, t); }
}`,
  },
  // Separable colour-guided weighted median filter (5-tap at full res, 3-tap at half res) guided
  // by full-res RGB aif: sharpens occlusion ramps into crisp jumps without inventing mid-air
  // intermediate depths and heals isolated surface speckles.
  wmfH: {
    layout: ["ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aif: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct U { w: u32, h: u32, inv2sig2: f32, r: u32 }
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> shD: array<f32, 136>;
var<workgroup> shC: array<vec3f, 136>;
@compute @workgroup_size(128, 1) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let y = wg.y; if (y >= u.h) { return; }
  let r = i32(min(u.r, 2u)); let W = i32(u.w); let x0 = i32(wg.x * 128u);
  for (var t = i32(lid.x); t < 128 + 2 * r; t += 128) {
    let idx = y * u.w + u32(clamp(x0 - r + t, 0, W - 1));
    shD[u32(t)] = src[idx];
    shC[u32(t)] = unpack4x8unorm(bitcast<u32>(aif[idx].y)).rgb;
  }
  workgroupBarrier();
  let gx = wg.x * 128u + lid.x; if (gx >= u.w) { return; }
  let c = i32(lid.x) + r; let c0 = shC[u32(c)];
  var ds: array<f32, 5>; var ws: array<f32, 5>; var n = 0; var tot = 0.0;
  for (var d = -r; d <= r; d++) {
    let dc = shC[u32(c + d)] - c0;
    let w = exp(-dot(dc, dc) * u.inv2sig2);
    ds[n] = shD[u32(c + d)]; ws[n] = w; tot += w; n++;
  }
  for (var i = 1; i < n; i++) {
    let kd = ds[i]; let kw = ws[i]; var j = i - 1;
    while (j >= 0 && ds[j] > kd) { ds[j + 1] = ds[j]; ws[j + 1] = ws[j]; j--; }
    ds[j + 1] = kd; ws[j + 1] = kw;
  }
  let half = 0.5 * tot; var acc = 0.0; var out = ds[n >> 1];
  for (var i = 0; i < n; i++) { acc += ws[i]; if (acc >= half) { out = ds[i]; break; } }
  dst[y * u.w + gx] = out;
}`,
  },
  wmfV: {
    layout: ["ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aif: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct U { w: u32, h: u32, inv2sig2: f32, r: u32 }
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> shD: array<f32, 320>;
var<workgroup> shC: array<vec3f, 320>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = i32(min(u.r, 2u)); let W = i32(u.w); let H = i32(u.h);
  let bx = i32(wg.x * 16u); let by = i32(wg.y * 16u) - r;
  let total = u32(16 * (16 + 2 * r));
  for (var t = li; t < total; t += 256u) {
    let sx = u32(clamp(bx + i32(t & 15u), 0, W - 1)); let sy = u32(clamp(by + i32(t >> 4u), 0, H - 1));
    let idx = sy * u.w + sx;
    shD[t] = src[idx];
    shC[t] = unpack4x8unorm(bitcast<u32>(aif[idx].y)).rgb;
  }
  workgroupBarrier();
  let gx = wg.x * 16u + lid.x; let gy = wg.y * 16u + lid.y;
  if (gx >= u.w || gy >= u.h) { return; }
  let cy = i32(lid.y) + r; let lx = lid.x; let c0 = shC[u32(cy) * 16u + lx];
  var ds: array<f32, 5>; var ws: array<f32, 5>; var n = 0; var tot = 0.0;
  for (var d = -r; d <= r; d++) {
    let t = u32(cy + d) * 16u + lx;
    let dc = shC[t] - c0;
    let w = exp(-dot(dc, dc) * u.inv2sig2);
    ds[n] = shD[t]; ws[n] = w; tot += w; n++;
  }
  for (var i = 1; i < n; i++) {
    let kd = ds[i]; let kw = ws[i]; var j = i - 1;
    while (j >= 0 && ds[j] > kd) { ds[j + 1] = ds[j]; ws[j + 1] = ws[j]; j--; }
    ds[j + 1] = kd; ws[j + 1] = kw;
  }
  let half = 0.5 * tot; var acc = 0.0; var out = ds[n >> 1];
  for (var i = 0; i < n; i++) { acc += ws[i]; if (acc >= half) { out = ds[i]; break; } }
  dst[gy * u.w + gx] = out;
}`,
  },
  // Pack 4-byte RGBA words into tightly packed 3-byte RGB stream on GPU (25% smaller readback,
  // zero JS RGBA->RGB loop).
  packRGB3: {
    layout: ["ro", "ro", "rw", "uni"],
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> aif: array<vec2f>;
@group(0) @binding(1) var<storage, read> rgbIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> rgb3: array<u32>;
struct U { nWords: u32, N: u32, sharp: u32 }
@group(0) @binding(3) var<uniform> u: U;
fn getByte(bi: u32) -> u32 {
  let p = bi / 3u; if (p >= u.N) { return 0u; }
  let pk = select(rgbIn[p], bitcast<u32>(aif[p].y), u.sharp != 0u);
  return (pk >> ((bi - p * 3u) * 8u)) & 255u;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  let wi = id.x; if (wi >= u.nWords) { return; }
  let b0 = wi * 4u;
  rgb3[wi] = getByte(b0) | (getByte(b0 + 1u) << 8u) | (getByte(b0 + 2u) << 16u) | (getByte(b0 + 3u) << 24u);
}`,
  },
};

// ------------------------------------------------------------------------------------ engine
const LAYOUT = {
  tex: { texture: { sampleType: "float" } },
  "tex-ext": { externalTexture: {} },
  samp: { sampler: { type: "filtering" } },
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
    pipes[name] = {
      layout,
      pipeline: device.createComputePipeline({
        label: name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: "main" },
      }),
    };
  }
  const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const dummyCol = device.createTexture({ size: [1, 1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
  const dummyColView = dummyCol.createView({ dimension: "2d-array" });
  const shared = { linearSampler, dummyColView, stage: null, stageSize: 0, stageBusy: false };
  return { device, info: adapter.info, begin: (opts) => new Session(device, pipes, shared, { ...DEFAULTS, ...opts }) };
}

const U = GPUBufferUsage;

class Session {
  constructor(device, pipes, shared, o) {
    Object.assign(this, { device, pipes, shared, o, n: 0, pushMs: 0, cap: 0, xfCap: 0, res: [], noExt: false });
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

  bg(name, resources) {
    const { layout } = this.pipes[name];
    return this.device.createBindGroup({
      layout,
      entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    });
  }

  dispatch(pass, name, group, x, y = 1) {
    pass.setPipeline(this.pipes[name].pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
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
    const needA0 = !o.sharp || B > 1;
    Object.assign(this, { W, H, B, w, h, qw, qh, N, cw: Math.floor(W / 2), ch: Math.floor(H / 2), qscale: 4 / B, needA0 });
    this.m0 = this.buf(N * 4);
    this.a0 = this.buf(needA0 ? N * 4 : 16);
    this.tmp1 = this.buf(N * 4);
    this.q = this.buf(qw * qh * 4); this.qt = this.buf(qw * qh * 4);
    this.st = [this.buf(qw * qh * 4), this.buf(qw * qh * 4)];
    this.stats = this.buf(N * 16); this.bestk = this.buf(N * 4); this.decided = this.buf(N * 4);
    this.aif = this.buf(o.sharp ? N * 8 : 16);
    this.rel = this.buf(16); this.gnAcc = this.buf(64);
    this.xfCap = 1024;
    this.xform = this.buf(this.xfCap * 16);
    this.labels = this.buf(this.xfCap * 4);
    this.out = null; // lazily allocated only if takeDecided() is called
    const r = Math.max(1, o.win >> o.halvings), m = 0.12;
    this.winR = r;
    this.agg = Math.round(o.aggR * scale);
    this.u = {
      measureQ1: this.uni("uuuuu", [w, h, qw, qh, needA0 ? 1 : 0]),
      measure: this.uni("uuuuu", [w, h, W, H, B]),
      quarter: this.uni("uu", [qw, qh]),
      blurX: this.uni("uuii", [qw, qh, 1, 0]), blurY: this.uni("uuii", [qw, qh, 0, 1]),
      grad: this.uni("uu", [qw, qh]),
      gnAccum: this.uni("uuuuuu", [qw, qh, Math.floor(qw * m), Math.floor(qw * (1 - m)), Math.floor(qh * m), Math.floor(qh * (1 - m))]),
      gn0: this.uni("uu", [0, 0]),
      gn1: this.uni("uu", [0, 1]),
      frame: this.uni("uuiufu", [w, h, r, 0, this.qscale, o.sharp ? 1 : 0]),
      sharpest: this.uni("uuuuuuf", [w, h, W, H, 0, B, this.qscale]),
      box: this.uni("uuif", [w, h, this.agg, o.aggEps]),
    };
    // Static bind groups reused across every push() call.
    this.g = {
      blurX: this.bg("blur13", [this.q, this.qt, this.u.blurX]),
      blurY: this.bg("blur13", [this.qt, this.q, this.u.blurY]),
      grad: [this.bg("grad", [this.q, this.st[0], this.u.grad]), this.bg("grad", [this.q, this.st[1], this.u.grad])],
      gnAccum: [
        this.bg("gnAccum", [this.st[1], this.st[0], this.rel, this.gnAcc, this.u.gnAccum]), // k even: prev=1, cur=0
        this.bg("gnAccum", [this.st[0], this.st[1], this.rel, this.gnAcc, this.u.gnAccum]), // k odd:  prev=0, cur=1
      ],
      gn0: this.bg("gnStep", [this.gnAcc, this.rel, this.xform, this.u.gn0]),
      gn1: this.bg("gnStep", [this.gnAcc, this.rel, this.xform, this.u.gn1]),
      rescaleH: this.bg("rescaleH", [this.m0, this.xform, this.tmp1, this.u.frame]),
    };
    this.grow(40);
  }

  ensureSrc2d() {
    if (!this.src) {
      this.src = this.device.createTexture({
        size: [this.W, this.H], format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.srcView = this.src.createView();
    }
    return this.srcView;
  }

  /** Room for at least n frames in the focus volume (and colour array if sharp = 0). */
  grow(n) {
    if (n <= this.cap) return;
    // Round capacity up to a multiple of 4 so 4-slice guided-filter batches always bind 4 distinct layers.
    const cap = Math.ceil(Math.max(n, Math.ceil(this.cap * 1.5)) / 4) * 4;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const vol = this.device.createTexture({ size: [this.w, this.h, cap], format: "r32float", usage });
    const col = this.o.sharp ? null : this.device.createTexture({ size: [this.cw, this.ch, cap], format: "rgba8unorm", usage });
    if (this.cap) {
      const enc = this.device.createCommandEncoder();
      enc.copyTextureToTexture({ texture: this.vol }, { texture: vol }, [this.w, this.h, this.n]);
      if (col) enc.copyTextureToTexture({ texture: this.col }, { texture: col }, [this.cw, this.ch, this.n]);
      this.device.queue.submit([enc.finish()]);
      this.vol.destroy(); this.col?.destroy();
    }
    const volLayers = Array.from({ length: cap }, (_, k) => vol.createView({ dimension: "2d", baseArrayLayer: k, arrayLayerCount: 1 }));
    Object.assign(this, { vol, col, cap, volLayers });
    if (cap > this.xfCap) {
      const newCap = cap;
      const xform = this.buf(newCap * 16), labels = this.buf(newCap * 4);
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.xform, 0, xform, 0, this.n * 16);
      this.device.queue.submit([enc.finish()]);
      Object.assign(this, { xform, labels, xfCap: newCap });
      this.g.gn0 = this.bg("gnStep", [this.gnAcc, this.rel, this.xform, this.u.gn0]);
      this.g.gn1 = this.bg("gnStep", [this.gnAcc, this.rel, this.xform, this.u.gn1]);
      this.g.rescaleH = this.bg("rescaleH", [this.m0, this.xform, this.tmp1, this.u.frame]);
    }
  }

  /** Align and focus-measure one frame (ImageBitmap, VideoFrame, canvas...). Returns the count. */
  async push(image) {
    const t0 = performance.now();
    const W = image.width ?? image.displayWidth, H = image.height ?? image.displayHeight;
    if (!this.m0) this.alloc(W, H);
    const k = this.n, d = this.device;
    this.grow(k + 1);

    // Zero-copy external texture import via VideoFrame avoids copyExternalImageToTexture stall;
    // falls back transparently to a 2D texture upload if VideoFrame is unavailable.
    let srcRes = null, vf = null, sfx = "";
    if (!this.noExt && typeof VideoFrame === "function") {
      try {
        if (image instanceof VideoFrame) {
          srcRes = d.importExternalTexture({ source: image });
        } else {
          vf = new VideoFrame(image, { timestamp: k * 33333 });
          srcRes = d.importExternalTexture({ source: vf });
        }
      } catch {
        vf?.close(); vf = null;
        this.noExt = true;
      }
    }
    if (!srcRes) {
      sfx = "2d";
      srcRes = this.ensureSrc2d();
      d.queue.copyExternalImageToTexture({ source: image }, { texture: this.src }, [this.W, this.H]);
    }

    this.uni("uu", [k, 0], this.u.gn0);
    this.uni("uu", [k, 1], this.u.gn1);
    this.uni("uuiufu", [this.w, this.h, this.winR, k, this.qscale, this.o.sharp ? 1 : 0], this.u.frame);
    if (this.o.sharp) this.uni("uuuuuuf", [this.w, this.h, this.W, this.H, k, this.B, this.qscale], this.u.sharpest);

    const g16 = (v) => Math.ceil(v / 16);
    const gV = Math.ceil((Math.ceil(this.h / SEG) * this.w) / 64);
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();

    if (this.B === 1) {
      const mName = "measureQ1" + sfx;
      this.dispatch(pass, mName, this.bg(mName, [srcRes, this.m0, this.a0, this.q, this.u.measureQ1]), g16(this.w), g16(this.h));
    } else {
      const mName = "measure" + sfx, qName = "quarter" + sfx;
      this.dispatch(pass, mName, this.bg(mName, [srcRes, this.m0, this.a0, this.u.measure]), g16(this.w), g16(this.h));
      this.dispatch(pass, qName, this.bg(qName, [srcRes, this.q, this.u.quarter]), g16(this.qw), g16(this.qh));
    }
    if (!this.o.sharp) {
      const cName = "colour" + sfx;
      this.dispatch(pass, cName, this.bg(cName, [srcRes, this.col.createView({ dimension: "2d", baseArrayLayer: k, arrayLayerCount: 1 })]), g16(this.cw), g16(this.ch));
    }
    this.dispatch(pass, "blur13", this.g.blurX, g16(this.qw), g16(this.qh));
    this.dispatch(pass, "blur13", this.g.blurY, g16(this.qw), g16(this.qh));
    this.dispatch(pass, "grad", this.g.grad[k & 1], g16(this.qw), g16(this.qh));
    if (k === 0) {
      this.dispatch(pass, "gnStep", this.g.gn1, 1);
    } else {
      const gAcc = this.g.gnAccum[k & 1];
      this.dispatch(pass, "gnAccum", gAcc, 1);
      this.dispatch(pass, "gnStep", this.g.gn0, 1);
      this.dispatch(pass, "gnAccum", gAcc, 1);
      this.dispatch(pass, "gnStep", this.g.gn1, 1);
    }
    this.dispatch(pass, "rescaleH", this.g.rescaleH, Math.ceil(this.w / 128), this.h);
    this.dispatch(pass, "storeV", this.bg("storeV", [this.tmp1, this.a0, this.xform, this.volLayers[k], this.stats, this.bestk, this.u.frame]), gV);
    if (this.o.sharp) {
      const sName = "sharpest" + sfx;
      this.dispatch(pass, sName, this.bg(sName, [srcRes, this.shared.linearSampler, this.m0, this.xform, this.stats, this.aif, this.u.sharpest]), g16(this.w), g16(this.h));
    }
    pass.end();
    d.queue.submit([enc.finish()]);
    vf?.close();
    this.n++;
    this.pushMs += performance.now() - t0;
    return this.n;
  }

  /** Pixels decided so far (each returned once), with depth from `labels` (1/m per frame so far)
   *  and all-in-focus colour: { w, h, px, z, rgb }. */
  async takeDecided(labels, r = 3, drop = 0.6, minConf = 2.0) {
    const empty = { w: this.w, h: this.h, px: new Uint32Array(0), z: new Float32Array(0), rgb: new Uint8Array(0) };
    if (this.n <= r) return empty;
    const d = this.device;
    this.out ??= this.buf(16 + this.N * 16);
    d.queue.writeBuffer(this.labels, 0, new Float32Array(labels));
    const u = this.uni("uuuuffuuu", [this.w, this.h, this.n, r, drop, minConf, this.cw, this.ch, this.o.sharp ? 1 : 0]);
    const colView = this.col ? this.col.createView({ dimension: "2d-array" }) : this.shared.dummyColView;
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.out, 0, 16);
    const pass = enc.beginComputePass();
    this.dispatch(
      pass, "decide",
      this.bg("decide", [this.vol.createView({ dimension: "2d-array" }), this.stats, this.bestk, this.decided, this.labels, this.xform, colView, this.aif, this.out, u]),
      Math.ceil(this.w / 16), Math.ceil(this.h / 16)
    );
    pass.end();
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
    const t0 = performance.now(), d = this.device, n = this.n, { w, h, N } = this;
    let tempStage = null;
    const sh = this.shared;
    try {
      d.queue.writeBuffer(this.labels, 0, new Float32Array(diopters));
      // Release early-points output buffer (33 MB at 1080p) before guided-filter scratch allocation,
      // and reuse bestk/decided (no longer needed in finish) as dio/conf outputs.
      this.out?.destroy(); this.out = null;
      const nrm = this.buf(N * 16);
      const dio = this.bestk, conf = this.decided;
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      const g16 = (v) => Math.ceil(v / 16);
      const gH = Math.ceil((Math.ceil(w / SEG) * h) / 64), gV = Math.ceil((Math.ceil(h / SEG) * w) / 64);
      const volArr = this.vol.createView({ dimension: "2d-array" });
      const colArr = this.col ? this.col.createView({ dimension: "2d-array" }) : sh.dummyColView;
      const smR = this.agg > 0 && this.o.sharp ? Math.max(1, 2 >> this.o.halvings) : 0;

      this.dispatch(pass, "normSmooth", this.bg("normSmooth", [this.stats, nrm, this.uni("uuuuu", [w, h, n, this.o.norm ? 1 : 0, smR])]), g16(w), g16(h));

      if (this.agg > 0) {
        const gst = this.buf(N * 8);
        const tmpP = this.buf(N * 16), tmpIP = this.buf(N * 16), abA = this.buf(N * 16), abB = this.buf(N * 16);
        const gVA = this.bg("boxVA4", [tmpP, tmpIP, gst, abA, abB, this.u.box]);
        const gHQ = this.bg("boxHQ4", [abA, abB, tmpP, tmpIP, this.u.box]);
        const gx128 = Math.ceil(w / 128);
        // Reuse tmpP (N*16 >= N*8) as scratch for guide box sums.
        this.dispatch(pass, "boxHG", this.bg("boxHG", [nrm, tmpP, this.u.box]), gH);
        this.dispatch(pass, "boxVG", this.bg("boxVG", [tmpP, gst, this.u.box]), gV);
        for (let k = 0; k < n; k += 4) {
          const cnt = Math.min(4, n - k);
          this.dispatch(pass, "boxHA4", this.bg("boxHA4", [volArr, nrm, tmpP, tmpIP, this.uni("uuiuu", [w, h, this.agg, k, n])]), gx128, h);
          this.dispatch(pass, "boxVA4", gVA, gV);
          this.dispatch(pass, "boxHQ4", gHQ, gx128, h);
          this.dispatch(
            pass, "boxVQ4",
            this.bg("boxVQ4", [tmpP, tmpIP, nrm, this.volLayers[k], this.volLayers[k + 1], this.volLayers[k + 2], this.volLayers[k + 3], this.uni("uuiu", [w, h, this.agg, cnt])]),
            gV
          );
        }
      }

      // Peak writes into dio, conf (and into tmp1 for packed RGB when sharp = 0).
      const uPeak = this.uni("uuuuuuu", [w, h, n, this.o.peakR, this.cw, this.ch, this.o.sharp ? 1 : 0]);
      this.dispatch(pass, "peak", this.bg("peak", [volArr, nrm, this.labels, this.xform, colArr, dio, conf, this.tmp1, uPeak]), g16(w), g16(h));

      // Separable colour-guided weighted median refinement (reuses tmp1 as ping-pong scratch).
      if (this.agg > 0 && this.o.sharp) {
        const wmfR = Math.max(1, 2 >> this.o.halvings);
        const sigC = 0.065;
        const uWmf = this.uni("uufu", [w, h, 1 / (2 * sigC * sigC), wmfR]);
        this.dispatch(pass, "wmfH", this.bg("wmfH", [dio, this.aif, this.tmp1, uWmf]), Math.ceil(w / 128), h);
        this.dispatch(pass, "wmfV", this.bg("wmfV", [this.tmp1, this.aif, dio, uWmf]), g16(w), g16(h));
      }

      // Pack 3-byte RGB stream into m0 (free during finish, N*4 >= ceil(N*3/4)*4).
      const rgbWords = Math.ceil((N * 3) / 4);
      const uPack = this.uni("uuu", [rgbWords, N, this.o.sharp ? 1 : 0]);
      this.dispatch(pass, "packRGB3", this.bg("packRGB3", [this.aif, this.tmp1, this.m0, uPack]), Math.ceil(rgbWords / 256));
      pass.end();

      // Single staging readback for diopter, conf, packed RGB, and alignment scales.
      const oConf = N * 4, oRgb = N * 8, oXf = oRgb + rgbWords * 4, totalBytes = oXf + n * 16;
      let stage;
      if (!sh.stageBusy) {
        sh.stageBusy = true;
        if (!sh.stage || sh.stageSize < totalBytes) {
          sh.stage?.destroy();
          sh.stage = d.createBuffer({ size: totalBytes, usage: U.MAP_READ | U.COPY_DST });
          sh.stageSize = totalBytes;
        }
        stage = sh.stage;
      } else {
        tempStage = d.createBuffer({ size: totalBytes, usage: U.MAP_READ | U.COPY_DST });
        stage = tempStage;
      }
      enc.copyBufferToBuffer(dio, 0, stage, 0, N * 4);
      enc.copyBufferToBuffer(conf, 0, stage, oConf, N * 4);
      enc.copyBufferToBuffer(this.m0, 0, stage, oRgb, rgbWords * 4);
      enc.copyBufferToBuffer(this.xform, 0, stage, oXf, n * 16);
      d.queue.submit([enc.finish()]);

      await stage.mapAsync(GPUMapMode.READ);
      const map = stage.getMappedRange();
      // Distinct ArrayBuffers so dff-worker.js can transfer [diopter, conf, colours] independently.
      const diopter = new Float32Array(N); diopter.set(new Float32Array(map, 0, N));
      const confOut = new Float32Array(N); confOut.set(new Float32Array(map, oConf, N));
      const colours = new Uint8Array(N * 3); colours.set(new Uint8Array(map, oRgb, N * 3));
      const xf = new Float32Array(map, oXf, n * 4);
      const scales = new Float32Array(n);
      for (let k = 0; k < n; k++) scales[k] = xf[k * 4];
      stage.unmap();
      if (!tempStage) sh.stageBusy = false;

      return {
        w, h, diopter, diopterSmooth: diopter, conf: confOut, colours, scales,
        pushMs: this.pushMs, finishMs: performance.now() - t0,
      };
    } finally {
      if (tempStage) tempStage.destroy();
      else if (sh.stageBusy) { try { sh.stage?.unmap(); } catch {} sh.stageBusy = false; }
      this.free();
    }
  }

  free() {
    for (const b of this.res) b.destroy();
    this.res = [];
    this.src?.destroy(); this.src = null;
    this.vol?.destroy(); this.vol = null;
    this.col?.destroy(); this.col = null;
  }
}
