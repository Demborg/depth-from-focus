//! Depth-from-focus core. Pure Rust, no I/O, no dependencies, so it compiles to wasm unchanged.
//!
//! Pipeline: grayscale frames (one per focus setting) -> breathing alignment (per-frame scale
//! about the image center) -> per-pixel focus measure (windowed sum-modified-Laplacian) ->
//! per-pixel argmax over the stack with Gaussian sub-frame interpolation -> depth index.

#[derive(Clone, Debug)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    /// From interleaved RGB(A) bytes, Rec.601 luma in [0,1].
    pub fn from_rgb(w: usize, h: usize, px: &[u8], channels: usize) -> Self {
        let data = px
            .chunks_exact(channels)
            .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
            .collect();
        Self { w, h, data }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    /// Bilinear sample with edge clamping.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let a = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let b = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// 2x2 box downsample.
    pub fn half(&self) -> Gray {
        let (w, h) = (self.w / 2, self.h / 2);
        let mut o = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                o.data[y * w + x] = 0.25
                    * (self.at(2 * x, 2 * y)
                        + self.at(2 * x + 1, 2 * y)
                        + self.at(2 * x, 2 * y + 1)
                        + self.at(2 * x + 1, 2 * y + 1));
            }
        }
        o
    }
}

/// Separable box blur of radius r (window 2r+1), edge-clamped, O(1) per pixel. The vertical pass
/// keeps running column sums so it streams through memory row by row.
pub fn box_blur(img: &Gray, r: usize) -> Gray {
    if r == 0 {
        return img.clone();
    }
    let (w, h) = (img.w, img.h);
    let norm = 1.0 / (2 * r + 1) as f32;
    // horizontal
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = &img.data[y * w..(y + 1) * w];
        let out = &mut tmp[y * w..(y + 1) * w];
        let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
        let mut acc: f32 = (-(r as isize)..=r as isize).map(at).sum();
        for x in 0..w {
            out[x] = acc * norm;
            acc += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
        }
    }
    // vertical, running sums over rows
    let rowi = |y: isize| y.clamp(0, h as isize - 1) as usize;
    let mut acc = vec![0.0f32; w];
    for dy in -(r as isize)..=r as isize {
        let ry = rowi(dy);
        for x in 0..w {
            acc[x] += tmp[ry * w + x];
        }
    }
    let mut out = Gray::new(w, h);
    for y in 0..h {
        let (add, sub) = (rowi(y as isize + r as isize + 1), rowi(y as isize - r as isize));
        let o = &mut out.data[y * w..(y + 1) * w];
        for x in 0..w {
            o[x] = acc[x] * norm;
            acc[x] += tmp[add * w + x] - tmp[sub * w + x];
        }
    }
    out
}

/// Approximate Gaussian (three box passes).
pub fn blur(img: &Gray, r: usize) -> Gray {
    box_blur(&box_blur(&box_blur(img, r), r), r)
}

/// Resample `img` so that it lines up with a reference frame whose magnification is 1:
/// out(p) = img(c + (p - c) * s). See `align_scales` for how s is chosen.
pub fn rescale_about_center(img: &Gray, s: f32) -> Gray {
    let (cx, cy) = ((img.w as f32 - 1.0) * 0.5, (img.h as f32 - 1.0) * 0.5);
    let mut o = Gray::new(img.w, img.h);
    for y in 0..img.h {
        let sy = cy + (y as f32 - cy) * s;
        for x in 0..img.w {
            o.data[y * img.w + x] = img.sample(cx + (x as f32 - cx) * s, sy);
        }
    }
    o
}

/// Blur-tolerant structure image used for alignment: gradient magnitude of a blurred image.
pub fn structure(img: &Gray, r: usize) -> Gray {
    let b = blur(img, r);
    let mut o = Gray::new(b.w, b.h);
    for y in 1..b.h - 1 {
        for x in 1..b.w - 1 {
            let gx = b.at(x + 1, y) - b.at(x - 1, y);
            let gy = b.at(x, y + 1) - b.at(x, y - 1);
            o.data[y * b.w + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    o
}

/// Normalized cross-correlation between `a` and `b` rescaled by s, over the central region.
fn ncc_at_scale(a: &Gray, b: &Gray, s: f32, margin: f32) -> f32 {
    let (cx, cy) = ((a.w as f32 - 1.0) * 0.5, (a.h as f32 - 1.0) * 0.5);
    let (x0, x1) = ((a.w as f32 * margin) as usize, (a.w as f32 * (1.0 - margin)) as usize);
    let (y0, y1) = ((a.h as f32 * margin) as usize, (a.h as f32 * (1.0 - margin)) as usize);
    let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in (y0..y1).step_by(2) {
        let by = cy + (y as f32 - cy) * s;
        for x in x0..x1 {
            let va = a.at(x, y) as f64;
            let vb = b.sample(cx + (x as f32 - cx) * s, by) as f64;
            sa += va;
            sb += vb;
            saa += va * va;
            sbb += vb * vb;
            sab += va * vb;
            n += 1.0;
        }
    }
    let cov = sab / n - (sa / n) * (sb / n);
    let va = saa / n - (sa / n).powi(2);
    let vb = sbb / n - (sb / n).powi(2);
    (cov / (va * vb).sqrt().max(1e-12)) as f32
}

/// Find s maximizing NCC(a(p), b(c + (p-c)s)) over [center-range, center+range], with parabolic refinement.
pub fn estimate_relative_scale(a: &Gray, b: &Gray, center: f32, range: f32, steps: usize) -> (f32, f32) {
    let mut scores = Vec::with_capacity(steps + 1);
    let at = |i: usize| center - range + 2.0 * range * i as f32 / steps as f32;
    for i in 0..=steps {
        scores.push(ncc_at_scale(a, b, at(i), 0.12));
    }
    let (k, &best) = scores.iter().enumerate().fold((0, &f32::MIN), |acc, (i, v)| if *v > *acc.1 { (i, v) } else { acc });
    let mut s = at(k);
    if k > 0 && k < steps {
        let (l, c, r) = (scores[k - 1], scores[k], scores[k + 1]);
        let den = l - 2.0 * c + r;
        if den < 0.0 {
            s += 0.5 * (l - r) / den * 2.0 * range / steps as f32;
        }
    }
    (s, best)
}

/// Adjacent frames of a focus sweep differ by ~0.02% in magnification, so search +-0.4% in
/// 0.04% steps; fall back to a wide coarse-to-fine search if the best lands at the edge.
pub fn estimate_relative_scale_fast(a: &Gray, b: &Gray) -> (f32, f32) {
    let (s, v) = estimate_relative_scale(a, b, 1.0, 0.004, 20);
    if (s - 1.0).abs() < 0.0038 {
        return (s, v);
    }
    let (s0, _) = estimate_relative_scale(a, b, 1.0, 0.03, 24);
    estimate_relative_scale(a, b, s0, 0.003, 20)
}

/// Per-frame magnification relative to the widest-field frame (all values <= 1), estimated by
/// chaining adjacent-frame scale estimates on blur-tolerant structure images.
/// Aligned frame i = rescale_about_center(frame_i, scales[i]).
pub fn align_scales(frames: &[Gray]) -> (Vec<f32>, Vec<f32>) {
    let st: Vec<Gray> = frames.iter().map(|f| structure(f, 2)).collect();
    let mut cum = vec![1.0f32];
    let mut ncc = vec![1.0f32];
    for i in 1..frames.len() {
        // find s with frame_i(c + (p-c)s) ~ frame_{i-1}(p)
        let (s, v) = estimate_relative_scale_fast(&st[i - 1], &st[i]);
        cum.push(cum[i - 1] * s);
        ncc.push(v);
    }
    // frame_i(c + (p-c)*cum[i]) ~ frame_0(p). Re-reference to the frame with the largest factor
    // (widest field of view) so every factor is <= 1 and aligned frames sample inside bounds.
    let max = cum.iter().cloned().fold(f32::MIN, f32::max);
    (cum.iter().map(|c| c / max).collect(), ncc)
}

/// Modified Laplacian |2I - I(x-1) - I(x+1)| + |2I - I(y-1) - I(y+1)| per pixel (no window).
pub fn modified_laplacian(img: &Gray) -> Gray {
    let (w, h) = (img.w, img.h);
    let mut ml = Gray::new(w, h);
    for y in 1..h - 1 {
        let (up, row, dn) = (&img.data[(y - 1) * w..y * w], &img.data[y * w..(y + 1) * w], &img.data[(y + 1) * w..(y + 2) * w]);
        let o = &mut ml.data[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let c = 2.0 * row[x];
            o[x] = (c - row[x - 1] - row[x + 1]).abs() + (c - up[x] - dn[x]).abs();
        }
    }
    ml
}

/// Windowed sum-modified-Laplacian focus measure (Nayar & Nakagawa), window radius r.
pub fn sml(img: &Gray, r: usize) -> Gray {
    let (w, h) = (img.w, img.h);
    let mut ml = Gray::new(w, h);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = 2.0 * img.at(x, y);
            let lx = (c - img.at(x - 1, y) - img.at(x + 1, y)).abs();
            let ly = (c - img.at(x, y - 1) - img.at(x, y + 1)).abs();
            ml.data[y * w + x] = lx + ly;
        }
    }
    box_blur(&ml, r)
}

/// Ring Difference Filter (Surh et al., CVPR 2017), square-window variant computed with box
/// sums: |mean over the inner box (radius r_in) - mean over the ring between radii r1..=r2|,
/// then summed over a (2*win+1)^2 window. The gap between r_in and r1 makes it less sensitive
/// to pixel noise than Laplacian-based measures.
pub fn rdf(img: &Gray, r_in: usize, r1: usize, r2: usize, win: usize) -> Gray {
    let area = |r: usize| ((2 * r + 1) * (2 * r + 1)) as f32;
    let inner = box_blur(img, r_in);
    let outer = box_blur(img, r2);
    let hole = box_blur(img, r1 - 1);
    let ring_area = area(r2) - area(r1 - 1);
    let mut o = Gray::new(img.w, img.h);
    for i in 0..o.data.len() {
        let ring = (outer.data[i] * area(r2) - hole.data[i] * area(r1 - 1)) / ring_area;
        o.data[i] = (inner.data[i] - ring).abs();
    }
    box_blur(&o, win)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measure {
    /// Sum-modified-Laplacian (Nayar & Nakagawa).
    Sml,
    /// Ring Difference Filter (Surh et al.): inner radius, ring from r1 to r2.
    Rdf(usize, usize, usize),
}

pub fn focus_measure(img: &Gray, m: Measure, win: usize) -> Gray {
    match m {
        Measure::Sml => sml(img, win),
        Measure::Rdf(r_in, r1, r2) => rdf(img, r_in, r1, r2, win),
    }
}

/// Precomputed guide statistics for the guided filter (He et al.), shared by every focus slice.
pub struct Guide {
    pub img: Gray,
    mean: Gray,
    var: Gray,
    r: usize,
    eps: f32,
}

impl Guide {
    pub fn new(img: Gray, r: usize, eps: f32) -> Self {
        let mean = box_blur(&img, r);
        let sq = Gray { w: img.w, h: img.h, data: img.data.iter().map(|v| v * v).collect() };
        let mut var = box_blur(&sq, r);
        for (v, m) in var.data.iter_mut().zip(&mean.data) {
            *v -= m * m;
        }
        Self { img, mean, var, r, eps }
    }

    /// Edge-preserving smoothing of `p`, following edges of the guide image.
    pub fn filter(&self, p: &Gray) -> Gray {
        let (w, h) = (p.w, p.h);
        let mean_p = box_blur(p, self.r);
        let ip = Gray { w, h, data: self.img.data.iter().zip(&p.data).map(|(a, b)| a * b).collect() };
        let corr = box_blur(&ip, self.r);
        let mut a = Gray::new(w, h);
        let mut b = Gray::new(w, h);
        for i in 0..w * h {
            let cov = corr.data[i] - self.mean.data[i] * mean_p.data[i];
            a.data[i] = cov / (self.var.data[i] + self.eps);
            b.data[i] = mean_p.data[i] - a.data[i] * self.mean.data[i];
        }
        let (ma, mb) = (box_blur(&a, self.r), box_blur(&b, self.r));
        Gray { w, h, data: (0..w * h).map(|i| ma.data[i] * self.img.data[i] + mb.data[i]).collect() }
    }
}

/// All-in-focus grayscale: for each pixel, the value from the frame with the highest measure.
pub fn all_in_focus(frames: &[Gray], measures: &[Gray]) -> Gray {
    let (w, h) = (measures[0].w, measures[0].h);
    let mut o = Gray::new(w, h);
    let mut best = vec![f32::MIN; w * h];
    for (f, m) in frames.iter().zip(measures) {
        for i in 0..w * h {
            if m.data[i] > best[i] {
                best[i] = m.data[i];
                o.data[i] = f.data[i];
            }
        }
    }
    o
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Peak {
    /// Three-point Gaussian fit around the maximum (Nayar). Prone to snapping to whole frames.
    Gauss3,
    /// Baseline-subtracted centroid over +-R frames around the maximum (a windowed soft-argmax).
    Centroid(usize),
}

pub struct DepthMap {
    pub w: usize,
    pub h: usize,
    /// Fractional frame index of best focus.
    pub index: Vec<f32>,
    /// Peak focus measure.
    pub peak: Vec<f32>,
    /// Peak / mean over the stack (>= 1). Low = textureless or ambiguous.
    pub conf: Vec<f32>,
    /// Width of the focus peak in frames (weighted std for Centroid, 0 for Gauss3).
    pub spread: Vec<f32>,
}

/// Per-pixel argmax over the stack with Gaussian (log-parabola) sub-frame interpolation.
pub fn depth_from_measures(m: &[Gray]) -> DepthMap {
    depth_from_volume(m, Peak::Gauss3)
}

/// Per-pixel best-focus index from a focus volume, with sub-frame refinement.
pub fn depth_from_volume(m: &[Gray], peak: Peak) -> DepthMap {
    let (w, h, n) = (m[0].w, m[0].h, m.len());
    let mut index = vec![0.0; w * h];
    let mut peakv = vec![0.0; w * h];
    let mut conf = vec![0.0; w * h];
    let mut spread = vec![0.0; w * h];
    for p in 0..w * h {
        let mut k = 0;
        let mut best = f32::MIN;
        let mut sum = 0.0;
        for (i, f) in m.iter().enumerate() {
            let v = f.data[p];
            sum += v;
            if v > best {
                best = v;
                k = i;
            }
        }
        let mut idx = k as f32;
        match peak {
            Peak::Gauss3 => {
                if k > 0 && k < n - 1 {
                    let e = 1e-6;
                    let (l, c, r) = ((m[k - 1].data[p] + e).ln(), (best + e).ln(), (m[k + 1].data[p] + e).ln());
                    let den = l - 2.0 * c + r;
                    if den < 0.0 {
                        idx += (0.5 * (l - r) / den).clamp(-0.5, 0.5);
                    }
                }
            }
            Peak::Centroid(rad) => {
                let (lo, hi) = (k.saturating_sub(rad), (k + rad).min(n - 1));
                let base = (lo..=hi).map(|i| m[i].data[p]).fold(f32::MAX, f32::min);
                let (mut sw, mut sk, mut skk) = (0.0f32, 0.0f32, 0.0f32);
                for i in lo..=hi {
                    let wgt = m[i].data[p] - base;
                    sw += wgt;
                    sk += wgt * i as f32;
                    skk += wgt * (i * i) as f32;
                }
                if sw > 0.0 {
                    idx = sk / sw;
                    spread[p] = (skk / sw - idx * idx).max(0.0).sqrt();
                }
            }
        }
        index[p] = idx;
        peakv[p] = best;
        conf[p] = best / (sum / n as f32).max(1e-9);
    }
    DepthMap { w, h, index, peak: peakv, conf, spread }
}

/// Linear interpolation of a per-frame quantity at a fractional index.
pub fn interp_index(values: &[f32], idx: f32) -> f32 {
    let n = values.len();
    let i0 = (idx.floor().max(0.0) as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    let f = idx - i0 as f32;
    values[i0] * (1.0 - f) + values[i1] * f
}

/// Fit z = a x + b y + c robustly (RANSAC + least-squares refit on inliers).
/// For a planar scene, inverse depth (diopters) is exactly affine in image coordinates.
pub fn fit_plane_ransac(pts: &[[f32; 3]], iters: usize, thresh: f32, seed: u64) -> Option<([f64; 3], Vec<bool>)> {
    if pts.len() < 3 {
        return None;
    }
    let mut rng = seed.max(1);
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % pts.len() as u64) as usize
    };
    let mut best: Option<([f64; 3], usize)> = None;
    for _ in 0..iters {
        let s = [pts[next()], pts[next()], pts[next()]];
        if let Some(pl) = lstsq_plane(&s) {
            let cnt = pts.iter().filter(|p| plane_res(&pl, p).abs() < thresh as f64).count();
            if best.map_or(true, |b| cnt > b.1) {
                best = Some((pl, cnt));
            }
        }
    }
    let (pl, _) = best?;
    let inl: Vec<bool> = pts.iter().map(|p| plane_res(&pl, p).abs() < thresh as f64).collect();
    let sel: Vec<[f32; 3]> = pts.iter().zip(&inl).filter(|(_, &b)| b).map(|(p, _)| *p).collect();
    let pl = lstsq_plane(&sel)?;
    let inl = pts.iter().map(|p| plane_res(&pl, p).abs() < thresh as f64).collect();
    Some((pl, inl))
}

pub fn plane_res(pl: &[f64; 3], p: &[f32; 3]) -> f64 {
    p[2] as f64 - (pl[0] * p[0] as f64 + pl[1] * p[1] as f64 + pl[2])
}

fn lstsq_plane(pts: &[[f32; 3]]) -> Option<[f64; 3]> {
    // Normal equations for [x y 1] * [a b c]^T = z
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for p in pts {
        let r = [p[0] as f64, p[1] as f64, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += r[i] * r[j];
            }
            atb[i] += r[i] * p[2] as f64;
        }
    }
    solve3(ata, atb)
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0; 3];
    for k in 0..3 {
        let mut m = a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        out[k] = det(&m) / d;
    }
    Some(out)
}

/// Confidence-weighted smoothing of a depth map: sum(w*d)/sum(w) under an approximate Gaussian
/// (three box passes of radius r), with w = conf^4. Trusts sharp, textured pixels and spreads
/// their depth into weak neighbours, which removes most of the grid-line "lattice" bias.
pub fn smooth_depth(w: usize, h: usize, depth: &[f32], conf: &[f32], r: usize) -> Vec<f32> {
    let wt: Vec<f32> = conf.iter().map(|c| (c * c) * (c * c)).collect();
    let num = blur(&Gray { w, h, data: depth.iter().zip(&wt).map(|(d, w)| d * w).collect() }, r);
    let den = blur(&Gray { w, h, data: wt }, r);
    num.data.iter().zip(&den.data).map(|(n, d)| n / d.max(1e-12)).collect()
}

/// Lens-lag relabelling. `req` = (request time ms, requested diopters) per frame; the lens at grab
/// time t sits where it was asked to be at t - lag (linear between requests, it moves continuously).
pub fn lagged_focus(req: &[(f32, f32)], grab_ms: &[f32], lag_ms: f32) -> Vec<f32> {
    grab_ms
        .iter()
        .map(|&g| {
            let t = g - lag_ms;
            if t <= req[0].0 {
                return req[0].1;
            }
            for w in req.windows(2) {
                if t <= w[1].0 {
                    let f = (t - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
                    return w[0].1 * (1.0 - f) + w[1].1 * f;
                }
            }
            req[req.len() - 1].1
        })
        .collect()
}

/// Default lens lag measured on the Pixel 9 (up vs down sweeps agree at ~271 ms).
pub const PIXEL9_LAG_MS: f32 = 271.0;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Focus-measure window radius in input pixels.
    pub win: usize,
    /// Halvings applied to the focus measure before argmax (1 -> half res).
    pub measure_halvings: usize,
    pub measure: Measure,
    pub peak: Peak,
    /// Guided-filter radius for focus-volume aggregation at measure resolution (0 = off).
    pub agg_r: usize,
    /// Guided-filter regularisation (guide intensities are in [0,1]).
    pub agg_eps: f32,
    /// Normalise each pixel's focus curve (divide by its sum over frames, scaled by confidence^norm_pow
    /// / ...) before aggregation, so every pixel votes with a peak shape instead of the strongest
    /// texture in the window dominating. 0 = off (raw magnitudes).
    pub norm: bool,
    /// Blur radius applied to the guide (all-in-focus) image, so fine texture such as printed
    /// grid lines is not treated as an edge; only larger structures are (0 = off).
    pub guide_blur: usize,
    /// Confidence-weighted depth smoothing radius at measure resolution (0 = off).
    pub smooth_r: usize,
}

impl Default for Params {
    fn default() -> Self {
        // Best on the flat-mat stack (tools/compare.py): normalised curves, guided aggregation,
        // centroid peak, no post-smoothing. Radii are for 1080x1920 input at half-res measure.
        Self { win: 6, measure_halvings: 1, measure: Measure::Sml, peak: Peak::Centroid(6), agg_r: 16, agg_eps: 1e-3, norm: true, guide_blur: 0, smooth_r: 0 }
    }
}

pub struct DepthResult {
    pub w: usize,
    pub h: usize,
    /// Inverse depth (1/m) per pixel, raw and smoothed.
    pub diopter: Vec<f32>,
    pub diopter_smooth: Vec<f32>,
    pub conf: Vec<f32>,
    /// Fractional best-focus frame index.
    pub index: Vec<f32>,
    /// Focus-peak width in frames (0 unless Peak::Centroid).
    pub spread: Vec<f32>,
    /// All-in-focus grayscale at measure resolution (the guide image).
    pub aif: Vec<f32>,
    pub scales: Vec<f32>,
}

/// One frame's contribution: (focus measure, aligned gray), both at measure resolution.
/// The per-pixel focus operator runs on the unaligned full-res frame (it is local, and breathing
/// is <1%); pooling, breathing alignment and the window sum then run on the small maps.
pub fn frame_stage(g: &Gray, s: f32, p: &Params) -> (Gray, Gray) {
    let (mut m, win_after) = match p.measure {
        Measure::Sml => (modified_laplacian(g), true),
        other => (focus_measure(g, other, p.win), false),
    };
    let mut a = g.half();
    m = m.half();
    for _ in 1..p.measure_halvings {
        m = m.half();
        a = a.half();
    }
    if p.measure_halvings == 0 {
        a = g.clone();
        m = if win_after { modified_laplacian(g) } else { m };
    }
    let (m, a) = (rescale_about_center(&m, s), rescale_about_center(&a, s));
    let m = if win_after { box_blur(&m, (p.win >> p.measure_halvings).max(1)) } else { m };
    (m, a)
}

/// Everything after the per-frame stage: guided aggregation of the focus volume, peak finding,
/// mapping to diopters and optional smoothing. `filter` is applied per slice (lets the CLI
/// parallelise it).
pub fn volume_stage(
    mut measures: Vec<Gray>,
    small: &[Gray],
    diopters: &[f32],
    scales: Vec<f32>,
    p: &Params,
    map: impl Fn(Vec<Gray>, &Guide) -> Vec<Gray>,
) -> DepthResult {
    let aif = all_in_focus(small, &measures);
    let raw_conf = if p.norm { Some(depth_from_volume(&measures, Peak::Gauss3).conf) } else { None };
    if p.agg_r > 0 && p.norm {
        // Each pixel's curve becomes (m_k - min) / sum(m - min), weighted by how peaked it was
        // (confidence - 1, clamped), so flat textureless curves contribute little.
        let n = measures.len();
        let conf = raw_conf.as_ref().unwrap();
        for px in 0..measures[0].data.len() {
            let lo = (0..n).map(|k| measures[k].data[px]).fold(f32::MAX, f32::min);
            let sum: f32 = (0..n).map(|k| measures[k].data[px] - lo).sum();
            let wgt = (conf[px] - 1.0).clamp(0.0, 4.0) / sum.max(1e-12);
            for k in 0..n {
                measures[k].data[px] = (measures[k].data[px] - lo) * wgt;
            }
        }
    }
    if p.agg_r > 0 {
        let g = if p.guide_blur > 0 { blur(&aif, p.guide_blur) } else { aif.clone() };
        let guide = Guide::new(g, p.agg_r, p.agg_eps);
        measures = map(measures, &guide);
    }
    let mut d = depth_from_volume(&measures, p.peak);
    if let Some(c) = raw_conf {
        d.conf = c;
    }
    let diopter: Vec<f32> = d.index.iter().map(|&i| interp_index(diopters, i)).collect();
    let diopter_smooth =
        if p.smooth_r > 0 { smooth_depth(d.w, d.h, &diopter, &d.conf, p.smooth_r) } else { diopter.clone() };
    DepthResult {
        w: d.w, h: d.h, diopter, diopter_smooth, conf: d.conf, index: d.index, spread: d.spread,
        aif: aif.data, scales,
    }
}

/// Whole pipeline, single-threaded (used by the wasm build; the CLI runs the same stages in parallel).
/// `frames` are grayscale, `diopters` the (lag-corrected) focus of each frame in 1/m.
pub fn run_pipeline(frames: &[Gray], diopters: &[f32], p: &Params) -> DepthResult {
    let small: Vec<Gray> = frames.iter().map(|g| g.half().half()).collect();
    let (scales, _) = align_scales(&small);
    drop(small);
    let (measures, small): (Vec<Gray>, Vec<Gray>) = frames.iter().zip(&scales).map(|(g, &s)| frame_stage(g, s, p)).unzip();
    volume_stage(measures, &small, diopters, scales, p, |v, g| v.iter().map(|m| g.filter(m)).collect())
}

/// Incremental pipeline: push frames as they arrive, each one is aligned and measured right away
/// (breathing scale chained from the previous frame), so only the volume stage remains at the end.
/// Frames are referenced to the first frame; samples that fall outside a frame are edge-clamped
/// (breathing is <1%, so this only touches a thin border).
pub struct StackBuilder {
    p: Params,
    prev_structure: Option<Gray>,
    cum: f32,
    pub scales: Vec<f32>,
    measures: Vec<Gray>,
    small: Vec<Gray>,
    // Running per-pixel peak tracking for early (streamed) decisions.
    best_v: Vec<f32>,
    best_k: Vec<u32>,
    sum: Vec<f32>,
    decided: Vec<bool>,
}

/// A pixel whose focus peak is clearly behind the sweep, decided before the stack is complete.
#[derive(Clone, Copy, Debug)]
pub struct EarlyPixel {
    pub px: u32,
    /// Fractional best-focus frame index (centroid over the window around the peak).
    pub index: f32,
    pub conf: f32,
}

impl StackBuilder {
    pub fn new(p: Params) -> Self {
        Self {
            p, prev_structure: None, cum: 1.0, scales: vec![], measures: vec![], small: vec![],
            best_v: vec![], best_k: vec![], sum: vec![], decided: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.measures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.measures.is_empty()
    }

    pub fn push(&mut self, g: &Gray) {
        let st = structure(&g.half().half(), 2);
        if let Some(prev) = &self.prev_structure {
            let (s, _) = estimate_relative_scale_fast(prev, &st);
            self.cum *= s;
        }
        self.prev_structure = Some(st);
        self.scales.push(self.cum);
        let (m, a) = frame_stage(g, self.cum, &self.p);
        let k = self.measures.len() as u32;
        if self.best_v.is_empty() {
            let n = m.data.len();
            (self.best_v, self.best_k, self.sum, self.decided) = (vec![f32::MIN; n], vec![0; n], vec![0.0; n], vec![false; n]);
        }
        for (i, &v) in m.data.iter().enumerate() {
            self.sum[i] += v;
            if v > self.best_v[i] {
                self.best_v[i] = v;
                self.best_k[i] = k;
            }
        }
        self.measures.push(m);
        self.small.push(a);
    }

    /// Measure-resolution size of the maps (0 before the first frame).
    pub fn dims(&self) -> (usize, usize) {
        self.measures.first().map_or((0, 0), |m| (m.w, m.h))
    }

    /// Pixels that can be decided now: their peak is at least `r` frames old, their sharpness has
    /// since dropped below `drop` x peak, and the curve is peaked enough (`min_conf`, peak/mean so
    /// far). Each pixel is returned once. Uses the raw (unaggregated) measures; finish() refines.
    pub fn take_decided(&mut self, r: usize, drop: f32, min_conf: f32) -> Vec<EarlyPixel> {
        let n = self.measures.len();
        let mut out = vec![];
        if n <= r {
            return out;
        }
        let last = &self.measures[n - 1];
        for i in 0..self.best_v.len() {
            if self.decided[i] {
                continue;
            }
            let k = self.best_k[i] as usize;
            if k + r > n - 1 || last.data[i] > drop * self.best_v[i] {
                continue;
            }
            let conf = self.best_v[i] / (self.sum[i] / n as f32).max(1e-9);
            if conf < min_conf {
                continue;
            }
            let (lo, hi) = (k.saturating_sub(r), k + r);
            let base = (lo..=hi).map(|j| self.measures[j].data[i]).fold(f32::MAX, f32::min);
            let (mut sw, mut sk) = (0.0f32, 0.0f32);
            for j in lo..=hi {
                let w = self.measures[j].data[i] - base;
                sw += w;
                sk += w * j as f32;
            }
            self.decided[i] = true;
            out.push(EarlyPixel { px: i as u32, index: if sw > 0.0 { sk / sw } else { k as f32 }, conf });
        }
        out
    }

    /// A quick preview without consuming the stack: the volume stage on measures pooled down by
    /// `levels` more halvings (guided-filter radius scaled to match). Call finish() for full detail.
    pub fn finish_coarse(&self, diopters: &[f32], levels: usize) -> DepthResult {
        let down = |v: &[Gray]| -> Vec<Gray> {
            v.iter().map(|g| (0..levels).fold(g.clone(), |a, _| a.half())).collect()
        };
        let mut p = self.p;
        p.agg_r = (p.agg_r >> levels).max(if p.agg_r > 0 { 1 } else { 0 });
        p.smooth_r >>= levels;
        let small = down(&self.small);
        volume_stage(down(&self.measures), &small, diopters, self.scales.clone(), &p, |v, g| v.iter().map(|m| g.filter(m)).collect())
    }

    pub fn finish(mut self, diopters: &[f32]) -> DepthResult {
        (self.best_v, self.best_k, self.sum, self.decided) = (vec![], vec![], vec![], vec![]);
        let p = self.p;
        volume_stage(self.measures, &self.small, diopters, self.scales, &p, |v, g| v.iter().map(|m| g.filter(m)).collect())
    }
}

/// Mark depth discontinuities ("flying pixels"): true where depth within the 3x3 neighbourhood
/// varies by more than `rel` of the local depth.
pub fn discontinuities(w: usize, h: usize, diopter: &[f32], rel: f32) -> Vec<bool> {
    let mut o = vec![false; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for dy in 0..3 {
                for dx in 0..3 {
                    let z = 1.0 / diopter[(y + dy - 1) * w + x + dx - 1].max(1e-3);
                    lo = lo.min(z);
                    hi = hi.max(z);
                }
            }
            o[y * w + x] = (hi - lo) > rel * lo;
        }
    }
    o
}

/// Turbo colormap (Google's polynomial approximation), t in [0,1].
pub fn turbo(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) as f64;
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    let c = |v: f64| (v.clamp(0.0, 1.0) * 255.0) as u8;
    [c(r), c(g), c(b)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_preview_matches_fine_layout() {
        let p = Params { measure_halvings: 1, ..Params::default() };
        let mut b = StackBuilder::new(p);
        for k in 0..8 {
            let mut g = checker(64, 48, 2);
            if k != 3 { g = blur(&g, 2 + (k as i32 - 3).unsigned_abs() as usize); }
            b.push(&g);
        }
        let d: Vec<f32> = (0..8).map(|k| 1.0 + k as f32).collect();
        let c = b.finish_coarse(&d, 1);
        let f = b.finish(&d);
        assert_eq!((c.w * 2, c.h * 2), (f.w, f.h));
        let mid = |r: &DepthResult| r.index[(r.h / 2) * r.w + r.w / 2];
        assert!((mid(&c) - 3.0).abs() < 0.6 && (mid(&f) - 3.0).abs() < 0.6, "{} {}", mid(&c), mid(&f));
    }

    fn checker(w: usize, h: usize, p: usize) -> Gray {
        let mut g = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = if ((x / p) + (y / p)) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        g
    }

    #[test]
    fn recovers_scale() {
        let a = checker(200, 160, 9);
        let b = rescale_about_center(&a, 1.0 / 1.02); // b is a zoomed in by 2%
        let (s, _) = estimate_relative_scale_fast(&structure(&a, 1), &structure(&b, 1));
        assert!((s - 1.02).abs() < 0.002, "s = {s}");
    }

    #[test]
    fn picks_sharpest_frame() {
        let sharp = checker(64, 64, 4);
        let frames = vec![blur(&sharp, 3), blur(&sharp, 1), sharp.clone(), blur(&sharp, 1), blur(&sharp, 3)];
        let m: Vec<Gray> = frames.iter().map(|f| sml(f, 2)).collect();
        let d = depth_from_measures(&m);
        let mid = d.index[32 * 64 + 32];
        assert!((mid - 2.0).abs() < 0.3, "idx = {mid}");
    }

    #[test]
    fn streams_decisions_after_peak() {
        let sharp = checker(64, 64, 4);
        let sig = [4, 3, 2, 1, 0, 1, 2, 3, 4, 4];
        let mut b = StackBuilder::new(Params { win: 2, measure_halvings: 0, ..Params::default() });
        let mut decided_at = None;
        for (k, &r) in sig.iter().enumerate() {
            b.push(&if r == 0 { sharp.clone() } else { blur(&sharp, r) });
            let d = b.take_decided(2, 0.7, 1.2);
            if decided_at.is_none() && d.iter().any(|e| e.px == 32 * 64 + 32) {
                let e = d.iter().find(|e| e.px == 32 * 64 + 32).unwrap();
                assert!((e.index - 4.0).abs() < 0.3, "idx {}", e.index);
                decided_at = Some(k);
            }
        }
        assert!(matches!(decided_at, Some(6..=7)), "decided at {decided_at:?}");
    }

    #[test]
    fn plane_fit() {
        let mut pts = vec![];
        for i in 0..400 {
            let (x, y) = ((i % 20) as f32, (i / 20) as f32);
            let z = 0.1 * x - 0.2 * y + 3.0 + if i % 7 == 0 { 5.0 } else { 0.0 };
            pts.push([x, y, z]);
        }
        let (pl, _) = fit_plane_ransac(&pts, 200, 0.01, 1).unwrap();
        assert!((pl[0] - 0.1).abs() < 1e-4 && (pl[1] + 0.2).abs() < 1e-4 && (pl[2] - 3.0).abs() < 1e-3);
    }
}
