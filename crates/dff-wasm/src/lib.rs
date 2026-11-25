//! Raw C-ABI wasm wrapper around dff-core (no wasm-bindgen; the JS glue is web/dff-wasm.js).
//!
//! Streaming: each pushed frame (full capture resolution, RGBA) is aligned and focus-measured
//! immediately, so the work overlaps the sweep and only the volume stage runs at the end. Only
//! the per-frame focus measures (at measure resolution) are kept, not the frames.
//!
//! JS flow: ctx = dff_new(w, h, ...params); for each frame { write RGBA at dff_rgba_ptr(ctx); dff_push(ctx) }
//! dff_finish(ctx, diopters_ptr) after writing per-frame diopters into dff_diopters_ptr(ctx, n);
//! read dff_out_w/h and Float32 views at dff_out_ptr(ctx, k).
use dff_core::*;

pub struct Ctx {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
    builder: Option<StackBuilder>,
    early_px: Vec<u32>,
    early_idx: Vec<f32>,
    early_conf: Vec<f32>,
    diopters: Vec<f32>,
    out: Option<DepthResult>,
}

/// measure: 0 = SML, 1 = RDF. peak_r: 0 = 3-point Gaussian, R > 0 = centroid over +-R frames.
/// agg_r: guided-filter radius on the focus volume (0 = off); norm: normalise curves first.
#[no_mangle]
pub extern "C" fn dff_new(
    w: u32, h: u32, win: u32, halvings: u32, smooth_r: u32, measure: u32, peak_r: u32, agg_r: u32, agg_eps: f32, norm: u32,
) -> *mut Ctx {
    let (w, h) = (w as usize, h as usize);
    let p = Params {
        win: win as usize,
        measure_halvings: halvings as usize,
        smooth_r: smooth_r as usize,
        measure: if measure == 1 { Measure::Rdf(0, 1, 2) } else { Measure::Sml },
        peak: if peak_r == 0 { Peak::Gauss3 } else { Peak::Centroid(peak_r as usize) },
        agg_r: agg_r as usize,
        agg_eps,
        guide_blur: 0,
        norm: norm != 0,
    };
    Box::into_raw(Box::new(Ctx { w, h, rgba: vec![0; w * h * 4], builder: Some(StackBuilder::new(p)), early_px: vec![], early_idx: vec![], early_conf: vec![], diopters: vec![], out: None }))
}

#[no_mangle]
pub extern "C" fn dff_free(ctx: *mut Ctx) {
    if !ctx.is_null() {
        unsafe { drop(Box::from_raw(ctx)) }
    }
}

/// Scratch RGBA buffer (w*h*4 bytes) for the next frame.
#[no_mangle]
pub extern "C" fn dff_rgba_ptr(ctx: *mut Ctx) -> *mut u8 {
    unsafe { (*ctx).rgba.as_mut_ptr() }
}

/// Align and focus-measure the frame in the scratch buffer. Returns the frame count.
#[no_mangle]
pub extern "C" fn dff_push(ctx: *mut Ctx) -> u32 {
    let c = unsafe { &mut *ctx };
    let g = Gray::from_rgb(c.w, c.h, &c.rgba, 4);
    let b = c.builder.as_mut().unwrap();
    b.push(&g);
    b.len() as u32
}

/// Decide pixels whose focus peak is clearly behind the sweep (see StackBuilder::take_decided).
/// Returns how many; read them via dff_early_ptr(ctx, 0 = pixel ids u32, 1 = index f32, 2 = conf f32).
#[no_mangle]
pub extern "C" fn dff_take_decided(ctx: *mut Ctx, r: u32, drop: f32, min_conf: f32) -> u32 {
    let c = unsafe { &mut *ctx };
    let d = c.builder.as_mut().unwrap().take_decided(r as usize, drop, min_conf);
    c.early_px = d.iter().map(|e| e.px).collect();
    c.early_idx = d.iter().map(|e| e.index).collect();
    c.early_conf = d.iter().map(|e| e.conf).collect();
    d.len() as u32
}

#[no_mangle]
pub extern "C" fn dff_early_ptr(ctx: *mut Ctx, which: u32) -> *const u8 {
    let c = unsafe { &*ctx };
    match which {
        0 => c.early_px.as_ptr() as *const u8,
        1 => c.early_idx.as_ptr() as *const u8,
        _ => c.early_conf.as_ptr() as *const u8,
    }
}

/// Measure-resolution map size and the breathing scale of frame k (valid during streaming).
#[no_mangle]
pub extern "C" fn dff_dims(ctx: *mut Ctx) -> u32 {
    let (w, h) = unsafe { (*ctx).builder.as_ref().unwrap().dims() };
    ((w as u32) << 16) | h as u32
}

#[no_mangle]
pub extern "C" fn dff_scale(ctx: *mut Ctx, k: u32) -> f32 {
    unsafe { (*ctx).builder.as_ref().unwrap().scales[k as usize] }
}

/// Buffer for n per-frame focus values (1/m), filled by JS before dff_finish.
#[no_mangle]
pub extern "C" fn dff_diopters_ptr(ctx: *mut Ctx, n: u32) -> *mut f32 {
    let c = unsafe { &mut *ctx };
    c.diopters = vec![1.0; n as usize];
    c.diopters.as_mut_ptr()
}

/// Coarse preview (measures pooled down by `levels`) into the output, keeping the stack so that
/// dff_finish can still run at full measure resolution.
#[no_mangle]
pub extern "C" fn dff_finish_coarse(ctx: *mut Ctx, levels: u32) {
    let c = unsafe { &mut *ctx };
    c.out = Some(c.builder.as_ref().unwrap().finish_coarse(&c.diopters, levels as usize));
}

#[no_mangle]
pub extern "C" fn dff_finish(ctx: *mut Ctx) {
    let c = unsafe { &mut *ctx };
    let b = c.builder.take().unwrap();
    c.out = Some(b.finish(&c.diopters));
}

#[no_mangle]
pub extern "C" fn dff_out_w(ctx: *mut Ctx) -> u32 {
    unsafe { (*ctx).out.as_ref().map_or(0, |o| o.w as u32) }
}

#[no_mangle]
pub extern "C" fn dff_out_h(ctx: *mut Ctx) -> u32 {
    unsafe { (*ctx).out.as_ref().map_or(0, |o| o.h as u32) }
}

/// 0 = diopter, 1 = diopter_smooth, 2 = conf, 3 = index, 4 = scales (one per frame), 5 = spread, 6 = aif.
#[no_mangle]
pub extern "C" fn dff_out_ptr(ctx: *mut Ctx, which: u32) -> *const f32 {
    let o = unsafe { (*ctx).out.as_ref().unwrap() };
    match which {
        0 => o.diopter.as_ptr(),
        1 => o.diopter_smooth.as_ptr(),
        2 => o.conf.as_ptr(),
        3 => o.index.as_ptr(),
        5 => o.spread.as_ptr(),
        6 => o.aif.as_ptr(),
        _ => o.scales.as_ptr(),
    }
}
