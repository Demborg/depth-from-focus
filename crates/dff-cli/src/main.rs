//! dff depth <stack_dir> <out_dir> [--win R] [--conf C] [--no-align]
//!
//! Reads data/stacks/<id>/meta.json + frames, writes depth/confidence/all-in-focus PNGs,
//! raw arrays for plotting, and metrics.json (plane-fit residual against the flat board).
use dff_core::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize)]
struct Frame {
    file: String,
    focus_reported: Option<f32>,
    focus_requested: f32,
    t_ms: Option<f32>,
    t_req_ms: Option<f32>,
}
#[derive(Deserialize)]
struct Meta {
    frames: Vec<Frame>,
}

struct Args {
    stack: String,
    out: String,
    win: usize,
    conf: f32,
    align: bool,
    lag_ms: Option<f32>,
    p: Params,
}

fn parse() -> Args {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 || a[1] != "depth" {
        eprintln!("usage: dff depth <stack_dir> <out_dir> [--win R] [--conf C] [--no-align] [--lag-ms L]\n  [--measure sml|rdf] [--peak gauss|centroid:R] [--agg R] [--eps E] [--smooth R]");
        std::process::exit(2);
    }
    let mut args = Args { stack: a[2].clone(), out: a[3].clone(), win: 6, conf: 1.6, align: true, lag_ms: None, p: Params::default() };
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--win" => { args.win = a[i + 1].parse().unwrap(); i += 1; }
            "--conf" => { args.conf = a[i + 1].parse().unwrap(); i += 1; }
            "--no-align" => args.align = false,
            "--lag-ms" => { args.lag_ms = Some(a[i + 1].parse().unwrap()); i += 1; }
            "--measure" => {
                // sml | rdf:r_in:r1:r2
                let v: Vec<&str> = a[i + 1].split(':').collect();
                args.p.measure = if v[0] == "rdf" {
                    let n = |k: usize, d: usize| v.get(k).map_or(d, |x| x.parse().unwrap());
                    Measure::Rdf(n(1, 1), n(2, 3), n(3, 4))
                } else {
                    Measure::Sml
                };
                i += 1;
            }
            "--peak" => {
                args.p.peak = match a[i + 1].split_once(':') {
                    Some(("centroid", r)) => Peak::Centroid(r.parse().unwrap()),
                    _ => Peak::Gauss3,
                };
                i += 1;
            }
            "--agg" => { args.p.agg_r = a[i + 1].parse().unwrap(); i += 1; }
            "--eps" => { args.p.agg_eps = a[i + 1].parse().unwrap(); i += 1; }
            "--norm" => args.p.norm = true,
            "--gblur" => { args.p.guide_blur = a[i + 1].parse().unwrap(); i += 1; }
            "--smooth" => { args.p.smooth_r = a[i + 1].parse().unwrap(); i += 1; }
            o => panic!("unknown arg {o}"),
        }
        i += 1;
    }
    args
}

fn write_f32(path: &Path, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, b).unwrap();
}

fn main() {
    let mut args = parse();
    args.p.win = args.win;
    let t0 = Instant::now();
    let dir = Path::new(&args.stack);
    let out = Path::new(&args.out);
    fs::create_dir_all(out).unwrap();
    let meta: Meta = serde_json::from_slice(&fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let focus: Vec<f32> = meta.frames.iter().map(|f| f.focus_reported.unwrap_or(f.focus_requested)).collect();
    let mut diopter: Vec<f32> = focus.iter().map(|d| 1.0 / d).collect();
    // Lens lag: on by default for stacks that carry request timestamps (--lag-ms 0 disables).
    let has_t = meta.frames.iter().all(|f| f.t_req_ms.is_some() && f.t_ms.is_some());
    let lag = args.lag_ms.unwrap_or(if has_t { PIXEL9_LAG_MS } else { 0.0 });
    if has_t && lag != 0.0 {
        let req: Vec<(f32, f32)> = meta.frames.iter().map(|f| (f.t_req_ms.unwrap(), 1.0 / f.focus_requested)).collect();
        let grab: Vec<f32> = meta.frames.iter().map(|f| f.t_ms.unwrap()).collect();
        diopter = lagged_focus(&req, &grab, lag);
    }

    // Decode all frames (full res gray + RGB kept for all-in-focus).
    let decoded: Vec<(Gray, image::RgbImage)> = meta
        .frames
        .par_iter()
        .map(|f| {
            let im = image::open(dir.join(&f.file)).unwrap().to_rgb8();
            let g = Gray::from_rgb(im.width() as usize, im.height() as usize, im.as_raw(), 3);
            (g, im)
        })
        .collect();
    let (w, h) = (decoded[0].0.w, decoded[0].0.h);
    eprintln!("decoded {} frames {}x{} in {:.2}s", decoded.len(), w, h, t0.elapsed().as_secs_f32());

    // Breathing: estimate scales on quarter-res images.
    let t = Instant::now();
    let (scales, ncc) = if args.align {
        let small: Vec<Gray> = decoded.par_iter().map(|(g, _)| g.half().half()).collect();
        align_scales(&small)
    } else {
        (vec![1.0; decoded.len()], vec![1.0; decoded.len()])
    };
    eprintln!("scales {:.4}..{:.4} in {:.2}s", scales[0], scales[scales.len() - 1], t.elapsed().as_secs_f32());

    // Align + focus measure at full res, downsample 2x, aggregate the volume, find peaks.
    let t = Instant::now();
    let p = args.p;
    let (measures, small): (Vec<Gray>, Vec<Gray>) =
        decoded.par_iter().zip(&scales).map(|((g, _), &s)| frame_stage(g, s, &p)).unzip();
    eprintln!("focus measures in {:.2}s", t.elapsed().as_secs_f32());
    let all_m: Vec<f32> = measures.iter().flat_map(|m| m.data.iter().copied()).collect();
    write_f32(&out.join("measures.f32"), &all_m);
    drop(all_m);
    let t = Instant::now();
    let d = volume_stage(measures, &small, &diopter, scales.clone(), &p, |v, g| v.par_iter().map(|m| g.filter(m)).collect());
    let (mw, mh) = (d.w, d.h);
    eprintln!("volume stage in {:.2}s ({:?})", t.elapsed().as_secs_f32(), p);
    let dio = d.diopter.clone();
    let good: Vec<bool> = d.conf.iter().map(|&c| c > args.conf).collect();

    // Robust plane fit in (x, y, diopter) on confident pixels: the board is the dominant plane.
    let mut pts = vec![];
    for y in (0..mh).step_by(2) {
        for x in (0..mw).step_by(2) {
            let p = y * mw + x;
            if good[p] {
                pts.push([x as f32 / mw as f32, y as f32 / mh as f32, dio[p]]);
            }
        }
    }
    let plane = fit_plane_ransac(&pts, 500, 0.15, 7);
    let mut plane_map = vec![f32::NAN; mw * mh];
    let mut resid = vec![f32::NAN; mw * mh];
    let mut metrics = json!({
        "frames": decoded.len(), "width": mw, "height": mh, "win": args.win, "conf_thresh": args.conf,
        "align": args.align, "scales": scales, "scale_ncc": ncc, "focus_m": focus, "diopter_used": diopter, "lag_ms": lag,
        "confident_frac": good.iter().filter(|&&b| b).count() as f32 / good.len() as f32,
        "seconds": t0.elapsed().as_secs_f32(),
    });
    if let Some((pl, _)) = &plane {
        let mut r_in = vec![];
        let mut r_mm = vec![];
        for y in 0..mh {
            for x in 0..mw {
                let p = y * mw + x;
                let (u, v) = (x as f32 / mw as f32, y as f32 / mh as f32);
                let pd = (pl[0] * u as f64 + pl[1] * v as f64 + pl[2]) as f32;
                plane_map[p] = pd;
                let r = dio[p] - pd;
                resid[p] = r;
                if good[p] && r.abs() < 0.15 {
                    r_in.push(r);
                    r_mm.push(1000.0 * (1.0 / dio[p] - 1.0 / pd));
                }
            }
        }
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
        let mut abs_mm: Vec<f32> = r_mm.iter().map(|x| x.abs()).collect();
        abs_mm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        metrics["plane"] = json!({
            "coef_diopter": pl, "inlier_frac_of_confident": r_in.len() as f32 / good.iter().filter(|&&b| b).count().max(1) as f32,
            "rms_diopter": rms(&r_in), "rms_mm": rms(&r_mm),
            "median_abs_mm": abs_mm.get(abs_mm.len() / 2).copied().unwrap_or(f32::NAN),
        });
    }

    // Raw arrays for plotting.
    write_f32(&out.join("diopter.f32"), &dio);
    write_f32(&out.join("diopter_smooth.f32"), &d.diopter_smooth);
    write_f32(&out.join("spread.f32"), &d.spread);
    write_f32(&out.join("conf.f32"), &d.conf);
    write_f32(&out.join("index.f32"), &d.index);
    write_f32(&out.join("plane.f32"), &plane_map);

    // PNGs: depth (turbo over robust diopter range, low-confidence greyed), confidence, AIF.
    let mut sorted: Vec<f32> = dio.iter().zip(&good).filter(|(_, &g)| g).map(|(v, _)| *v).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| sorted.get(((sorted.len() as f32 - 1.0) * f) as usize).copied().unwrap_or(0.0);
    let (lo, hi) = (q(0.02), q(0.98));
    metrics["diopter_range_p2_p98"] = json!([lo, hi]);
    let mut img = image::RgbImage::new(mw as u32, mh as u32);
    let mut img_raw = image::RgbImage::new(mw as u32, mh as u32);
    let mut cimg = image::GrayImage::new(mw as u32, mh as u32);
    let mut aif = image::RgbImage::new(mw as u32, mh as u32);
    for y in 0..mh {
        for x in 0..mw {
            let p = y * mw + x;
            let c = turbo((dio[p] - lo) / (hi - lo));
            img_raw.put_pixel(x as u32, y as u32, image::Rgb(c));
            let c = if good[p] { c } else { [40, 40, 40] };
            img.put_pixel(x as u32, y as u32, image::Rgb(c));
            cimg.put_pixel(x as u32, y as u32, image::Luma([((d.conf[p] - 1.0) / 4.0 * 255.0).clamp(0.0, 255.0) as u8]));
            // All-in-focus: sample the nearest-best frame at the aligned position.
            let k = d.index[p].round() as usize;
            let (cx, cy) = ((w as f32 - 1.0) * 0.5, (h as f32 - 1.0) * 0.5);
            let s = scales[k];
            let sx = (cx + (2.0 * x as f32 + 0.5 - cx) * s).clamp(0.0, w as f32 - 1.0) as u32;
            let sy = (cy + (2.0 * y as f32 + 0.5 - cy) * s).clamp(0.0, h as f32 - 1.0) as u32;
            aif.put_pixel(x as u32, y as u32, *decoded[k].1.get_pixel(sx, sy));
        }
    }
    img.save(out.join("depth.png")).unwrap();
    img_raw.save(out.join("depth_raw.png")).unwrap();
    cimg.save(out.join("conf.png")).unwrap();
    aif.save(out.join("aif.png")).unwrap();
    metrics["seconds"] = json!(t0.elapsed().as_secs_f32());
    fs::write(out.join("metrics.json"), serde_json::to_string_pretty(&metrics).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&metrics["plane"]).unwrap());
    eprintln!("total {:.2}s", t0.elapsed().as_secs_f32());
}
