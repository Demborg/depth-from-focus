//! Per-stage timings for one 1080x1920 frame, single-threaded: cargo run --release --example stage_bench
use dff_core::*;
use std::time::Instant;

fn main() {
    let (w, h) = (1080, 1920);
    let rgba: Vec<u8> = (0..w * h * 4).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
    let p = Params::default();
    let reps = 10;
    let time = |name: &str, f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..reps { f(); }
        println!("{name:28} {:7.2} ms", t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
    };
    let g = Gray::from_rgb(w, h, &rgba, 4);
    time("rgba -> gray", &mut || { std::hint::black_box(Gray::from_rgb(w, h, &rgba, 4)); });
    let q = g.half().half();
    time("quarter (half x2)", &mut || { std::hint::black_box(g.half().half()); });
    let st = structure(&q, 2);
    time("structure (quarter)", &mut || { std::hint::black_box(structure(&q, 2)); });
    time("scale search (fast)", &mut || { std::hint::black_box(estimate_relative_scale_fast(&st, &st)); });
    time("rescale about center", &mut || { std::hint::black_box(rescale_about_center(&g, 0.997)); });
    time("sml win6", &mut || { std::hint::black_box(sml(&g, 6)); });
    let m = sml(&g, 6);
    time("halving x2 (measure+gray)", &mut || { std::hint::black_box((m.half().half(), g.half().half())); });
    time("frame_stage total", &mut || { std::hint::black_box(frame_stage(&g, 0.997, &Params { measure_halvings: 2, ..p })); });
}
