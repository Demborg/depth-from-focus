"""Package runs into report/ for the interactive results page.

uv run --with numpy --with pillow --with scipy --with matplotlib tools/build_report.py grid02 grid03_fast
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
turbo = matplotlib.colormaps["turbo_r"]
CW, CH = 135, 240  # curve grid (quarter of the half-res measure grid)


def colour(depth_cm, mask, lo, hi):
    t = np.clip((depth_cm - lo) / (hi - lo), 0, 1)
    rgb = (turbo(t)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = (rgb[~mask] * 0.18 + 30).astype(np.uint8)
    return rgb


def build(run):
    out, stack = ROOT / "out" / run, ROOT / "data" / "stacks" / run
    d = REP / run
    (d / "frames").mkdir(parents=True, exist_ok=True)
    m = json.loads((out / "metrics.json").read_text())
    meta = json.loads((stack / "meta.json").read_text())
    W, H, N = m["width"], m["height"], m["frames"]
    ld = lambda n: np.fromfile(out / n, dtype="<f4").reshape(H, W)
    dio, conf = ld("diopter.f32"), ld("conf.f32")
    meas = np.fromfile(out / "measures.f32", dtype="<f4").reshape(N, H, W)

    for i, f in enumerate(meta["frames"]):
        Image.open(stack / f["file"]).resize((360, 640), Image.LANCZOS).save(d / "frames" / f"{i:03d}.jpg", quality=82)

    # depth: raw and confidence-weighted smooth (the best variant from the scratch experiments)
    wt = conf**4
    smooth = ndimage.gaussian_filter(dio * wt, 10) / np.maximum(ndimage.gaussian_filter(wt, 10), 1e-9)
    good = conf > m["conf_thresh"]
    cm_raw, cm_s = 100 / dio, 100 / smooth
    lo, hi = np.percentile(cm_s[good], [2, 97])
    smask = ndimage.gaussian_filter(good.astype(float), 6) > 0.35
    Image.fromarray(colour(cm_raw, good, lo, hi)).save(d / "depth_raw.png")
    Image.fromarray(colour(cm_s, smask, lo, hi)).save(d / "depth_smooth.png")
    c8 = (np.clip((conf - 1) / 4, 0, 1) * 255).astype(np.uint8)
    Image.fromarray((matplotlib.colormaps["magma"](c8)[..., :3] * 255).astype(np.uint8)).save(d / "conf.png")
    shutil.copy(out / "aif.png", d / "aif.png")

    # per-pixel focus curves on a coarse grid, normalised per pixel to uint8
    k = W // CW
    mc = meas[:, : CH * k, : CW * k].reshape(N, CH, k, CW, k).mean((2, 4))
    mc = mc / np.maximum(mc.max(0, keepdims=True), 1e-9)
    # stacked vertically: frame k occupies rows [k*CH, (k+1)*CH). Lossless PNG, decoded via canvas.
    Image.fromarray((mc * 255).astype(np.uint8).reshape(N * CH, CW), "L").save(d / "curves.png")
    (d / "curves.u8").unlink(missing_ok=True)
    dg = lambda a: a[: CH * k, : CW * k].reshape(CH, k, CW, k).mean((1, 3))
    grid = {
        "depth_raw_cm": np.round(dg(cm_raw), 1).ravel().tolist(),
        "depth_smooth_cm": np.round(dg(cm_s), 1).ravel().tolist(),
        "conf": np.round(dg(conf), 2).ravel().tolist(),
    }
    ev = json.loads((out / "eval.json").read_text()) if (out / "eval.json").exists() else {}
    info = {
        "run": run, "n": N, "cw": CW, "ch": CH, "cm_range": [float(lo), float(hi)],
        "focus_m": m["focus_m"], "t_ms": [f["t_ms"] for f in meta["frames"]],
        "total_ms": meta["total_ms"], "settle": meta["params"]["settle"],
        "seconds_cpu": m["seconds"], "scales": m["scales"], "eval": ev, **grid,
    }
    (d / "info.json").write_text(json.dumps(info))
    for fig in ["fig_lens_response.png", "fig_profile.png", "fig_sweep.png", "fig_overview.png", "fig_focus_curves.png"]:
        if (out / fig).exists():
            shutil.copy(out / fig, d / fig)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        build(r)
    print("ok", sorted(p.name for p in REP.iterdir()))
