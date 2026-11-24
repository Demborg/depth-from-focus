"""Score runs against the flat cutting board.

Board mask = teal pixels in the all-in-focus image (eroded). Ground truth = least-squares plane
in (u, v, diopter) on the board (inverse depth of a plane is affine in image coords), with
iterative 3-sigma trimming. Reports error in diopters (scale-free) and in mm.

  uv run --with numpy --with pillow --with scipy tools/eval.py out/grid01 out/grid02 ...
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def board_mask(aif):
    r, g, b = [aif[..., i].astype(np.float32) for i in range(3)]
    teal = (g > r + 4) & (b > r + 2) & (g > 30)
    teal = ndimage.binary_opening(teal, iterations=2)
    lab, n = ndimage.label(teal)
    if n == 0:
        return teal
    sizes = ndimage.sum(teal, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    # fill shadow holes: convex hull of the largest teal component
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(big)
    pts = np.c_[xs, ys]
    hull = pts[ConvexHull(pts).vertices]
    gy, gx = np.mgrid[0 : big.shape[0], 0 : big.shape[1]]
    big = MPath(hull).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(big.shape)
    return ndimage.binary_erosion(big, iterations=6)


def evaluate(out: Path):
    m = json.loads((out / "metrics.json").read_text())
    W, H = m["width"], m["height"]
    dio = np.fromfile(out / "diopter.f32", dtype="<f4").reshape(H, W)
    conf = np.fromfile(out / "conf.f32", dtype="<f4").reshape(H, W)
    aif = np.asarray(Image.open(out / "aif.png").convert("RGB"))
    mask = board_mask(aif)
    ys, xs = np.nonzero(mask)
    u, v, z = xs / W, ys / H, dio[ys, xs]
    A = np.c_[u, v, np.ones_like(u)]
    keep = np.ones(len(z), bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        s = r[keep].std()
        keep = np.abs(r) < 3 * s
    pz = A @ coef
    r = z - pz
    r_mm = 1000 * (1 / z - 1 / pz)
    np.save(out / "board_mask.npy", mask)
    res = {
        "run": out.name,
        "board_px": int(mask.sum()),
        "rms_D_all": float(np.sqrt(np.mean(r**2))),
        "mad_D": float(np.median(np.abs(r))),
        "outlier_frac(>0.1D)": float(np.mean(np.abs(r) > 0.1)),
        "median_abs_mm": float(np.median(np.abs(r_mm))),
        "p90_abs_mm": float(np.percentile(np.abs(r_mm), 90)),
        "board_depth_cm": [float(100 / pz.max()), float(100 / pz.min())],
        "seconds": m["seconds"],
    }
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    rows = [evaluate(Path(p)) for p in sys.argv[1:]]
    keys = ["run", "board_px", "mad_D", "rms_D_all", "outlier_frac(>0.1D)", "median_abs_mm", "p90_abs_mm", "board_depth_cm"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) if not isinstance(r[k], list) else f"{r[k][0]:.0f}-{r[k][1]:.0f}" for k in keys))
