"""Run dff with several configurations on one stack and score each against the flat mat.

uv run --with numpy --with scipy tools/compare.py data/stacks/lag_up "baseline=" "agg=--agg 8" ...

Metrics (board mask from out/grid02/board_mask.npy, ground truth = plane fit in diopters):
  int_D   median |residual| on the board interior (raw, unsmoothed depth)
  edge_D  same, in a band along the board edge (where smoothing drags in background)
  snap%   share of confident pixels within +-0.05 frame of a whole frame (10% = no snapping)
  cov%    share of board pixels above the confidence threshold
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
stack = sys.argv[1]
mask = np.load(ROOT / "out/grid02/board_mask.npy")
interior = ndimage.binary_erosion(mask, iterations=25)
edge = mask & ~interior


def fit_plane(z, m, W, H):
    y, x = np.nonzero(m & np.isfinite(z))
    A = np.c_[x / W, y / H, np.ones(len(x))]
    zz = z[y, x]
    k = np.ones(len(zz), bool)
    for _ in range(5):
        c, *_ = np.linalg.lstsq(A[k], zz[k], rcond=None)
        r = zz - A @ c
        k = np.abs(r) < 3 * r[k].std()
    gy, gx = np.mgrid[0:H, 0:W]
    return c[0] * gx / W + c[1] * gy / H + c[2]


rows = []
for spec in sys.argv[2:]:
    name, _, flags = spec.partition("=")
    out = ROOT / "out" / f"cmp_{name}"
    r = subprocess.run([str(ROOT / "target/release/dff"), "depth", stack, str(out), *flags.split()], capture_output=True, text=True)
    if r.returncode:
        print(name, "FAILED", r.stderr[-400:])
        continue
    m = json.loads((out / "metrics.json").read_text())
    W, H = m["width"], m["height"]
    ld = lambda n: np.fromfile(out / n, dtype="<f4").reshape(H, W)
    d, ds, conf, idx = ld("diopter.f32"), ld("diopter_smooth.f32"), ld("conf.f32"), ld("index.f32")
    plane = fit_plane(d, interior, W, H)
    res = np.abs(d - plane)
    ress = np.abs(ds - plane)
    good = conf > m["conf_thresh"]
    fr = np.abs((idx[good] + 0.5) % 1 - 0.5)
    rows.append({
        "config": name,
        "int_D": np.median(res[interior]), "edge_D": np.median(res[edge]),
        "int_smooth_D": np.median(ress[interior]), "edge_smooth_D": np.median(ress[edge]),
        "snap%": 100 * np.mean(fr < 0.05), "cov%": 100 * np.mean(good[mask]), "sec": m["seconds"],
    })

keys = list(rows[0].keys())
print(" | ".join(f"{k:>13s}" for k in keys))
for r in rows:
    print(" | ".join(f"{r[k]:>13.4f}" if isinstance(r[k], float) else f"{r[k]:>13s}" for k in keys))
