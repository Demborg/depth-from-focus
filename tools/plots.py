"""Figures for one depth run.  uv run --with numpy --with matplotlib tools/plots.py out/<id> data/stacks/<id>"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, stack = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads((out / "metrics.json").read_text())
W, H, N = m["width"], m["height"], m["frames"]
ld = lambda n: np.fromfile(out / n, dtype="<f4").reshape(H, W)
dio, conf, idx, plane = ld("diopter.f32"), ld("conf.f32"), ld("index.f32"), ld("plane.f32")
meas = np.fromfile(out / "measures.f32", dtype="<f4").reshape(N, H, W)
focus = np.array(m["focus_m"])
fdio = 1 / focus
aif = plt.imread(out / "aif.png")
good = conf > m["conf_thresh"]
board = good & (np.abs(dio - plane) < 0.15)
if (out / "board_mask.npy").exists():  # from tools/eval.py: colour-segmented board + its own plane fit
    bm = np.load(out / "board_mask.npy")
    yy, xx = np.nonzero(bm)
    A = np.c_[xx / W, yy / H, np.ones(len(xx))]
    z = dio[yy, xx]; keep = np.ones(len(z), bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        rr = z - A @ coef; keep = np.abs(rr) < 3 * rr[keep].std()
    gy, gx = np.mgrid[0:H, 0:W]
    plane = coef[0] * gx / W + coef[1] * gy / H + coef[2]
    board = bm
depth_cm = 100 / dio
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# 1. Overview ---------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(15, 7))
ax[0].imshow(aif); ax[0].set_title("all-in-focus (from stack)")
lo, hi = np.percentile(depth_cm[board], [1, 99])
d = np.where(good, depth_cm, np.nan)
im = ax[1].imshow(d, cmap="turbo_r", vmin=lo - 5, vmax=hi + 15); ax[1].set_title(f"depth (conf > {m['conf_thresh']:.2g})")
plt.colorbar(im, ax=ax[1], fraction=0.046, label="cm")
im = ax[2].imshow(conf, cmap="magma", vmin=1, vmax=5); ax[2].set_title("confidence = peak / mean")
plt.colorbar(im, ax=ax[2], fraction=0.046)
r_mm = np.where(board, 1000 * (1 / dio - 1 / plane), np.nan)
im = ax[3].imshow(r_mm, cmap="RdBu_r", vmin=-30, vmax=30); ax[3].set_title("board residual vs fitted plane")
plt.colorbar(im, ax=ax[3], fraction=0.046, label="mm")
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig(out / "fig_overview.png"); plt.close(fig)

# 2. Focus curves at a few pixels ---------------------------------------------------------------
def pick(mask, q):
    ys, xs = np.nonzero(mask)
    order = np.argsort(dio[ys, xs])
    i = order[int(q * (len(order) - 1))]
    return ys[i], xs[i]

pts = {"board, far": pick(board, 0.03), "board, mid": pick(board, 0.5), "board, near": pick(board, 0.97)}
weak = (~good) & (conf > 1.05)
if weak.any():
    pts["low-confidence"] = pick(weak, 0.5)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
for (name, (y, x)), c in zip(pts.items(), ["C0", "C2", "C3", "0.5"]):
    curve = meas[:, y, x]
    ax[0].plot(fdio, curve / curve.max(), "o-", ms=3, color=c, label=f"{name}: peak {100 / dio[y, x]:.1f} cm, conf {conf[y, x]:.1f}")
    ax[0].axvline(dio[y, x], color=c, ls=":", lw=1)
    ax[1].plot(x, y, "o", color=c, ms=8, mec="k")
ax[0].set_xlabel("focus setting (diopters = 1 / metres)"); ax[0].set_ylabel("focus measure (normalised)")
sec = ax[0].secondary_xaxis("top", functions=(lambda D: 100 / np.maximum(D, 1e-3), lambda c: 100 / np.maximum(c, 1e-3)))
sec.set_xlabel("focus distance (cm)"); sec.set_ticks([15, 20, 25, 30, 40, 60, 100, 300])
ax[0].legend(fontsize=8); ax[0].set_title("per-pixel focus curves across the sweep"); ax[0].grid(alpha=0.3)
ax[1].imshow(aif); ax[1].axis("off")
fig.tight_layout(); fig.savefig(out / "fig_focus_curves.png"); plt.close(fig)

# 3. Sweep diagnostics: breathing + timing -----------------------------------------------------
meta = json.loads((stack / "meta.json").read_text())
t = np.array([f["t_ms"] for f in meta["frames"]]) / 1000
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
sc = np.array(m["scales"])
ax[0].plot(fdio, (1 / sc - 1) * 100, "o-", ms=3)
ax[0].set_xlabel("focus (diopters)"); ax[0].set_ylabel("relative magnification (%)"); ax[0].set_title("focus breathing (estimated)"); ax[0].grid(alpha=0.3)
ax[1].plot(t, focus * 100, "o-", ms=3)
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("focus distance (cm)"); ax[1].set_yscale("log"); ax[1].set_title(f"sweep: {N} frames in {meta['total_ms'] / 1000:.1f}s"); ax[1].grid(alpha=0.3)
hist_vals = idx[good].ravel()
ax[2].hist(hist_vals, bins=np.arange(-0.5, N, 0.25), color="C1")
ax[2].set_xlabel("best-focus frame index (sub-frame)"); ax[2].set_ylabel("pixels"); ax[2].set_title("where the confident pixels focus"); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "fig_sweep.png"); plt.close(fig)

# 4. Board as a 3D surface (approx. pinhole intrinsics, Pixel 9 main cam ~69 deg on long side) --
f_px = 1780 * max(W, H) / 1920  # calibrated from the mat grid (tools/calib_fov.py)
ys, xs = np.nonzero(good)
sel = np.random.default_rng(0).choice(len(ys), min(25000, len(ys)), replace=False)
ys, xs = ys[sel], xs[sel]
Z = 1 / dio[ys, xs]
X = (xs - W / 2) / f_px * Z
Y = (ys - H / 2) / f_px * Z
keep = (Z > np.percentile(Z, 1)) & (Z < np.percentile(Z, 97))
fig = plt.figure(figsize=(13, 6))
for k, (el, az) in enumerate([(25, -60), (5, -100)]):
    a = fig.add_subplot(1, 2, k + 1, projection="3d")
    cols = aif[ys[keep], xs[keep], :3]
    a.scatter(X[keep] * 100, Z[keep] * 100, -Y[keep] * 100, c=cols, s=0.6, depthshade=False)
    a.set_xlabel("x (cm)"); a.set_ylabel("depth (cm)"); a.set_zlabel("y (cm)")
    a.view_init(elev=el, azim=az); a.set_box_aspect((1, 1.4, 1.3))
fig.suptitle("confident pixels back-projected to 3D (colours from all-in-focus image)")
fig.tight_layout(); fig.savefig(out / "fig_3d.png"); plt.close(fig)
print("wrote figures to", out)
