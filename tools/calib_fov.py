"""Calibrate the camera focal length from the cutting mat's 1 cm grid.

For a candidate focal length f, the measured mat plane (inverse depth affine in pixel coords)
defines a 3D plane. Resample the all-in-focus image onto that plane in millimetres, FFT it, and
read the two grid periods. The right f gives a square 10 mm grid.

uv run --with numpy --with scipy --with pillow --with matplotlib tools/calib_fov.py out/<run>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from eval import board_mask  # noqa: E402

out = Path(sys.argv[1])
m = json.loads((out / "metrics.json").read_text())
W, H = m["width"], m["height"]
dio = np.fromfile(out / "diopter_smooth.f32", dtype="<f4").reshape(H, W)
aif = np.asarray(Image.open(out / "aif.png").convert("RGB"))
gray = aif.mean(2) / 255.0
mask = board_mask(aif)
ys, xs = np.nonzero(mask)
A = np.c_[xs, ys, np.ones(len(xs))]
z = dio[ys, xs]
keep = np.ones(len(z), bool)
for _ in range(5):
    coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
    r = z - A @ coef
    keep = np.abs(r) < 3 * r[keep].std()
a, b, c = coef  # 1/Z = a*x + b*y + c, x/y in pixels at this resolution
cx, cy = (W - 1) / 2, (H - 1) / 2
center = np.array([np.median(xs), np.median(ys)])

STEP_MM, HALF = 0.5, 90  # resample +-90 mm around the mat centre at 0.5 mm


def grid_periods(f):
    # Plane n.P = 1 with P = Z * ((x-cx)/f, (y-cy)/f, 1)
    n = np.array([a * f, b * f, c + a * cx + b * cy])
    nh = n / np.linalg.norm(n)
    Zc = 1 / (a * center[0] + b * center[1] + c)
    P0 = Zc * np.array([(center[0] - cx) / f, (center[1] - cy) / f, 1.0])
    e1 = np.cross(nh, [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(nh, e1)
    s = np.arange(-HALF, HALF, STEP_MM) / 1000.0
    S, T = np.meshgrid(s, s)
    P = P0[None, None] + S[..., None] * e1 + T[..., None] * e2
    u = f * P[..., 0] / P[..., 2] + cx
    v = f * P[..., 1] / P[..., 2] + cy
    patch = ndimage.map_coordinates(gray, [v, u], order=1, mode="nearest")
    patch = patch - patch.mean()
    patch *= np.outer(np.hanning(len(s)), np.hanning(len(s)))
    PAD = 4  # zero-pad for sub-bin frequency resolution
    N = len(s) * PAD
    F = np.abs(np.fft.fftshift(np.fft.fft2(patch, s=(N, N))))
    ky, kx = np.mgrid[0:N, 0:N] - N // 2
    rad = np.hypot(kx, ky)
    F[(rad < 8 * PAD) | (rad > 60 * PAD)] = 0  # periods between ~3 and ~22 mm
    peaks = []
    for _ in range(2):
        i = np.unravel_index(np.argmax(F), F.shape)
        k = np.array([kx[i], ky[i]], float)
        peaks.append(k)
        # suppress this peak, its mirror and anything near the same direction
        ang = np.arctan2(ky, kx) - np.arctan2(k[1], k[0])
        ang = np.abs((ang + np.pi / 2) % np.pi - np.pi / 2)
        F[ang < 0.35] = 0
    per = [N * STEP_MM / np.linalg.norm(k) for k in peaks]
    angle = np.degrees(np.arccos(abs(peaks[0] @ peaks[1]) / (np.linalg.norm(peaks[0]) * np.linalg.norm(peaks[1]))))
    return per, angle, patch


fs = np.arange(0.9, 1.6, 0.01) * (max(W, H) / 2) / np.tan(np.radians(69 / 2))
rows = []
for f in fs:
    per, ang, _ = grid_periods(f)
    err = sum(np.log(p / 10.0) ** 2 for p in per) + np.radians(90 - ang) ** 2
    rows.append((f, per, ang, err))
best = min(rows, key=lambda r: r[3])
f0 = (max(W, H) / 2) / np.tan(np.radians(69 / 2))
fov_long = 2 * np.degrees(np.arctan((max(W, H) / 2) / best[0]))
print(f"assumed f = {f0:.0f} px (69 deg long side) at {W}x{H}")
print(f"best    f = {best[0]:.0f} px -> {fov_long:.1f} deg long side; grid periods {best[1][0]:.2f} / {best[1][1]:.2f} mm, angle {best[2]:.1f} deg")
for f, per, ang, err in rows[:: max(1, len(rows) // 12)]:
    print(f"  f={f:6.0f}  periods {per[0]:5.2f} {per[1]:5.2f} mm  angle {ang:5.1f}  err {err:.4f}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
for a_, f in zip(ax[:2], [f0, best[0]]):
    per, ang, patch = grid_periods(f)
    a_.imshow(patch, cmap="gray", extent=[-HALF, HALF, HALF, -HALF])
    a_.set_title(f"f={f:.0f}px: {per[0]:.1f}/{per[1]:.1f} mm, {ang:.0f}°")
    a_.set_xlabel("mm")
ax[2].plot([r[0] for r in rows], [r[3] for r in rows]); ax[2].axvline(f0, ls="--", c="k", lw=1, label="assumed")
ax[2].set_xlabel("focal length (px)"); ax[2].set_ylabel("grid error"); ax[2].legend(); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "fig_fov_calib.png")
json.dump({"W": W, "H": H, "f_px": float(best[0]), "fov_long_deg": float(fov_long), "periods_mm": best[1], "angle_deg": best[2]}, open(out / "fov_calib.json", "w"))
