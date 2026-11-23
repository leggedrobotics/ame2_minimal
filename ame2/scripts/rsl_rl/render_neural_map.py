"""Render a neural-map recording (``play.py --record_nmap``) into an mp4: true terrain vs. the lidar neural map.

    python render_neural_map.py clip.npz --out clip.mp4 [--title "stairs up"] [--std_max 0.25] [--workers 8]

Layout: left, a 3D view following the robot (true terrain as a grey wireframe, the neural map as a surface
coloured by height, the latest lidar returns, the robot's links); right, top-down true height, top-down
neural-map height, and the elevation profile along the robot's heading (truth vs. map, lidar returns).
Cells never fused (std at its initial value) are left blank; the map's std is a fusion weight, not metres. Needs numpy, matplotlib and ffmpeg.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np

D: dict = {}
A: argparse.Namespace


def _load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["hits_off"] = np.concatenate([[0], np.cumsum(d["hits_len"])])
    return d


def _frame(k: int) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps, colors

    d = D
    x0, x1 = d["x_range"]; y0, y1 = d["y_range"]; res = float(d["res"])
    xs = np.arange(x0, x1 + 1e-6, res); ys = np.arange(y0, y1 + 1e-6, res)
    ix = (xs >= A.crop_x[0] - 1e-6) & (xs <= A.crop_x[1] + 1e-6)
    iy = (ys >= A.crop_y[0] - 1e-6) & (ys <= A.crop_y[1] + 1e-6)
    xs, ys = xs[ix], ys[iy]
    cx0, cx1, cy0, cy1 = xs[0], xs[-1], ys[0], ys[-1]
    Xl, Yl = np.meshgrid(xs, ys, indexing="ij")
    bp, yaw = d["base_pos"][k], float(d["base_yaw"][k])
    c, s = np.cos(yaw), np.sin(yaw)
    Xw, Yw = c * Xl - s * Yl + bp[0], s * Xl + c * Yl + bp[1]
    crop = np.ix_(ix, iy)
    gt = d["gt"][k][crop].astype(float)
    gt[~np.isfinite(gt) | (gt < -50)] = np.nan  # ray missed the terrain mesh (void)
    est, std = d["est"][k][crop], np.sqrt(np.maximum(d["var"][k][crop], 0.0))
    seen = std < A.std_max  # fused at least once (never-observed cells keep the initial std)
    est_m = np.where(seen, est, np.nan)
    both = seen & np.isfinite(gt)
    err = np.abs(est - gt)[both]
    hits = d["hits"][d["hits_off"][k]:d["hits_off"][k + 1]]
    r = hits[:, :2] - bp[:2]
    hx, hy = c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1]
    hits_in = (hx > cx0) & (hx < cx1) & (hy > cy0) & (hy < cy1)

    ground = bp[2] - A.base_height  # nominal ground under the robot
    zlo, zhi = ground - A.zspan / 2, ground + A.zspan / 2  # per-frame window that follows the robot
    norm = colors.Normalize(*A.clim)  # one colour scale for the whole clip
    cmap = colormaps["viridis"]

    fig = plt.figure(figsize=(16, 9), dpi=A.dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.7, 0.5, 0.5], height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.12,
                          left=0.01, right=0.94, top=0.9, bottom=0.07)

    # --- 3D view following the robot ---
    ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
    ax.plot_wireframe(Xw, Yw, gt, rstride=2, cstride=2, color="0.4", linewidth=0.5, alpha=0.7)
    m = seen.ravel()
    ax.scatter(Xw.ravel()[m], Yw.ravel()[m], est.ravel()[m], c=est.ravel()[m], cmap=cmap, norm=norm,
               s=9, marker="s", linewidths=0, depthshade=False, alpha=0.95)
    if hits_in.any():
        h = hits[hits_in]
        ax.scatter(h[:, 0], h[:, 1], h[:, 2], s=1.2, c="#d62728", alpha=0.5, depthshade=False)
    body = d["body_pos"][k]
    ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=16, c="k", depthshade=False)
    half = max(cx1 - cx0, cy1 - cy0) / 2 + 0.3
    mx, my = c * (cx0 + cx1) / 2 + bp[0], s * (cx0 + cx1) / 2 + bp[1]
    ax.set_xlim(mx - half, mx + half); ax.set_ylim(my - half, my + half); ax.set_zlim(zlo, zhi + 0.9)
    ax.set_box_aspect((1, 1, (A.zspan + 0.9) / (2 * half)), zoom=1.3)
    ax.view_init(elev=30, azim=np.degrees(yaw) - 145)
    ax.set_axis_off()
    ax.set_title("neural map cells (colour = height) on the true terrain (grey mesh)\nred: current lidar scan, black: robot links",
                 fontsize=12)

    # --- top-down truth / estimate (robot frame: forward = up, left = left) ---
    ext = [cy1, cy0, cx0, cx1]
    for col, (img, name) in enumerate([(gt, "true terrain height"), (est_m, "neural map height")]):
        a = fig.add_subplot(gs[0:2, 1 + col])
        im = a.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
        a.plot(0, 0, marker="^", ms=10, color="w", mec="k")
        a.set_title(name, fontsize=11); a.set_xticks([]); a.set_yticks([])
        a.set_facecolor("0.88")
    cb = fig.colorbar(im, ax=fig.axes[-2:], location="right", fraction=0.04, pad=0.02)
    cb.set_label("z [m]", fontsize=9); cb.ax.tick_params(labelsize=8)

    # --- elevation profile along the heading (centre row) ---
    a = fig.add_subplot(gs[2, 1:])
    j = int(np.argmin(np.abs(ys)))
    a.plot(xs, gt[:, j], color="0.2", lw=2.2, label="true terrain")
    a.plot(xs, est_m[:, j], color="#1f77b4", lw=1.8, marker=".", ms=3, label="neural map")
    mp = hits_in & (np.abs(hy) < 0.06)
    a.scatter(hx[mp], hits[mp, 2], s=4, c="#d62728", alpha=0.7, label="lidar (|y| < 6 cm)", zorder=3)
    a.axvline(0, color="k", lw=0.8, ls=":")
    a.set_xlim(cx0, cx1); a.set_ylim(zlo, zhi)
    a.set_xlabel("distance ahead of the robot [m]", fontsize=9); a.set_ylabel("z [m]", fontsize=9)
    a.tick_params(labelsize=8); a.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.8)
    a.set_title("elevation profile along the heading", fontsize=11)

    stats = (f"map vs. truth on observed cells: {100 * float(np.median(err)):.1f} cm median, "
             f"{100 * float((err < 0.05).mean()):.0f}% within 5 cm" if err.size else "map: no cells observed yet")
    fig.suptitle(f"{A.title}    t = {d['t'][k]:4.1f} s    {stats}", fontsize=14)
    out = os.path.join(A.tmp, f"f{k:05d}.png")
    fig.savefig(out); plt.close(fig)
    return out


def main():
    global D, A
    p = argparse.ArgumentParser()
    p.add_argument("npz"); p.add_argument("--out", required=True); p.add_argument("--title", default="")
    p.add_argument("--std_max", type=float, default=2.5,
                   help="cells with std below this count as observed (never-fused cells keep the initial std 3.16)")
    p.add_argument("--zspan", type=float, default=2.6, help="height range shown, centred on the ground under the robot")
    p.add_argument("--base_height", type=float, default=0.74, help="nominal base height over the ground")
    p.add_argument("--crop_x", type=float, nargs=2, default=(-1.0, 2.5)); p.add_argument("--crop_y", type=float, nargs=2, default=(-1.2, 1.2))
    p.add_argument("--dpi", type=int, default=120); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    A = p.parse_args()
    D = _load(A.npz)
    # clip-wide colour scale: robust range of the true terrain inside the crop
    x0, x1 = D["x_range"]; y0, y1 = D["y_range"]; res = float(D["res"])
    xs = np.arange(x0, x1 + 1e-6, res); ys = np.arange(y0, y1 + 1e-6, res)
    ix = (xs >= A.crop_x[0] - 1e-6) & (xs <= A.crop_x[1] + 1e-6); iy = (ys >= A.crop_y[0] - 1e-6) & (ys <= A.crop_y[1] + 1e-6)
    g = D["gt"][:, ix][:, :, iy]
    g = g[np.isfinite(g) & (g > -50)]
    A.clim = tuple(np.percentile(g, [1, 99])) if g.size else (-1.0, 1.0)
    A.tmp = tempfile.mkdtemp(prefix="nmap_frames_")
    n = len(D["t"])
    with Pool(A.workers) as pool:
        pool.map(_frame, range(n))
    fps = A.speed / float(D["dt"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", os.path.join(A.tmp, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", A.out], check=True)
    shutil.rmtree(A.tmp)
    print(f"[NMAP] wrote {A.out} ({n} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
