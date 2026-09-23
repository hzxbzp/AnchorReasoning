"""Shared rendering for the per-frame trajectory + reasoning visualizations.

Layout of one figure:
  - main axes  : panorama (FRONT_LEFT | FRONT | FRONT_RIGHT), with the ground-truth (green)
                 and the predicted (red) trajectory projected onto the road.
  - top center : intent label (large).
  - top-left   : text box = model name + scene/frame + ADE + chain of thought.
  - top-right  : BEV inset (history / GT / prediction, x forward, y left).

Projection uses one fixed Waymo calibration for every frame, because the decoded validation
``frame.json`` carries no per-frame calibration while the Waymo sensor rig is constant across
scenes. ``WAYMO_CALIB_JSON`` points at a JSON file mapping camera name to
``{"intrinsic": [fx, fy, cx, cy, k1, k2, p1, p2, k3], "extrinsic": <4x4 camera-to-vehicle>}``;
that block can be taken from the ``camera_calibrations`` of any decoded training frame.json.
"""
from __future__ import annotations
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

from training.core.paths import DATA

# left-to-right camera order in the stitched panorama
PANORAMA_CAMERAS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")
CALIB_JSON = os.environ.get("WAYMO_CALIB_JSON", os.path.join(DATA, "calib_fixed.json"))

# rater-score -> color (red=low .. green=high), matches the RFS 0-10 scale
try:
    _SCORE_CMAP = matplotlib.colormaps["RdYlGn"]        # matplotlib >= 3.5
except AttributeError:                                  # older matplotlib
    import matplotlib.cm as _cm
    _SCORE_CMAP = _cm.get_cmap("RdYlGn")
_SCORE_NORM = Normalize(vmin=0.0, vmax=10.0)


def score_color(s: float):
    return _SCORE_CMAP(_SCORE_NORM(float(s)))


def load_calib(path: str | Path = CALIB_JSON) -> tuple[dict, dict]:
    """Return (intrinsics{cam: 9-vector}, extrinsics{cam: 4x4}) as numpy."""
    c = json.load(Path(path).open())
    intr, extr = {}, {}
    for cam, v in c.items():
        ex = v["extrinsic"]
        if isinstance(ex, dict):          # frame.json stores it as {"transform": [...16]}
            ex = ex["transform"]
        intr[cam] = np.asarray(v["intrinsic"], dtype=np.float64)
        extr[cam] = np.asarray(ex, dtype=np.float64).reshape(4, 4)
    return intr, extr


def project_points_to_panorama(points_xyz, intr: dict, extr: dict, width: int, height: int) -> np.ndarray:
    """Project ego-frame points (N,3) onto the panorama; returns (N,2) pixels, NaN when out of view.

    The panorama is the three forward cameras side by side, so each point is projected into every
    camera and kept in the strip where it sits closest to the strip centre. Vehicle frame is
    x forward / y left / z up, the extrinsic is camera-to-vehicle and the camera frame has x out
    of the lens, so depth is the camera-frame x component.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(pts), 2), np.nan)
    best = np.full(len(pts), np.inf)          # distance to the strip centre of the chosen camera
    strip_w = width / len(PANORAMA_CAMERAS)
    for i, cam in enumerate(PANORAMA_CAMERAS):
        if cam not in intr or cam not in extr:          # calibration without this camera
            continue
        K, T = intr[cam], extr[cam]
        R, t = T[:3, :3], T[:3, 3]
        cam_pts = (pts - t) @ R               # rows of R^T @ (p - t): vehicle -> camera
        depth = cam_pts[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0] * (-cam_pts[:, 1] / depth) + K[2]
            v = K[1] * (-cam_pts[:, 2] / depth) + K[3]
        d = np.abs(u - strip_w / 2.0)
        take = (depth > 0.5) & (u >= 0) & (u < strip_w) & (v >= 0) & (v < height) & (d < best)
        out[take, 0] = u[take] + i * strip_w
        out[take, 1] = v[take]
        best[take] = d[take]
    return out


def gt_future_xyz(frame_json: dict) -> np.ndarray:
    """(T,3) ego-frame GT future from frame.json future_states."""
    fs = frame_json["future_states"]
    return np.stack([np.asarray(fs["pos_x"]), np.asarray(fs["pos_y"]),
                     np.asarray(fs["pos_z"])], axis=1).astype(np.float64)


def pref_trajectories(frame_json: dict) -> list[tuple[np.ndarray, float]]:
    """All rater preference trajectories as [(xyz (N,3), score), ...].

    They store pos_x/pos_y (+ preference_score) but no pos_z -> ground (z=0)."""
    out = []
    for t in frame_json.get("preference_trajectories", []) or []:
        x = np.asarray(t["pos_x"], dtype=np.float64)
        y = np.asarray(t["pos_y"], dtype=np.float64)
        z = np.zeros_like(x)
        out.append((np.stack([x, y, z], axis=1), float(t.get("preference_score", float("nan")))))
    return out


def hist_xyz(frame_json: dict) -> np.ndarray:
    ps = frame_json["past_states"]
    x = np.asarray(ps["pos_x"], dtype=np.float64)
    y = np.asarray(ps["pos_y"], dtype=np.float64)
    z = np.asarray(ps.get("pos_z", np.zeros_like(x)), dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def _draw_proj(ax, px, color, label, anchor_idx=None, anchor_t=None):
    """Draw a projected polyline+points, skipping NaN (out-of-view) points.

    ``anchor_idx``: indices of the points the model actually emitted (the 5 waypoints at
    t = 1..5 s); the remaining points are PCHIP-interpolated. Anchors are drawn as large filled
    markers with a time label, interpolated points as small hollow markers."""
    valid = ~np.isnan(px[:, 0])
    if valid.sum() == 0:
        return
    if anchor_idx is None:
        ax.plot(px[valid, 0], px[valid, 1], "-", color=color, lw=3.5, alpha=0.9, label=label)
        ax.scatter(px[valid, 0], px[valid, 1], c=color, s=22, edgecolors="black",
                   linewidths=0.5, zorder=5)
        return
    anchors = np.zeros(len(px), dtype=bool); anchors[list(anchor_idx)] = True
    ax.plot(px[valid, 0], px[valid, 1], "-", color=color, lw=2.2, alpha=0.85, label=label)
    interp = valid & ~anchors
    if interp.sum():
        ax.scatter(px[interp, 0], px[interp, 1], facecolors="none", edgecolors=color, s=34,
                   linewidths=1.3, zorder=5, label="interpolated (PCHIP, 0.25 s)")
    anc = valid & anchors
    if anc.sum():
        ax.scatter(px[anc, 0], px[anc, 1], c=color, s=150, edgecolors="white", linewidths=1.6,
                   zorder=7, marker="o", label="predicted waypoints (t = 1..5 s)")
        for k, i in enumerate(list(anchor_idx)):
            if valid[i]:
                t = anchor_t[k] if anchor_t is not None else k + 1
                ax.annotate(f"{t:g}s", (px[i, 0], px[i, 1]), xytext=(7, 7), textcoords="offset points",
                            fontsize=11, fontweight="bold", color="white", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="white", lw=0.8, alpha=0.9))


def _bev(ax, hist, gt, pred, prefs=None, anchor_idx=None):
    if hist is not None:
        ax.plot(hist[:, 0], hist[:, 1], "-", color="white", lw=1.5, alpha=0.7)
    for traj, sc in (prefs or []):
        ax.plot(traj[:, 0], traj[:, 1], "--", color=score_color(sc), lw=1.4, alpha=0.85)
    ax.plot(gt[:, 0], gt[:, 1], "-", color="lime", lw=2.2, label="GT")
    ax.scatter([0], [0], c="cyan", marker="*", s=120, zorder=6, edgecolors="k")
    if pred is not None:
        ax.plot(pred[:, 0], pred[:, 1], "-", color="red", lw=1.8, label="pred (interp.)")
        if anchor_idx is not None:
            ai = list(anchor_idx)
            ax.scatter(pred[ai, 0], pred[ai, 1], c="red", s=42, edgecolors="white", linewidths=0.9,
                       zorder=7, label="pred waypoints")
    pts = [gt] + ([pred] if pred is not None else []) + ([hist] if hist is not None else [])
    pts += [t for t, _ in (prefs or [])]
    allp = np.concatenate(pts, 0)
    r = max(np.abs(allp[:, :2]).max() * 1.1, 5.0)
    ax.set_xlim(-2, r); ax.set_ylim(-r * 0.6, r * 0.6)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=6)
    for s in ax.spines.values():
        s.set_color("white")
    ax.set_title("BEV (x↑ fwd, y← left)", color="white", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, facecolor="black", labelcolor="white", framealpha=0.5)


def make_figure(out_path, pano_path, gt_ego, pred_ego, hist_ego,
                cot_text, intent, header_lines, intr, extr, prefs=None, pred_anchor_idx=None):
    """Render one composite figure. pred_ego may be None (GT-only).

    prefs: list of (xyz (N,3), score) rater preference trajectories — all are
    projected, colored by rater score (red=low .. green=high)."""
    prefs = prefs or []
    pano = np.array(Image.open(pano_path).convert("RGB"))
    H, W = pano.shape[:2]

    gt_px = project_points_to_panorama(gt_ego, intr, extr, W, H)
    pred_px = (project_points_to_panorama(pred_ego, intr, extr, W, H)
               if pred_ego is not None else None)

    fig = plt.figure(figsize=(20, 20 * H / W))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(pano)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()

    # rater preference trajectories first (under GT/pred), colored by score
    for traj, sc in prefs:
        px = project_points_to_panorama(traj, intr, extr, W, H)
        v = ~np.isnan(px[:, 0])
        if v.sum():
            ax.plot(px[v, 0], px[v, 1], "--", color=score_color(sc), lw=2.5, alpha=0.9,
                    label=f"rater s={sc:.0f}")
    _draw_proj(ax, gt_px, "lime", "GT logged (5 s)")
    if pred_px is not None:
        _draw_proj(ax, pred_px, "red", "prediction", anchor_idx=pred_anchor_idx)
    ax.legend(loc="lower right", fontsize=13, framealpha=0.85)

    # intent (top center)
    ax.text(0.5, 0.965, f"intent: {intent}", transform=ax.transAxes,
            ha="center", va="top", fontsize=26, fontweight="bold", color="yellow",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    # text box top-left: header + chain-of-thought
    wrapped = "\n".join(textwrap.fill(line, 60) for line in cot_text.splitlines())
    txt = "\n".join(header_lines) + "\n— reasoning —\n" + wrapped
    ax.text(0.008, 0.985, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color="white", family="monospace",
            bbox=dict(boxstyle="round", fc="black", alpha=0.62))

    # BEV inset top-right
    bx = fig.add_axes([0.74, 0.60, 0.245, 0.38])
    _bev(bx, hist_ego, gt_ego, pred_ego, prefs, anchor_idx=pred_anchor_idx)

    fig.savefig(out_path, dpi=90, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
