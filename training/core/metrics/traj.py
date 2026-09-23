"""Trajectory metrics (Waymo official convention) + trajectory-derived motion classes.

Inputs are already on the official 4 Hz / 20-step grid (t = 0.25 ... 5.0 s, ego frame, metres):
the evaluation scripts up-sample the model's 5 waypoints with ``traj_codec.upsample_pchip`` and
read the GT from ``frame.future_states``. This module therefore does no interpolation.

  ADE@h = mean L2 over the first ceil(h / 0.25) steps, h in {1, 3, 5} s (steps 4 / 12 / 20)
  FDE@5s = L2 at step 20
  traj_score = mean_over_frames exp(-ADE@5s / 2 m)   (a missing prediction scores 0)

Buckets: by GT ``sample_class`` (stay / start / stop / decel / accel / keep, identical to
``labels.sample_class``) and by current speed ``v0``.

The longitudinal class rules live here (pure functions of a 20-point trajectory + v0) so that
``metrics.chain`` can classify *predicted* trajectories with exactly the same code; the lateral
proxy reuses the heading profile of ``core.labels`` (same skip / reversal-truncation rules as
``labels.lat_class``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from training.core import labels as L
from training.core.paths import DT, FUT_STEPS

__all__ = [
    "HORIZONS", "SAMPLE_CLASSES", "MOTION_LON_CLASSES", "V0_BUCKETS",
    "as_xy20", "speeds_from_xy", "sample_class_from_xy", "motion_lon_from_xy", "motion_lat_from_xy",
    "v0_bucket", "ade_fde", "traj_metrics",
]

HORIZONS: Dict[str, int] = {"ade@1s": 4, "ade@3s": 12, "ade@5s": 20}   # ceil(h / 0.25)
KEYS = ["ade@1s", "ade@3s", "ade@5s", "fde@5s"]
SAMPLE_CLASSES = ["stay", "start", "stop", "decel", "accel", "keep"]
MOTION_LON_CLASSES = ["accelerate", "decelerate", "keep", "stop"]
V0_BUCKETS = ["v0<0.5", "0.5<=v0<3", "3<=v0<8", "v0>=8"]
STOP_V = 0.5          # m/s: "stopped" threshold
START_VMAX = 1.0      # m/s: max speed that makes a stopped ego a "start"
TURN_DEG = L.TURN_DEG   # 25 deg: heading change that counts as a turn (single source: labels)


# --------------------------------------------------------------------------- helpers
def as_xy20(xy: Any) -> Optional[np.ndarray]:
    """Coerce a prediction/GT into a float (20, 2) array.

    ``None`` / empty -> ``None``. Shorter sequences are padded with their last point, so a short
    prediction is penalised; longer ones are truncated. Non-finite -> ``None``.
    """
    if xy is None:
        return None
    a = np.asarray(xy, dtype=np.float64)
    if a.size == 0:
        return None
    a = a.reshape(-1, 2)
    if not np.all(np.isfinite(a)):
        return None
    if a.shape[0] < FUT_STEPS:
        a = np.concatenate([a, np.repeat(a[-1:], FUT_STEPS - a.shape[0], axis=0)], axis=0)
    return a[:FUT_STEPS]


def speeds_from_xy(xy: Sequence[Sequence[float]], dt: float = DT) -> List[float]:
    """Per-step speeds (m/s) of a trajectory starting at the origin: |p_i - p_{i-1}| / dt, p_{-1} = (0, 0)."""
    P = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in xy]
    return [math.hypot(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]) / dt for i in range(len(P) - 1)]


def _v_end(sp: Sequence[float]) -> float:
    k = min(4, len(sp))          # last 1 s at 4 Hz
    return float(sum(sp[-k:]) / k) if k else 0.0


def sample_class_from_xy(xy: Sequence[Sequence[float]], v0: float) -> str:
    """Fine motion class used for sampling / bucketing (== ``labels.sample_class``).

    stay:  v0 < 0.5 and max(v) < 0.5        start: v0 < 0.5 and max(v) >= 1.0
    stop:  v0 >= 0.5 and v_end < 0.5        decel: v_end < v0 - max(1, 0.2 v0)
    accel: v_end > v0 + max(1, 0.2 v0)      keep:  otherwise
    (v_end = mean speed of the last 1 s; v0 = current speed from history.)
    """
    sp = speeds_from_xy(xy)
    if not sp:
        return "keep"
    v_end, v_max = _v_end(sp), max(sp)
    v0 = float(v0)
    if v0 < STOP_V and v_max < STOP_V:
        return "stay"
    if v0 < STOP_V and v_max >= START_VMAX:
        return "start"
    if v0 >= STOP_V and v_end < STOP_V:
        return "stop"
    thr = max(1.0, 0.2 * v0)
    if v_end < v0 - thr:
        return "decel"
    if v_end > v0 + thr:
        return "accel"
    return "keep"


def motion_lon_from_xy(xy: Sequence[Sequence[float]], v0: float) -> str:
    """Closed-set longitudinal trend of a trajectory (== ``labels.motion_label()['lon']``).

    stop: v_end < 0.5; accelerate: (v0 < 0.5 and max(v) >= 1.0) or v_end - v0 > max(1, 0.2 v0);
    decelerate: v0 - v_end > max(1, 0.2 v0) and v_end >= 0.5; keep: otherwise.
    """
    sp = speeds_from_xy(xy)
    if not sp:
        return "keep"
    v_end, v_max = _v_end(sp), max(sp)
    v0 = float(v0)
    if v_end < STOP_V:
        return "stop"
    thr = max(1.0, 0.2 * v0)
    if (v0 < STOP_V and v_max >= START_VMAX) or (v_end - v0 > thr):
        return "accelerate"
    if (v0 - v_end > thr) and v_end >= STOP_V:
        return "decelerate"
    return "keep"


def motion_lat_from_xy(xy: Sequence[Sequence[float]]) -> str:
    """Heading-only lateral proxy: ``left_turn`` / ``right_turn`` when the heading change between the
    first and the last informative segment is >= 25 deg, else ``straight``.

    The heading profile is ``labels.heading_profile`` (the one behind ``labels.lat_class``): the
    origin is prepended, points closer than ``labels.MIN_STEP_M`` (0.3 m) to the previously kept
    point are skipped and the path is truncated at the first backing reversal (consecutive heading
    jump > ``labels.REVERSAL_DEG`` = 120 deg), so a trajectory that creeps forward and rolls back is
    ``straight`` rather than a 180 deg turn.  This is the *trajectory-derived* lateral class used for
    motion-trajectory consistency; the training label (``labels.motion_label``) additionally applies
    the intent / curvature-residual test.
    """
    pts = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in xy]
    theta, _resid = L.heading_profile(pts)
    if abs(theta) >= TURN_DEG:
        return "left_turn" if theta > 0 else "right_turn"
    return "straight"


def v0_bucket(v0: float) -> str:
    """Speed bucket name for ``v0`` (m/s)."""
    v0 = float(v0)
    if v0 < 0.5:
        return V0_BUCKETS[0]
    if v0 < 3.0:
        return V0_BUCKETS[1]
    if v0 < 8.0:
        return V0_BUCKETS[2]
    return V0_BUCKETS[3]


def ade_fde(pred20: Any, gt20: Any) -> Optional[Dict[str, float]]:
    """Official ADE@1/3/5 s + FDE@5 s for one frame (``None`` if either side is unusable)."""
    p, g = as_xy20(pred20), as_xy20(gt20)
    if p is None or g is None:
        return None
    d = np.hypot(p[:, 0] - g[:, 0], p[:, 1] - g[:, 1])
    out = {name: float(d[:k].mean()) for name, k in HORIZONS.items()}
    out["fde@5s"] = float(d[FUT_STEPS - 1])
    return out


def _mean(xs: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _summ(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-frame rows (``{'m': ade_fde dict|None, 'score': float}``)."""
    with_pred = [r for r in rows if r["m"] is not None]
    out: Dict[str, Any] = {"n": len(rows), "n_with_pred": len(with_pred),
                           "pred_rate": (len(with_pred) / len(rows)) if rows else None}
    for k in KEYS:
        out[k] = _mean(r["m"][k] for r in with_pred)
    out["traj_score"] = _mean(r["score"] for r in rows)
    out["traj_score_valid"] = _mean(r["score"] for r in with_pred)
    return out


def traj_metrics(pred20_list: Sequence[Any], gt20_list: Sequence[Any], v0_list: Sequence[float]) -> Dict[str, Any]:
    """Aggregate official trajectory metrics over frames.

    Args:
        pred20_list: per frame, a (20, 2) array-like in the ego frame or ``None`` (no valid prediction;
            the frame is counted in ``n_total`` / ``pred_rate`` and scores 0 in ``traj_score``).
        gt20_list: per frame, the GT (20, 2) future.
        v0_list: per frame, the current speed |vel[-1]| (m/s) used by the class rules and v0 buckets.

    Returns a dict with ``ade@1s ade@3s ade@5s fde@5s`` (means over frames with a prediction),
    ``n_total n_with_pred pred_rate``, ``traj_score`` (mean exp(-ADE5/2), missing -> 0) and
    ``traj_score_valid`` (same, valid frames only), ``by_class`` / ``by_v0`` (same summary per GT
    ``sample_class`` / v0 bucket), ``gt_class_dist``, ``pred_class_dist``, ``class_confusion``
    (``'gt->pred'`` counts, ``pred='none'`` when missing), ``class_acc`` (GT class reproduced by the
    predicted trajectory, over frames with a prediction) and ``start_pred_static_rate``
    (GT start frames whose prediction stays stopped).
    """
    n = len(gt20_list)
    if not (len(pred20_list) == n == len(v0_list)):
        raise ValueError("pred20_list, gt20_list and v0_list must have the same length")
    rows: List[Dict[str, Any]] = []
    gt_dist: Dict[str, int] = {c: 0 for c in SAMPLE_CLASSES}
    pred_dist: Dict[str, int] = {c: 0 for c in SAMPLE_CLASSES + ["none"]}
    confusion: Dict[str, int] = {}
    n_cls_ok = n_cls_tot = 0
    for pred, gt, v0 in zip(pred20_list, gt20_list, v0_list):
        g = as_xy20(gt)
        if g is None:
            continue
        p = as_xy20(pred)
        v0 = float(v0)
        gcls = sample_class_from_xy(g, v0)
        pcls = sample_class_from_xy(p, v0) if p is not None else "none"
        m = ade_fde(p, g) if p is not None else None
        score = math.exp(-m["ade@5s"] / 2.0) if m is not None else 0.0
        rows.append({"m": m, "score": score, "gcls": gcls, "pcls": pcls, "v0b": v0_bucket(v0)})
        gt_dist[gcls] += 1
        pred_dist[pcls] += 1
        confusion[f"{gcls}->{pcls}"] = confusion.get(f"{gcls}->{pcls}", 0) + 1
        if p is not None:
            n_cls_tot += 1
            n_cls_ok += int(pcls == gcls)

    out: Dict[str, Any] = _summ(rows)
    out["n_total"] = out.pop("n")
    out["by_class"] = {c: _summ([r for r in rows if r["gcls"] == c]) for c in SAMPLE_CLASSES}
    out["by_v0"] = {b: _summ([r for r in rows if r["v0b"] == b]) for b in V0_BUCKETS}
    out["gt_class_dist"] = gt_dist
    out["pred_class_dist"] = pred_dist
    out["class_confusion"] = confusion
    out["class_acc"] = (n_cls_ok / n_cls_tot) if n_cls_tot else None
    n_start = gt_dist["start"]
    out["start_pred_static_rate"] = (confusion.get("start->stay", 0) / n_start) if n_start else None
    return out
