"""Trajectory text codec, PCHIP upsampling and trajectory metrics for this code base.

Conventions
-----------
* Ego frame, meters: current ego position at the origin, +x forward, +y left.
* Model-facing representation: 5 waypoints at t = 1, 2, 3, 4, 5 s, one decimal each, written as
  ``"[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]"`` inside ``<traj>...</traj>``.
* Evaluation / submission grid = Waymo official: 4 Hz, 20 points at t = 0.25, 0.50, ..., 5.00 s
  (the grid used by the official rater-feedback scoring and by the ADE/FDE definition).
* Upsampling 5 -> 20 uses shape-preserving PCHIP (``scipy.interpolate.PchipInterpolator``) per
  coordinate over the knots t = [-0.5, -0.25, 0, 1, 2, 3, 4, 5]: the last two history points, the
  origin and the 5 predicted waypoints. PCHIP passes through every knot and stays monotone
  between them, so it introduces neither overshoot nor backward-motion artefacts. Evaluation and
  submission must both go through :func:`upsample_pchip`.

Everything here is pure numpy/scipy; no file IO, no torch.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from training.core.paths import DT, FUT_STEPS

Point = Tuple[float, float]

# ----------------------------------------------------------------------------------------------
# Grid constants
# ----------------------------------------------------------------------------------------------
N_WAYPOINTS = 5
#: Official output grid, t = 0.25 .. 5.00 s (20 points).
TRAJ_T = np.arange(1, FUT_STEPS + 1, dtype=float) * DT
#: Times of the 5 text waypoints, t = 1 .. 5 s.
WAYPOINT_T = np.arange(1, N_WAYPOINTS + 1, dtype=float)
#: Indices into the 20-point GT future that fall on t = 1..5 s -> [3, 7, 11, 15, 19].
GT_1HZ_IDX = [int(round(t / DT)) - 1 for t in WAYPOINT_T]
#: Times of the two history anchors used by the upsampler (t = -0.5, -0.25 s).
HIST_ANCHOR_T = (-2.0 * DT, -1.0 * DT)
#: Steps counted by ADE@1s / ADE@3s / ADE@5s (ceil(h / DT)).
HORIZON_STEPS = {"ade1": 4, "ade3": 12, "ade5": FUT_STEPS}
#: Waypoint-weight formula constants: w_i = BASE * clip(1 + d_i / SCALE, 1, CLIP_MAX).
TRAJ_WEIGHT_BASE = 3.0
TRAJ_WEIGHT_SCALE_M = 2.0
TRAJ_WEIGHT_CLIP = (1.0, 3.0)


# ----------------------------------------------------------------------------------------------
# Text encoding
# ----------------------------------------------------------------------------------------------
def _fmt1(v: float) -> str:
    """Format one coordinate with one decimal, normalising ``-0.0`` to ``0.0``."""
    r = round(float(v), 1)
    if r == 0.0:  # also catches -0.0
        r = 0.0
    return f"{r:.1f}"


def encode_point(pt: Sequence[float]) -> str:
    """One waypoint -> ``"[x, y]"`` (one decimal each)."""
    return f"[{_fmt1(pt[0])}, {_fmt1(pt[1])}]"


def encode_traj_segments(points5: Iterable[Sequence[float]]) -> List[str]:
    """Per-waypoint text segments whose concatenation equals :func:`encode_traj_text`.

    Segment 0 is ``"[x1, y1]"``; segment i>0 is ``", [xi, yi]"`` (separator attached to the
    waypoint it introduces). Intended for ``target_builder`` so each waypoint's tokens can carry
    their own loss weight (``waypoint_weights[i] / 3.0``) while the joined text stays byte-identical.
    """
    pts = [tuple(p) for p in points5]
    if not pts:
        raise ValueError("encode_traj_segments: need at least one waypoint")
    return [encode_point(p) if i == 0 else ", " + encode_point(p) for i, p in enumerate(pts)]


def encode_traj_text(points5: Iterable[Sequence[float]]) -> str:
    """5 waypoints -> ``"[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]"`` (one decimal).

    Accepts any (n, 2) iterable (expected n = 5, the t = 1..5 s points from
    ``labels.traj_points_1hz``). Values are rounded with ``round(v, 1)``; ``-0.0`` is written as
    ``0.0``. Raises ``ValueError`` on an empty input.
    """
    return "".join(encode_traj_segments(points5))


# ----------------------------------------------------------------------------------------------
# Text parsing
# ----------------------------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"
_PAIR_RE = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_NUM_RE = re.compile(_NUM)
_TRAJ_TAG_RE = re.compile(r"<traj>(.*?)(?:</traj>|$)", re.S)


def parse_traj_text(s: Optional[str], n_points: int = N_WAYPOINTS) -> Optional[List[Point]]:
    """Parse the text inside ``<traj>`` into exactly ``n_points`` ``(x, y)`` tuples, or ``None``.

    Accepted input is either a bare waypoint list (``"[x1, y1], ..., [x5, y5]"``, e.g. the body a
    caller already cut out of the tag, or the output of :func:`encode_traj_text`) or a text that
    contains ``<traj>...</traj>`` (a missing closing tag is fine), in which case only the tag body
    is read. Tolerant to: surrounding/extra whitespace and newlines, integers without a decimal
    point, signed values, and more than ``n_points`` pairs (the first ``n_points`` are used).

    Missing or unbalanced brackets are tolerated **only inside a ``<traj>`` body** (consecutive
    numbers are then paired up: ``"<traj>0.4, 0.0, 2.9, 0.0, ...</traj>"``). Without the tag the
    bracket pairs are the only thing that counts -- bare numbers in a tag-less text are never
    paired, so the ``<point>x,y</point>`` / ``<rank>`` digits of a chain answer that simply has no
    ``<traj>`` segment can not turn into a bogus trajectory.

    Returns ``None`` when the input is ``None``/empty, contains fewer than ``n_points`` numeric
    pairs, or holds non-numeric tokens where numbers are expected (the "invalid trajectory"
    case counted by ``pred_rate``).
    """
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    m = _TRAJ_TAG_RE.search(s)
    in_tag = m is not None
    body = (m.group(1) if in_tag else s).strip()
    if not body:
        return None
    pairs = _PAIR_RE.findall(body)
    if len(pairs) < n_points and in_tag:
        # bare-number pairing fallback: only for the text between <traj> and </traj>
        nums = _NUM_RE.findall(body)
        pairs = list(zip(nums[0::2], nums[1::2]))
    if len(pairs) < n_points:
        return None
    try:
        out = [(float(a), float(b)) for a, b in pairs[:n_points]]
    except ValueError:
        return None
    if not all(math.isfinite(v) for p in out for v in p):
        return None
    return out


# ----------------------------------------------------------------------------------------------
# Frame helpers (past_states -> history / velocity)
# ----------------------------------------------------------------------------------------------
def hist_xy_from_frame(frame: Optional[dict]) -> Optional[List[Point]]:
    """``frame['past_states'].pos_x/pos_y`` -> list of (x, y) (normally 16 points, last = origin).

    Returns ``None`` when the frame has no usable past positions (the upsampler then degrades to
    the anchor-free knot set).
    """
    ps = (frame or {}).get("past_states") or {}
    px, py = ps.get("pos_x") or [], ps.get("pos_y") or []
    n = min(len(px), len(py))
    if n == 0:
        return None
    return [(float(px[i]), float(py[i])) for i in range(n)]


def ego_velocity(frame: Optional[dict]) -> Point:
    """Current ego velocity ``(vx, vy)`` in the ego frame = ``past_states.vel_*[-1]``.

    Falls back to the finite difference of the last two past positions over ``DT`` when the
    velocity arrays are missing, and to ``(0, 0)`` when there is no history at all.
    """
    ps = (frame or {}).get("past_states") or {}
    vx, vy = ps.get("vel_x") or [], ps.get("vel_y") or []
    if vx and vy:
        return float(vx[-1]), float(vy[-1])
    px, py = ps.get("pos_x") or [], ps.get("pos_y") or []
    if len(px) >= 2 and len(py) >= 2:
        return (float(px[-1]) - float(px[-2])) / DT, (float(py[-1]) - float(py[-2])) / DT
    return 0.0, 0.0


def cv_from_velocity(vx: float, vy: float) -> np.ndarray:
    """Constant-velocity rollout from the origin on the official grid -> (20, 2)."""
    return np.stack([float(vx) * TRAJ_T, float(vy) * TRAJ_T], axis=1)


def cv_extrapolation(frame: Optional[dict]) -> np.ndarray:
    """(20, 2) constant-velocity extrapolation from the origin using ``past_states.vel_*[-1]``.

    p_i = v0 * t_i for t_i = 0.25 .. 5.0 s. This is the "copy the momentum" baseline used as the
    reference for the waypoint information weights and as a constant-velocity row in evaluation.
    """
    return cv_from_velocity(*ego_velocity(frame))


# ----------------------------------------------------------------------------------------------
# Upsampling 5 waypoints -> 20 points (PCHIP + history anchors)
# ----------------------------------------------------------------------------------------------
def _hist_anchors(hist_xy16) -> Optional[List[Point]]:
    """History knots for t = -0.5 s and -0.25 s = ``hist[-3]``, ``hist[-2]`` (``hist[-1]`` is the origin).

    Returns ``None`` when the history is missing, shorter than 3 points, or non-finite.
    """
    if hist_xy16 is None:
        return None
    try:
        h = np.asarray(hist_xy16, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if h.shape[0] < 3:
        return None
    anchors = h[-3:-1]
    if not np.all(np.isfinite(anchors)):
        return None
    return [(float(anchors[0, 0]), float(anchors[0, 1])), (float(anchors[1, 0]), float(anchors[1, 1]))]


def upsample_pchip(points5, hist_xy16=None, t_out: Optional[Sequence[float]] = None) -> np.ndarray:
    """5 waypoints @1 Hz (+ history) -> (20, 2) trajectory on the official 4 Hz grid.

    Knots: t = [-0.5, -0.25, 0, 1, 2, 3, 4, 5] with values ``hist[-3]``, ``hist[-2]``, the origin
    and the waypoints; ``x(t)`` and ``y(t)`` are each interpolated with ``PchipInterpolator``
    (strictly through the knots, monotone between them -- no overshoot / reversing artefacts),
    then sampled at t = 0.25 .. 5.0 s.

    Parameters
    ----------
    points5 : (k, 2) array-like -- waypoint i is at t = i + 1 s (k = 5 normally). If k < 5 the
        curve is held at the last waypoint beyond its time (short predictions are penalised
        rather than extrapolated).
    hist_xy16 : (n>=3, 2) array-like of past positions ending at the origin, or ``None``. When
        missing/unusable the knot set degrades to t = [0, 1, ..., k] without anchors.
    t_out : optional sample times (default :data:`TRAJ_T`).

    Returns
    -------
    np.ndarray of shape (len(t_out), 2), float64.
    """
    pts = np.asarray(points5, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        raise ValueError("upsample_pchip: need at least one waypoint")
    if not np.all(np.isfinite(pts)):
        raise ValueError("upsample_pchip: non-finite waypoint")
    k = pts.shape[0]
    t_nodes: List[float] = [0.0] + [float(i + 1) for i in range(k)]
    xy_nodes: List[Point] = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in pts]
    anchors = _hist_anchors(hist_xy16)
    if anchors is not None:
        t_nodes = list(HIST_ANCHOR_T) + t_nodes
        xy_nodes = anchors + xy_nodes
    t_arr = np.asarray(t_nodes, dtype=float)
    xy_arr = np.asarray(xy_nodes, dtype=float)
    t_eval = TRAJ_T if t_out is None else np.asarray(t_out, dtype=float)
    t_eval = np.clip(t_eval, t_arr[0], t_arr[-1])  # hold ends instead of extrapolating
    out = np.empty((t_eval.shape[0], 2), dtype=float)
    for c in range(2):
        out[:, c] = PchipInterpolator(t_arr, xy_arr[:, c], extrapolate=False)(t_eval)
    return out


def traj_text_to_xy20(s: Optional[str], hist_xy16=None) -> Optional[np.ndarray]:
    """Convenience: ``parse_traj_text`` then ``upsample_pchip``; ``None`` if the text is invalid."""
    pts = parse_traj_text(s)
    if pts is None:
        return None
    return upsample_pchip(pts, hist_xy16)


def points_1hz(xy20) -> List[Point]:
    """Pick the t = 1..5 s points (indices 3, 7, 11, 15, 19) out of a (>=20, 2) trajectory, unrounded.

    ``labels.traj_points_1hz(frame)`` is the rounded, frame-facing version; this operates on arrays
    (GT futures, upsampled predictions) and is used by the evaluation scripts.
    """
    xy = np.asarray(xy20, dtype=float).reshape(-1, 2)
    if xy.shape[0] < FUT_STEPS:
        raise ValueError(f"points_1hz: need {FUT_STEPS} points, got {xy.shape[0]}")
    return [(float(xy[i, 0]), float(xy[i, 1])) for i in GT_1HZ_IDX]


# ----------------------------------------------------------------------------------------------
# Waypoint loss weights
# ----------------------------------------------------------------------------------------------
def waypoint_weights(points5, frame: Optional[dict]) -> List[float]:
    """Information weights for the 5 text waypoints.

    ``w_i = 3.0 * clip(1 + d_i / 2.0 m, 1, 3)`` where ``d_i`` is the distance between GT waypoint i
    (t = i + 1 s) and the constant-velocity extrapolation ``v0 * (i + 1)`` from
    ``past_states.vel_*[-1]``. Waypoints that merely copy the momentum get the base weight 3.0
    (= the ``traj`` field weight); start/brake/turn waypoints get up to 9.0. ``target_builder``
    uses ``waypoint_weights[i] / 3.0`` as the per-waypoint multiplier on top of ``W['traj']``.
    Returns one weight per input point (5 for a normal label).
    """
    pts = np.asarray(points5, dtype=float).reshape(-1, 2)
    vx, vy = ego_velocity(frame)
    lo, hi = TRAJ_WEIGHT_CLIP
    out: List[float] = []
    for i, (x, y) in enumerate(pts):
        t = float(i + 1)
        d = math.hypot(x - vx * t, y - vy * t)
        out.append(float(TRAJ_WEIGHT_BASE * min(max(1.0 + d / TRAJ_WEIGHT_SCALE_M, lo), hi)))
    return out


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def traj_ade_fde(pred20, gt20) -> Tuple[float, float, float, float]:
    """Official ADE@1s / ADE@3s / ADE@5s / FDE@5s for one frame on the 4 Hz grid.

    ``ADE@h`` = mean L2 error over the first ``ceil(h / 0.25)`` steps (4 / 12 / 20);
    ``FDE@5s`` = L2 error at step 20. A prediction shorter than the GT is padded with its last
    point (short outputs are penalised, not discarded); if the GT is shorter than
    20 points the horizons are truncated to what exists. Returns ``(nan, nan, nan, nan)`` when
    either input is empty so callers can filter with ``math.isnan`` without exceptions.
    """
    pred = np.asarray(pred20, dtype=float).reshape(-1, 2) if pred20 is not None else np.zeros((0, 2))
    gt = np.asarray(gt20, dtype=float).reshape(-1, 2) if gt20 is not None else np.zeros((0, 2))
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    if pred.shape[0] < gt.shape[0]:
        pad = np.repeat(pred[-1:], gt.shape[0] - pred.shape[0], axis=0)
        pred = np.concatenate([pred, pad], axis=0)
    n = min(pred.shape[0], gt.shape[0])
    d = np.hypot(pred[:n, 0] - gt[:n, 0], pred[:n, 1] - gt[:n, 1])
    k1, k3, k5 = (min(HORIZON_STEPS[h], n) for h in ("ade1", "ade3", "ade5"))
    return float(d[:k1].mean()), float(d[:k3].mean()), float(d[:k5].mean()), float(d[k5 - 1])


def step_speeds(xy) -> np.ndarray:
    """Per-step speeds |p_i - p_{i-1}| / DT of a trajectory starting from the origin (p_{-1} = 0)."""
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    prev = np.vstack([np.zeros((1, 2)), p[:-1]])
    return np.hypot(p[:, 0] - prev[:, 0], p[:, 1] - prev[:, 1]) / DT


def min_forward_step(xy) -> float:
    """Smallest per-step x-displacement (first step relative to the origin); < -tol means "reversed"."""
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    prev = np.vstack([np.zeros((1, 2)), p[:-1]])
    return float((p[:, 0] - prev[:, 0]).min())


__all__ = [
    "N_WAYPOINTS", "TRAJ_T", "WAYPOINT_T", "GT_1HZ_IDX", "HIST_ANCHOR_T", "HORIZON_STEPS",
    "TRAJ_WEIGHT_BASE", "TRAJ_WEIGHT_SCALE_M", "TRAJ_WEIGHT_CLIP",
    "encode_point", "encode_traj_segments", "encode_traj_text", "parse_traj_text",
    "hist_xy_from_frame", "ego_velocity", "cv_from_velocity", "cv_extrapolation",
    "upsample_pchip", "traj_text_to_xy20", "points_1hz", "waypoint_weights",
    "traj_ade_fde", "step_speeds", "min_forward_step",
]
