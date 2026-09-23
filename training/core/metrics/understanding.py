"""Scene-understanding metrics (the S1 segment of S1/S2 answers).

Core metrics:
  context per-dim accuracy (loose match), events set-F1, presence accuracy, object recall
  (overall / impact_rank==1) by greedy same-type matching (nearest point to the GT SAM2 mask,
  bbox-centre fallback), localisation hit relaxed (dist-to-mask <= tau = clip(0.5*bbox_diag, 15, 60) px,
  inside the image) / strict (inside the mask), attribute loose accuracy, count consistency,
  implication (LLM judge or token-overlap proxy).  NOTE: ``implication`` is the cause-AND-effect
  conjunction; read ``implication_cause`` and ``implication_effect`` separately, since the
  conjunction hides which of the two axes failed.

Additional metrics:
  rank_acc / rank_acc_pm1 (matched objects), attribute slots — location split into direction
  (front / front left / front right) and lane relation (ego-lane / adjacent-same-dir / oncoming /
  cross-street / roadside), intention/state majority-collapse counts ("X -> cruising"), minority-class
  and pooled accuracies, direction-pixel consistency (direction implied by the predicted point's
  x / PANO_W: < 0.45 front left, > 0.55 front right, else front — versus the predicted location's
  direction), coarse-class recall (vehicle / VRU / control / other) that separates type errors from
  misses, count confusion, object precision, and ``composite`` = the S1 proxy composite.

Record format::

    {'pred': parse_output(...) dict (or raw text),  'pano': panorama_geo.json dict,
     'coco_anns': panorama_geo_sam2_coco.json['annotations'] (may be None/[]),
     'scale': (rw/ow, rh/oh) from adapter.image_inputs (None -> (1, 1)),
     'point_codec': adapter.point_codec (has .decode(x, y, scale) -> panorama px) or None
                    (fallback: divide by scale)}
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from training.core.paths import PANO_H, PANO_W
from training.core.metrics.composite import s1_composite
from training.core.parse_output import parse_output

__all__ = [
    "evaluate_understanding", "rle_to_mask", "bbox_match", "loose_eq", "location_slots",
    "direction_from_x", "coarse_class", "canon_attr", "Agg", "CTX_KEYS", "ATTR_FIELDS", "COARSE",
]

CTX_KEYS = ["weather", "daytime", "visibility", "scenario", "road"]
ATTR_FIELDS = ("location", "intention", "state", "content")
TAU_LO, TAU_HI = 15.0, 60.0
DIRECTIONS = ("front left", "front", "front right")
LANES = ("ego-lane", "adjacent-same-dir", "oncoming", "cross-street", "roadside")
MAJORITY = "cruising"
MAJORITY_SET = {"cruising", "stopping", "parking", "crossing"}      # everything else is a "minority" class
COARSE: Dict[str, Tuple[str, ...]] = {
    "vehicle": ("Car", "Truck", "Bus", "Construction vehicle", "Emergency vehicle", "School bus"),
    "vru": ("Pedestrian", "Cyclist", "Motorcyclist", "Scooter rider", "Animal"),
    "control": ("Traffic light", "Sign", "Stop line", "Cross walk", "Bump", "Temporary control"),
    "other": ("Object", "Other"),
}
COARSE_CLASSES = tuple(COARSE)
_COARSE_LOOKUP = {t.lower(): c for c, ts in COARSE.items() for t in ts}


# --------------------------------------------------------------------------- geometry helpers
def rle_to_mask(size: Sequence[int], counts: Any) -> Optional[np.ndarray]:
    """COCO uncompressed RLE (column-major) -> uint8 (h, w) mask; compressed (str/bytes) -> None."""
    if not size or counts is None:
        return None
    h, w = int(size[0]), int(size[1])
    if isinstance(counts, (str, bytes)):
        return None  # compressed RLE -> caller falls back to bbox center
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for c in counts:
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def bbox_match(pano_bbox: Dict[str, float], coco_anns: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Match a panorama_geo bbox {x,y,w,h} to the closest coco annotation (L1 on x,y,w,h)."""
    if not coco_anns:
        return None
    px, py, pw, ph = pano_bbox.get("x", 0), pano_bbox.get("y", 0), pano_bbox.get("w", 0), pano_bbox.get("h", 0)
    best, bd = None, 1e18
    for ca in coco_anns:
        b = ca.get("bbox") or [0, 0, 0, 0]
        d = abs(b[0] - px) + abs(b[1] - py) + abs(b[2] - pw) + abs(b[3] - ph)
        if d < bd:
            bd, best = d, ca
    return best


def _tau(bbox: Dict[str, float]) -> float:
    diag = math.hypot(bbox.get("w", 0), bbox.get("h", 0))
    return min(TAU_HI, max(TAU_LO, 0.5 * diag))


def loose_eq(pred: Any, gt: Any) -> bool:
    """Loose match: case-insensitive equality or substring either way; gt may be a list (any)."""
    if pred is None:
        return False
    p = str(pred).strip().lower()
    gts = gt if isinstance(gt, list) else [gt]
    for g in gts:
        if g is None:
            continue
        g = str(g).strip().lower()
        if p == g or g in p or p in g:
            return True
    return False


class Agg:
    """Accumulate counts for a P/R/F1 or accuracy metric."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.correct = self.total = 0

    def acc(self, ok: bool) -> None:
        self.total += 1
        self.correct += int(ok)

    def prf(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.tp + self.fp + self.fn == 0:      # no support -> undefined
            return None, None, None
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    def accuracy(self) -> Optional[float]:
        return self.correct / self.total if self.total else None


# --------------------------------------------------------------------------- attribute helpers
def canon_attr(v: Any) -> Optional[str]:
    """Canonical class string of an attribute value (list -> first element, as the training target)."""
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s or None


_LEFT = re.compile(r"\bleft\b")
_RIGHT = re.compile(r"\bright\b")
_FRONT = re.compile(r"\b(front|ahead)\b")
_LANE_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("adjacent-same-dir", re.compile(r"adjacent|same[- ]direction|neighbou?r|next lane|parallel lane")),
    ("oncoming", re.compile(r"oncoming|opposite|opposing|incoming lane")),
    ("cross-street", re.compile(r"cross[- ]?street|crossing street|perpendicular|intersecting|side street")),
    ("roadside", re.compile(r"road ?side|side of the road|curb|kerb|sidewalk|shoulder|parking (?:lot|space|lane|area)")),
    ("ego-lane", re.compile(r"same lane|ego(?:'s)?[- ]lane|in (?:the |my |our |its )?(?:current |own )?lane\b|current lane|own lane|ego path")),
)


def location_slots(text: Any) -> Dict[str, Optional[str]]:
    """Split a ``location`` phrase into slots: ``{'direction': front|front left|front right|None,
    'lane': ego-lane|adjacent-same-dir|oncoming|cross-street|roadside|None}``."""
    s = canon_attr(text) or ""
    direction: Optional[str] = None
    ml, mr = _LEFT.search(s), _RIGHT.search(s)
    if ml and mr:
        direction = "front left" if ml.start() < mr.start() else "front right"
    elif ml:
        direction = "front left"
    elif mr:
        direction = "front right"
    elif _FRONT.search(s):
        direction = "front"
    lane: Optional[str] = None
    for name, rx in _LANE_RULES:
        if rx.search(s):
            lane = name
            break
    return {"direction": direction, "lane": lane}


def direction_from_x(x_pano: float, width: float = PANO_W) -> str:
    """Direction implied by a panorama x coordinate: x/W < 0.45 front left, > 0.55 front right, else front."""
    r = float(x_pano) / float(width)
    if r < 0.45:
        return "front left"
    if r > 0.55:
        return "front right"
    return "front"


def coarse_class(type_str: Any) -> str:
    """Coarse class of an object type (exact canonical name first, then loose containment)."""
    t = (str(type_str) if type_str is not None else "").strip().lower()
    if not t:
        return "other"
    if t in _COARSE_LOOKUP:
        return _COARSE_LOOKUP[t]
    for name, c in _COARSE_LOOKUP.items():
        if name in t or t in name:
            return c
    return "other"


def _gt_ranks(ganns: Sequence[Dict[str, Any]]) -> List[int]:
    """GT rank per annotation (in file order): impact_rank if present, else its position after stable
    sorting with missing ranks as 99 (== target_builder rule)."""
    def key(i: int) -> float:
        r = (ganns[i].get("attributes") or {}).get("impact_rank")
        return float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else 99.0
    order = sorted(range(len(ganns)), key=key)
    ranks = [0] * len(ganns)
    for pos, i in enumerate(order, 1):
        r = (ganns[i].get("attributes") or {}).get("impact_rank")
        ranks[i] = int(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else pos
    return ranks


def _to_pano(point: Optional[Tuple[float, float]], scale: Optional[Tuple[float, float]], codec: Any) -> Optional[Tuple[float, float]]:
    """Model-coordinate point -> panorama pixels via the adapter codec (fallback: divide by scale)."""
    if point is None:
        return None
    x, y = float(point[0]), float(point[1])
    sc = tuple(scale) if scale is not None else (1.0, 1.0)
    if codec is not None and hasattr(codec, "decode"):
        px, py = codec.decode(x, y, sc)
        return float(px), float(py)
    return x / sc[0], y / sc[1]


class _GtObj:
    """Per-GT-object geometry cache (mask pixels decoded once)."""

    def __init__(self, ann: Dict[str, Any], coco: Optional[Sequence[Dict[str, Any]]]) -> None:
        self.ann = ann
        self.at = ann.get("attributes") or {}
        self.bbox = ann.get("bbox") or {}
        self.mask_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.mask: Optional[np.ndarray] = None
        ca = bbox_match(self.bbox, coco)
        if ca is not None:
            seg = ca.get("segmentation") or {}
            m = rle_to_mask(seg.get("size"), seg.get("counts")) if seg else None
            if m is not None:
                ys, xs = np.nonzero(m)
                self.mask = m
                self.mask_xy = (xs.astype(np.float64), ys.astype(np.float64))
        self.center = (self.bbox.get("x", 0) + self.bbox.get("w", 0) / 2.0,
                       self.bbox.get("y", 0) + self.bbox.get("h", 0) / 2.0)

    def dist(self, pt: Tuple[float, float]) -> float:
        """Distance (px) from a panorama point to the mask (0 inside), bbox centre if no mask."""
        x, y = pt
        if self.mask is None:
            return math.hypot(x - self.center[0], y - self.center[1])
        h, w = self.mask.shape
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and self.mask[yi, xi]:
            return 0.0
        xs, ys = self.mask_xy   # type: ignore[misc]
        if len(xs) == 0:
            return 1e9
        return float(np.sqrt((xs - x) ** 2 + (ys - y) ** 2).min())


def _greedy_match(gts: Sequence[_GtObj], preds: Sequence[Dict[str, Any]], pts: Sequence[Optional[Tuple[float, float]]],
                  same: Any) -> Tuple[List[Optional[int]], List[float]]:
    """Greedy matching: for each GT (file order) pick the nearest unused prediction with ``same(pred, gt)``.
    Predictions without a point are eligible but sorted last (d=1e17). Returns (match index, distance) per GT."""
    used: set = set()
    match: List[Optional[int]] = []
    dists: List[float] = []
    for g in gts:
        best, bestd = None, 1e18
        for i, po in enumerate(preds):
            if i in used or not same(po, g):
                continue
            d = 1e17 if pts[i] is None else g.dist(pts[i])
            if d < bestd:
                bestd, best = d, i
        if best is not None:
            used.add(best)
        match.append(best)
        dists.append(bestd)
    return match, dists


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


# --------------------------------------------------------------------------- main entry
def evaluate_understanding(records: Sequence[Dict[str, Any]], impl_judge: Any = None) -> Dict[str, Any]:
    """Compute understanding metrics over ``records`` (see module docstring for the record format).

    ``impl_judge``: optional callable(items) -> list of ``{'overall','cause','effect'}`` (LLM judge) where
    items are ``{'obj_desc','reference','prediction'}``; otherwise a token-overlap proxy is used.
    Returns a flat dict (``None`` = no support). Nested dicts: ``count_confusion``,
    ``location_direction_confusion``, ``location_lane_confusion``, ``attr_intention_confusion``,
    ``attr_state_confusion``, ``obj_recall_by_rank``.
    """
    ctx = {k: Agg() for k in CTX_KEYS}
    events = Agg(); presence = Agg()
    det = Agg(); det_r1 = Agg(); det_by_rank: Dict[str, Agg] = {"1": Agg(), "2": Agg(), "3+": Agg()}
    det_by_coarse = {c: Agg() for c in COARSE_CLASSES}
    coarse_det = Agg(); coarse_by = {c: Agg() for c in COARSE_CLASSES}
    type_err = Agg()                    # coarse hit but type-level miss (over GT objects)
    loc = Agg(); loc_strict = Agg()
    attr = {k: Agg() for k in ATTR_FIELDS}
    rank_acc = Agg(); rank_pm1 = Agg(); rank_missing = Agg(); rank_monotonic = Agg()
    dir_acc = Agg(); lane_acc = Agg(); loc_slot_acc = Agg()
    dir_conf: Counter = Counter(); lane_conf: Counter = Counter()
    attr_conf = {"intention": Counter(), "state": Counter()}
    to_major = {"intention": 0, "state": 0}; n_nonmajor = {"intention": 0, "state": 0}
    pred_major = {"intention": Agg(), "state": Agg()}
    minority_acc = {"intention": Agg(), "state": Agg()}
    dir_px = Agg(); dir_px_gt = Agg()
    impl = Agg(); impl_cause = Agg(); impl_effect = Agg(); impl_missing = Agg()
    impl_items: List[Dict[str, Any]] = []
    cnt_self = Agg(); cnt_gt = Agg(); cnt_mae: List[float] = []
    n_emit: List[int] = []; n_gt: List[int] = []
    count_conf: Counter = Counter(); under = over = exact = 0
    n_pred_total = n_pred_matched = 0
    n_frames = 0

    for rec in records:
        pred = rec["pred"]
        if isinstance(pred, str):
            pred = parse_output(pred)
        pano = rec.get("pano") or {}
        coco = rec.get("coco_anns")
        scale = rec.get("scale")
        codec = rec.get("point_codec")
        n_frames += 1
        gctx = pano.get("context") or {}
        for k in CTX_KEYS:
            if gctx.get(k) is not None:
                ctx[k].acc(loose_eq((pred.get("context") or {}).get(k), gctx.get(k)))
        gset = {t for e in (pano.get("traffic_events") or []) for t in (e.get("types") or [])}
        pset = set(pred.get("events") or [])
        events.tp += len(pset & gset); events.fp += len(pset - gset); events.fn += len(gset - pset)
        ganns = list(pano.get("annotations") or [])
        presence.acc(bool(pred.get("has_objects")) == bool(ganns))

        pobjs = list(pred.get("objects") or [])
        emit, gn = len(pobjs), len(ganns)
        n_emit.append(emit); n_gt.append(gn)
        pc = pred.get("n_objects")
        if pc is not None:
            cnt_self.acc(pc == emit)
            cnt_gt.acc(pc == gn)
            cnt_mae.append(abs(pc - emit))
        count_conf[f"{min(gn, 4)}->{min(emit, 4)}"] += 1
        under += int(emit < gn); over += int(emit > gn); exact += int(emit == gn)
        n_pred_total += emit
        ranks_emitted = [o.get("rank") for o in pobjs]
        if emit >= 2:
            rs = [r for r in ranks_emitted if r is not None]
            rank_monotonic.acc(len(rs) == emit and all(rs[i] < rs[i + 1] for i in range(len(rs) - 1)))
        for o in pobjs:
            rank_missing.acc(o.get("rank") is None)

        pts = [_to_pano(o.get("point"), scale, codec) for o in pobjs]
        # direction-pixel consistency of the predictions themselves (no GT needed)
        for o, pt in zip(pobjs, pts):
            d = location_slots(o.get("location")).get("direction")
            if pt is not None and d is not None:
                dir_px.acc(direction_from_x(pt[0]) == d)

        gts = [_GtObj(a, coco) for a in ganns]
        gt_ranks = _gt_ranks(ganns)
        for g in gts:      # GT-side reference for the same consistency statistic (bbox centre)
            d = location_slots(g.at.get("location")).get("direction")
            if d is not None:
                dir_px_gt.acc(direction_from_x(g.center[0]) == d)

        match, dists = _greedy_match(gts, pobjs, pts, lambda po, g: loose_eq(po.get("type"), g.at.get("type")))
        cmatch, _ = _greedy_match(gts, pobjs, pts, lambda po, g: coarse_class(po.get("type")) == coarse_class(g.at.get("type")))
        n_pred_matched += sum(1 for i in match if i is not None)

        for gi, g in enumerate(gts):
            at = g.at
            hit = match[gi] is not None
            cc = coarse_class(at.get("type"))
            det.acc(hit); det_by_coarse[cc].acc(hit)
            if at.get("impact_rank") == 1:
                det_r1.acc(hit)
            rk = at.get("impact_rank")
            if isinstance(rk, (int, float)) and not isinstance(rk, bool):
                det_by_rank["1" if rk == 1 else "2" if rk == 2 else "3+"].acc(hit)
            chit = cmatch[gi] is not None
            coarse_det.acc(chit); coarse_by[cc].acc(chit)
            type_err.acc(chit and not hit)
            if not hit:
                continue
            po = pobjs[match[gi]]; pt = pts[match[gi]]; bestd = dists[gi]
            tau = _tau(g.bbox)
            in_img = True
            if pt is not None:
                in_img = (0 <= pt[0] < PANO_W and 0 <= pt[1] < PANO_H)
            loc.acc(bestd <= tau and in_img)
            loc_strict.acc(bestd == 0.0)
            # rank
            pr = po.get("rank")
            rank_acc.acc(pr is not None and pr == gt_ranks[gi])
            rank_pm1.acc(pr is not None and abs(pr - gt_ranks[gi]) <= 1)
            # attributes (loose accuracy, only where GT has the field)
            for f in ATTR_FIELDS:
                if at.get(f):
                    attr[f].acc(loose_eq(po.get(f), at.get(f)))
            # location slots
            if at.get("location"):
                gs, ps = location_slots(at.get("location")), location_slots(po.get("location"))
                ok_d = ok_l = None
                if gs["direction"] is not None:
                    ok_d = ps["direction"] == gs["direction"]
                    dir_acc.acc(ok_d)
                    if not ok_d:
                        dir_conf[f"{gs['direction']}->{ps['direction']}"] += 1
                if gs["lane"] is not None:
                    ok_l = ps["lane"] == gs["lane"]
                    lane_acc.acc(ok_l)
                    if not ok_l:
                        lane_conf[f"{gs['lane']}->{ps['lane']}"] += 1
                if ok_d is not None or ok_l is not None:
                    loc_slot_acc.acc((ok_d is not False) and (ok_l is not False))
            # intention / state majority collapse
            for f in ("intention", "state"):
                gc = canon_attr(at.get(f))
                if gc is None:
                    continue
                pcn = canon_attr(po.get(f))
                ok = loose_eq(po.get(f), at.get(f))
                pred_major[f].acc(pcn is not None and MAJORITY in pcn)
                if gc != MAJORITY:
                    n_nonmajor[f] += 1
                    if pcn is not None and MAJORITY in pcn:
                        to_major[f] += 1
                if gc not in MAJORITY_SET:
                    minority_acc[f].acc(ok)
                if not ok:
                    attr_conf[f][f"{gc}->{pcn}"] += 1
            # implication
            if a_impl := g.ann.get("driving_implication"):
                impl_missing.acc(not po.get("implication"))
                impl_items.append({"obj_desc": f"{at.get('type', '')} {at.get('location', '') or ''}".strip(),
                                   "reference": a_impl, "prediction": po.get("implication")})

    n_impl_dropped = 0
    if impl_judge is not None and impl_items:
        for s in impl_judge(impl_items):
            if s.get("cause") is None:            # judge could not answer (API error / unparseable)
                n_impl_dropped += 1               # -> drop the item, do not charge it to the model
                continue
            impl.acc(bool(s["overall"])); impl_cause.acc(bool(s["cause"])); impl_effect.acc(bool(s["effect"]))
    else:
        for it in impl_items:
            if it["prediction"]:
                gset_ = set(str(it["reference"]).lower().split())
                pset_ = set(str(it["prediction"]).lower().split())
                impl.acc(len(gset_ & pset_) / max(len(gset_ | pset_), 1) > 0.15)
            else:
                impl.acc(False)

    m: Dict[str, Any] = {"n_frames": n_frames}
    # ---- core keys
    for k in CTX_KEYS:
        m[f"ctx_{k}_acc"] = ctx[k].accuracy()
    m["ctx_macro_acc"] = _mean(ctx[k].accuracy() for k in CTX_KEYS)
    m["events_f1"] = events.prf()[2]
    m["presence_f1"] = presence.accuracy()          # balanced binary -> accuracy proxy
    m["count_consistency"] = cnt_self.accuracy()
    m["count_vs_gt"] = cnt_gt.accuracy()
    m["count_mae"] = float(np.mean(cnt_mae)) if cnt_mae else None
    m["mean_emitted_obj"] = float(np.mean(n_emit)) if n_emit else None
    m["mean_gt_obj"] = float(np.mean(n_gt)) if n_gt else None
    m["obj_recall"] = det.accuracy()
    m["obj_recall_rank1"] = det_r1.accuracy()
    m["loc_hit_relaxed"] = loc.accuracy()
    m["loc_hit_strict"] = loc_strict.accuracy()
    for f in ATTR_FIELDS:
        m[f"attr_{f}_acc"] = attr[f].accuracy()
    m["attr_macro_acc"] = _mean(attr[f].accuracy() for f in ATTR_FIELDS)
    m["implication"] = impl.accuracy()
    m["implication_n_with_gt"] = len(impl_items)
    m["implication_n_scored"] = impl.total
    m["implication_n_dropped"] = n_impl_dropped
    m["implication_cause"] = impl_cause.accuracy()
    m["implication_effect"] = impl_effect.accuracy()
    if impl_judge is not None and n_impl_dropped > 0.20 * max(impl.total + n_impl_dropped, 1):
        # > 20 % of the judge items dropped (API outage): the surviving sample is biased -> withhold
        m["implication"] = m["implication_cause"] = m["implication_effect"] = None
        m["implication_judge_incomplete"] = True
    # ---- additional keys
    m["obj_precision"] = _rate(n_pred_matched, n_pred_total)
    m["n_pred_objects"] = n_pred_total
    m["n_fp_objects"] = n_pred_total - n_pred_matched
    m["obj_recall_by_rank"] = {k: v.accuracy() for k, v in det_by_rank.items()}
    for c in COARSE_CLASSES:
        m[f"obj_recall_{c}"] = det_by_coarse[c].accuracy()
        m[f"coarse_recall_{c}"] = coarse_by[c].accuracy()
    m["coarse_recall"] = coarse_det.accuracy()
    m["type_error_rate"] = type_err.accuracy()       # P(coarse hit & type miss) over GT objects
    m["count_confusion"] = dict(count_conf)
    m["count_under_rate"] = _rate(under, n_frames); m["count_over_rate"] = _rate(over, n_frames)
    m["count_exact_rate"] = _rate(exact, n_frames)
    m["rank_acc"] = rank_acc.accuracy()
    m["rank_acc_pm1"] = rank_pm1.accuracy()
    m["rank_missing_rate"] = rank_missing.accuracy()
    m["rank_monotonic_rate"] = rank_monotonic.accuracy()
    m["attr_location_direction_acc"] = dir_acc.accuracy()
    m["attr_location_lane_acc"] = lane_acc.accuracy()
    m["attr_location_slot_acc"] = loc_slot_acc.accuracy()
    m["location_direction_confusion"] = dict(dir_conf)
    m["location_lane_confusion"] = dict(lane_conf)
    for f in ("intention", "state"):
        m[f"attr_{f}_to_cruising"] = to_major[f]
        m[f"attr_{f}_to_cruising_rate"] = _rate(to_major[f], n_nonmajor[f])
        m[f"attr_{f}_pred_cruising_rate"] = pred_major[f].accuracy()
        m[f"attr_{f}_minority_acc"] = minority_acc[f].accuracy()
        m[f"attr_{f}_confusion"] = dict(attr_conf[f].most_common(30))
    m["attr_to_cruising_total"] = to_major["intention"] + to_major["state"]
    pooled_n = attr["intention"].total + attr["state"].total
    m["attr_intention_state_pooled_acc"] = _rate(attr["intention"].correct + attr["state"].correct, pooled_n)
    m["attr_slot_macro_acc"] = _mean([attr["intention"].accuracy(), attr["state"].accuracy(), attr["location"].accuracy()])
    m["direction_pixel_consistency"] = dir_px.accuracy()
    m["direction_pixel_n"] = dir_px.total
    m["gt_direction_pixel_consistency"] = dir_px_gt.accuracy()
    m["implication_missing_rate"] = impl_missing.accuracy()
    m["composite"] = s1_composite(m)
    return m
