"""Key-object metrics, three levels — additive to the metrics in ``understanding.py``.

L1  presence      : does the frame contain key objects at all (all frames)
L2  type confusion: GT class x predicted class (+ ``missed`` column, ``false_positive`` row),
                    on the frames that DO have key objects
L3  localisation  : how precisely the found objects are pointed at

Matching (the criterion the three levels share)
-----------------------------------------------
The model emits ONE point per object (no box), so assignment uses point-to-region distance:
``d = 0`` inside the GT SAM2 mask, else the minimum distance to the mask (bbox *boundary* when a
mask is missing; ``understanding.py`` instead falls back to the bbox *centre*, which is a
different, harsher quantity).

Two gates, deliberately:
  * ``tau_assign = clip(1.5*bbox_diag, 60, 300)`` px decides "the model is referring to this
    object".  It is LOOSE and TYPE-AGNOSTIC on purpose: with a tight or type-gated gate a
    mis-typed or sloppily-pointed object silently becomes a *miss*, which truncates both the
    confusion matrix and the localisation distribution.
  * ``tau_loc = clip(0.5*bbox_diag, 15, 60)`` px decides "the point is accurate".
With a single gate every matched object would satisfy the localisation criterion by construction
and L3 would be identically 1.0.

Assignment: GT objects in impact-rank order (most important first) each claim the nearest unused
prediction inside ``tau_assign``; one-to-one; surplus predictions on an already-claimed GT become
false positives (so spraying points cannot inflate recall).

Classes are the dataset's own four (``vehicle`` / ``traffic control`` / ``vulnerable user`` /
``obstacle``, i.e. the annotation's ``class`` prefix) — NOT ``understanding.COARSE``, which puts
Animal under VRU and Object under "other".
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from training.core.paths import PANO_H, PANO_W
from training.core.metrics.understanding import Agg, _GtObj, _gt_ranks, _to_pano, canon_attr

__all__ = ["evaluate_objects", "match_objects", "match_frame", "match_all", "tau_assign",
           "tau_loc", "TAXON_CLASSES", "type_to_taxon", "gt_taxon"]

TAU_LOC_K, TAU_LOC_LO, TAU_LOC_HI = 0.5, 15.0, 60.0
TAU_ASSIGN_K, TAU_ASSIGN_LO, TAU_ASSIGN_HI = 1.5, 60.0, 300.0
SENSITIVITY = (0.5, 1.0, 2.0)                 # tau_assign multipliers reported as a robustness check

TAXON_CLASSES = ("vehicle", "traffic control", "vulnerable user", "obstacle")
MISSED, FALSE_POS, UNKNOWN = "missed", "false_positive", "unknown"

# fine type -> dataset class.  A bare "Other" exists in three classes, so it stays UNKNOWN.
_TYPE_TAXON: Dict[str, str] = {}
for _c, _types in (
    ("vehicle", ("car", "truck", "bus", "emergency vehicle", "construction vehicle", "school bus")),
    ("traffic control", ("traffic light", "sign", "stop line", "cross walk", "crosswalk",
                         "temporary control", "bump")),
    ("vulnerable user", ("pedestrian", "cyclist", "motorcyclist", "scooter rider")),
    ("obstacle", ("object", "animal")),
):
    for _t in _types:
        _TYPE_TAXON[_t] = _c


def type_to_taxon(type_str: Any) -> str:
    """Dataset class of a fine type string; bare/unmatched 'Other' -> ``unknown``."""
    t = canon_attr(type_str) or ""
    if t in _TYPE_TAXON:
        return _TYPE_TAXON[t]
    for name, c in _TYPE_TAXON.items():          # loose containment ("a car" / "traffic lights")
        if name in t or (len(t) > 3 and t in name):
            return c
    return UNKNOWN


def gt_taxon(ann: Dict[str, Any]) -> str:
    """GT class straight from the annotation's ``class`` field ("vehicle 3" -> "vehicle")."""
    pre = str(ann.get("class") or "").rstrip("0123456789 ").strip().lower()
    if pre in TAXON_CLASSES:
        return pre
    return type_to_taxon((ann.get("attributes") or {}).get("type"))


def _diag(bbox: Dict[str, float]) -> float:
    return math.hypot(float(bbox.get("w", 0) or 0), float(bbox.get("h", 0) or 0))


def tau_loc(bbox: Dict[str, float]) -> float:
    return min(TAU_LOC_HI, max(TAU_LOC_LO, TAU_LOC_K * _diag(bbox)))


def tau_assign(bbox: Dict[str, float]) -> float:
    return min(TAU_ASSIGN_HI, max(TAU_ASSIGN_LO, TAU_ASSIGN_K * _diag(bbox)))


def _dist_boundary(g: _GtObj, pt: Tuple[float, float]) -> float:
    """Point-to-region distance: 0 inside the mask, else min distance to it.  No mask -> distance
    to the bbox BOUNDARY (0 inside the box), the right analogue of "inside the mask"."""
    x, y = pt
    if g.mask is not None:
        return g.dist(pt)
    b = g.bbox
    x0, y0 = float(b.get("x", 0) or 0), float(b.get("y", 0) or 0)
    x1, y1 = x0 + float(b.get("w", 0) or 0), y0 + float(b.get("h", 0) or 0)
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def match_objects(gts: Sequence[_GtObj], pts: Sequence[Optional[Tuple[float, float]]],
                  ranks: Sequence[int], gate_scale: float = 1.0
                  ) -> Tuple[List[Optional[int]], List[float], List[int]]:
    """Type-agnostic, gated, one-to-one assignment.  Returns (match idx per GT, distance per GT,
    unmatched prediction indices).  GT order = impact rank (ties keep file order)."""
    order = sorted(range(len(gts)), key=lambda i: (ranks[i], i))
    match: List[Optional[int]] = [None] * len(gts)
    dist: List[float] = [float("inf")] * len(gts)
    used: set = set()
    for i in order:
        g = gts[i]
        gate = tau_assign(g.bbox) * gate_scale
        best, bestd = None, float("inf")
        for j, p in enumerate(pts):
            if j in used or p is None:
                continue
            d = _dist_boundary(g, p)
            if d < bestd:
                bestd, best = d, j
        if best is not None and bestd <= gate:
            match[i], dist[i] = best, bestd
            used.add(best)
    fps = [j for j, p in enumerate(pts) if j not in used]
    return match, dist, fps


def match_frame(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry + assignment for ONE record.  Shared by ``evaluate_objects`` and
    ``behaviour.evaluate_behaviour`` so the SAM2 masks are decoded only once per frame."""
    ganns = list((rec.get("pano") or {}).get("annotations") or [])
    pobjs = list((rec.get("pred") or {}).get("objects") or [])
    gts = [_GtObj(a, rec.get("coco_anns")) for a in ganns]
    ranks = _gt_ranks(ganns) if ganns else []
    pts = [_to_pano(po.get("point"), rec.get("scale"), rec.get("point_codec")) for po in pobjs]
    match, dist, fps = match_objects(gts, pts, ranks) if gts else ([], [], list(range(len(pobjs))))
    return {"ganns": ganns, "pobjs": pobjs, "gts": gts, "ranks": ranks, "pts": pts,
            "match": match, "dist": dist, "fps": fps}


def match_all(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``match_frame`` for every record (pass the result to the metric functions)."""
    return [match_frame(r) for r in records]


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(xs, q)) if xs else None


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def evaluate_objects(records: Sequence[Dict[str, Any]], pre: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """L1 + L2 + L3 over the eval records (same record format as ``evaluate_understanding``).

    ``pre``: output of ``match_all(records)`` when the caller already computed it."""
    pres = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    n_gt_obj = n_pred_obj = n_found = n_found_r1 = n_gt_r1 = n_fp = 0
    conf_t: Dict[str, Dict[str, int]] = {}
    conf_f: Dict[str, Dict[str, int]] = {}
    type_ok_t = type_ok_f = 0
    loc_px: List[float] = []; loc_norm: List[float] = []
    in_mask = Agg(); near = Agg()
    in_mask_typed = Agg(); near_typed = Agg()
    loc_px_typed: List[float] = []
    sens = {m: [0, 0] for m in SENSITIVITY}                  # multiplier -> [found, total]

    def bump(table: Dict[str, Dict[str, int]], row: str, col: str) -> None:
        table.setdefault(row, {})
        table[row][col] = table[row].get(col, 0) + 1

    pre = list(pre) if pre is not None else [None] * len(records)   # type: ignore[list-item]
    for rec, mf in zip(records, pre):
        pred = rec["pred"]
        pano = rec["pano"]
        ganns = list(pano.get("annotations") or [])
        pobjs = list(pred.get("objects") or [])
        # ---------------- L1 presence (every frame) ----------------
        gt_yes, p_yes = bool(ganns), bool(pred.get("has_objects"))
        pres["tp" if (gt_yes and p_yes) else "fn" if gt_yes else "fp" if p_yes else "tn"] += 1
        if not ganns:
            n_fp += len(pobjs)                               # every object on an empty frame is an FP
            continue
        # ---------------- geometry (frames with GT objects) ----------------
        mf = mf if mf is not None else match_frame(rec)
        gts, ranks, pts = mf["gts"], mf["ranks"], mf["pts"]
        match, dist, fps = mf["match"], mf["dist"], mf["fps"]
        n_gt_obj += len(gts); n_pred_obj += len(pobjs); n_fp += len(fps)
        for m in SENSITIVITY:
            mm, _, _ = match_objects(gts, pts, ranks, gate_scale=m)
            sens[m][0] += sum(1 for x in mm if x is not None); sens[m][1] += len(gts)
        for i, g in enumerate(gts):
            gt_t, gt_f = gt_taxon(g.ann), canon_attr(g.at.get("type")) or UNKNOWN
            is_r1 = ranks[i] == 1
            n_gt_r1 += int(is_r1)
            j = match[i]
            if j is None:
                bump(conf_t, gt_t, MISSED); bump(conf_f, gt_f, MISSED)
                continue
            n_found += 1; n_found_r1 += int(is_r1)
            p_t = type_to_taxon(pobjs[j].get("type"))
            p_f = canon_attr(pobjs[j].get("type")) or UNKNOWN
            bump(conf_t, gt_t, p_t); bump(conf_f, gt_f, p_f)
            type_ok_t += int(p_t == gt_t); type_ok_f += int(p_f == gt_f)
            # ---------------- L3 localisation ----------------
            d = dist[i]
            px, py = pts[j]                                  # type: ignore[misc]
            inside_img = 0 <= px < PANO_W and 0 <= py < PANO_H
            hit0 = d == 0.0 and inside_img
            hitt = d <= tau_loc(g.bbox) and inside_img
            in_mask.acc(hit0); near.acc(hitt)
            loc_px.append(d)
            loc_norm.append(d / max(TAU_LOC_K * _diag(g.bbox), 1e-6))
            if p_t == gt_t:
                in_mask_typed.acc(hit0); near_typed.acc(hitt); loc_px_typed.append(d)
        for j in fps:
            bump(conf_t, FALSE_POS, type_to_taxon(pobjs[j].get("type")))
            bump(conf_f, FALSE_POS, canon_attr(pobjs[j].get("type")) or UNKNOWN)

    n_frames = pres["tp"] + pres["fp"] + pres["fn"] + pres["tn"]
    p = _rate(pres["tp"], pres["tp"] + pres["fp"])
    r = _rate(pres["tp"], pres["tp"] + pres["fn"])
    per_class = {}
    for c in TAXON_CLASSES:
        row = conf_t.get(c, {})
        n = sum(row.values())
        col = sum(conf_t.get(g, {}).get(c, 0) for g in list(TAXON_CLASSES) + [FALSE_POS])
        per_class[c] = {"n": n, "recall": _rate(row.get(c, 0), n), "precision": _rate(row.get(c, 0), col),
                        "missed": row.get(MISSED, 0)}
    out: Dict[str, Any] = {
        # ---- L1 ----
        "n_frames": n_frames,
        "presence_acc": _rate(pres["tp"] + pres["tn"], n_frames),
        "presence_precision": p, "presence_recall": r,
        "presence_f1_yes": (2 * p * r / (p + r)) if (p and r) else (0.0 if n_frames else None),
        "presence_confusion": dict(pres),
        "presence_baseline_always_yes": _rate(pres["tp"] + pres["fn"], n_frames),
        # ---- L2 ----
        "n_frames_with_gt": pres["tp"] + pres["fn"],
        "n_gt_objects": n_gt_obj, "n_pred_objects": n_pred_obj, "n_false_positive": n_fp,
        "det_recall": _rate(n_found, n_gt_obj),
        "det_recall_rank1": _rate(n_found_r1, n_gt_r1),
        "det_precision": _rate(n_found, n_pred_obj),
        "type_acc_found_taxon": _rate(type_ok_t, n_found),
        "type_acc_found_fine": _rate(type_ok_f, n_found),
        "confusion_taxon": conf_t, "confusion_fine": conf_f, "per_class_taxon": per_class,
        # ---- L3 (primary = all found objects; *_typed = found AND class correct) ----
        "in_mask_rate": in_mask.accuracy(), "near_rate": near.accuracy(),
        "loc_err_px_median": _pct(loc_px, 50), "loc_err_px_p90": _pct(loc_px, 90),
        "loc_err_norm_median": _pct(loc_norm, 50), "loc_err_norm_p90": _pct(loc_norm, 90),
        "in_mask_rate_typed": in_mask_typed.accuracy(), "near_rate_typed": near_typed.accuracy(),
        "loc_err_px_median_typed": _pct(loc_px_typed, 50),
        # ---- robustness check ----
        "det_recall_sensitivity": {f"{m}x": _rate(v[0], v[1]) for m, v in sens.items()},
    }
    return out


# --------------------------------------------------------------------------- self-test fixture
# Frame template: (class prefix, fine type, state, intention, content).  Two "moving" vehicles make
# ``moving`` the majority coarse state class, and the two signals give the red<->green statistic
# support, so the self-tests here and in ``metrics.behaviour`` have well-defined majority baselines.
_SYNTH_TEMPLATE: Tuple[Tuple[str, str, Optional[str], Optional[str], Optional[str]], ...] = (
    ("vehicle", "Car", "cruising", "cruising", None),
    ("vehicle", "Truck", "cruising", "cruising", None),
    ("vehicle", "Car", "stopping", "stopping", None),
    ("vehicle", "Car", "turn left", "turn left", None),
    ("vulnerable user", "Pedestrian", "crossing", "crossing", None),
    ("traffic control", "Traffic light", None, None, "red"),
    ("traffic control", "Traffic light", None, None, "green"),
)
_SYNTH_CANVAS = (400, 1400)          # (h, w) of the synthetic mask canvas, well inside the panorama


def _rect_rle(x0: int, y0: int, w: int, h: int) -> Dict[str, Any]:
    """COCO uncompressed-RLE segmentation (column-major) of one filled rectangle."""
    H, W = _SYNTH_CANVAS
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    flat = m.flatten(order="F")
    cut = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], cut, [flat.size]))).astype(int).tolist()
    counts = ([0] + runs) if flat[0] else runs      # the decoder starts on a run of zeros
    return {"size": [H, W], "counts": counts}


def _synthetic_records(n_frames: int = 12) -> List[Dict[str, Any]]:
    """Deterministic synthetic eval records for the self-tests -- no dataset or annotation file
    is read.  Every frame carries the objects of ``_SYNTH_TEMPLATE`` as non-overlapping rectangles
    with a matching mask, so an oracle predictor can point exactly inside every one of them and a
    point in the top-left corner falls outside every ``tau_assign`` gate.
    """
    recs: List[Dict[str, Any]] = []
    for _ in range(n_frames):
        ganns, coco = [], []
        for i, (klass, typ, state, intention, content) in enumerate(_SYNTH_TEMPLATE):
            x0, y0, w, h = 420 + 130 * i, 120 + 90 * (i % 2), 90, 70
            at: Dict[str, Any] = {"type": typ, "impact_rank": i + 1,
                                  "location": "front left" if i % 2 else "front right"}
            for k, v in (("state", state), ("intention", intention), ("content", content)):
                if v is not None:
                    at[k] = v
            ganns.append({"class": f"{klass} {i + 1}", "bbox": {"x": x0, "y": y0, "w": w, "h": h},
                          "attributes": at})
            coco.append({"bbox": [x0, y0, w, h], "segmentation": _rect_rle(x0, y0, w, h)})
        recs.append({"pano": {"annotations": ganns}, "coco_anns": coco,
                     "scale": (1.0, 1.0), "point_codec": None, "pred": None})
    return recs


if __name__ == "__main__":
    # ---- self-test on synthetic frames: oracle vs a degenerate predictor ----
    recs = _synthetic_records()

    def oracle(rec):
        """pred = GT, with each point the mask pixel nearest the mask centroid (always inside)."""
        objs = []
        for a in (rec["pano"].get("annotations") or []):
            g = _GtObj(a, rec["coco_anns"])
            if g.mask_xy is None:
                objs.append({"type": (a.get("attributes") or {}).get("type"), "point": g.center})
                continue
            xs, ys = g.mask_xy
            k = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
            objs.append({"type": (a.get("attributes") or {}).get("type"), "point": (xs[k], ys[k])})
        return {"has_objects": bool(objs), "objects": objs}

    for rec in recs:
        rec["pred"] = oracle(rec)
    m = evaluate_objects(recs)
    print("ORACLE  presence_acc=%.4f det_recall=%.4f type_acc=%.4f in_mask=%.4f fp=%d"
          % (m["presence_acc"], m["det_recall"], m["type_acc_found_taxon"], m["in_mask_rate"],
             m["n_false_positive"]))
    assert m["det_recall"] == 1.0 and m["in_mask_rate"] == 1.0 and m["n_false_positive"] == 0, m
    assert m["type_acc_found_taxon"] == 1.0, m["confusion_taxon"]

    # degenerate: always "yes" + one Car at a fixed corner -> the gate must reject it
    for rec in recs:
        rec["pred"] = {"has_objects": True, "objects": [{"type": "Car", "point": (20.0, 20.0)}]}
    m2 = evaluate_objects(recs)
    print("CORNER  presence_acc=%.4f (baseline %.4f) det_recall=%.4f fp=%d"
          % (m2["presence_acc"], m2["presence_baseline_always_yes"], m2["det_recall"],
             m2["n_false_positive"]))
    assert m2["det_recall"] < 0.05, "the tau_assign gate is not rejecting far-away points"
    print("objects.py self-test OK")
