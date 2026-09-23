"""Geometry-free L1 / L2 for models that emit object TYPES but no image points.

Without a point there is nothing to assign spatially, so a frame is scored as two class multisets:
GT classes vs predicted classes.  Same-class items pair up (true positives), leftover GT items are
``missed``, leftover predictions are ``false_positive``.  Cross-class confusion is undefined here
(it needs geometry) and is reported as such; L3 is not computed.  The numbers are therefore an
UPPER bound on what a point-based ``det_recall`` would give the same model.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence

from training.core.metrics.objects import TAXON_CLASSES, UNKNOWN, gt_taxon, type_to_taxon
from training.core.metrics.understanding import canon_attr

__all__ = ["evaluate_types_only"]


def _rate(a: int, b: int) -> Optional[float]:
    return (a / b) if b else None


def evaluate_types_only(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pres = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tp = Counter(); fn = Counter(); fp = Counter()
    tp_f = Counter(); fn_f = Counter(); fp_f = Counter()
    n_gt = n_pred = n_gt_r1 = tp_r1 = 0
    for rec in records:
        ganns = list((rec.get("pano") or {}).get("annotations") or [])
        pobjs = list((rec.get("pred") or {}).get("objects") or [])
        gt_yes, p_yes = bool(ganns), bool(pobjs) or bool((rec.get("pred") or {}).get("has_objects"))
        pres["tp" if (gt_yes and p_yes) else "fn" if gt_yes else "fp" if p_yes else "tn"] += 1
        g = Counter(gt_taxon(a) for a in ganns)
        p = Counter(type_to_taxon(o.get("type")) for o in pobjs)
        gf = Counter(canon_attr((a.get("attributes") or {}).get("type")) or UNKNOWN for a in ganns)
        pf = Counter(canon_attr(o.get("type")) or UNKNOWN for o in pobjs)
        n_gt += len(ganns); n_pred += len(pobjs)
        for c in set(g) | set(p):
            k = min(g[c], p[c]); tp[c] += k; fn[c] += g[c] - k; fp[c] += p[c] - k
        for c in set(gf) | set(pf):
            k = min(gf[c], pf[c]); tp_f[c] += k; fn_f[c] += gf[c] - k; fp_f[c] += pf[c] - k
        # rank-1 element: found if its class appears among the predictions
        r1 = [a for a in ganns if (a.get("attributes") or {}).get("impact_rank") == 1]
        for a in r1:
            n_gt_r1 += 1; tp_r1 += int(p[gt_taxon(a)] > 0)
    n_frames = sum(pres.values())
    per = {c: {"n": tp[c] + fn[c], "recall": _rate(tp[c], tp[c] + fn[c]), "precision": _rate(tp[c], tp[c] + fp[c]),
               "missed": fn[c], "false_positive": fp[c]} for c in TAXON_CLASSES}
    return {
        "mode": "type_only (no points -> class multiset matching; upper bound on point-based recall)",
        "n_frames": n_frames, "n_gt_objects": n_gt, "n_pred_objects": n_pred,
        "presence_acc": _rate(pres["tp"] + pres["tn"], n_frames), "presence_confusion": dict(pres),
        "presence_baseline_always_yes": _rate(pres["tp"] + pres["fn"], n_frames),
        "det_recall_typeonly": _rate(sum(tp.values()), n_gt),
        "det_recall_rank1_typeonly": _rate(tp_r1, n_gt_r1),
        "det_precision_typeonly": _rate(sum(tp.values()), n_pred),
        "fine_type_recall_typeonly": _rate(sum(tp_f.values()), n_gt),
        "per_class_taxon": per,
        "n_false_positive": sum(fp.values()),
        "type_acc_found_taxon": None, "in_mask_rate": None,      # undefined without geometry
    }
