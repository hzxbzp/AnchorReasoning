"""Behaviour-judgement metrics: do we get the OTHER agents' behaviour and the traffic
controls' content right?

Three closed sets, each scored only on objects the model actually found (``objects.match_frame``):

* ``state``     — what a vehicle / VRU is doing now
* ``intention`` — what it is about to do
* ``content``   — what a traffic control says

The raw annotations are free-ish text (dozens of distinct raw values per field, a long tail of
typos and phrases such as "the man wants to open the car's door"), so every value is first
canonicalised to a closed label.  Without that step the comparison is a substring test and pure
wording mismatches count as errors: GT ``"left turn"`` vs prediction ``"turn left"`` is judged
WRONG by ``understanding.loose_eq`` (neither contains the other).

Two granularities: ``fine`` (11 / 11 labels) and ``coarse`` (5 + other), because the rare classes
carry too few samples to read on their own.  ``macro_recall`` averages only over GT classes with at
least ``MIN_SUPPORT`` samples and reports which classes were kept.

``content`` additionally reports the **red<->green confusion rate**: the one perception error in
this system that can directly cause a collision.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

from training.core.metrics.objects import gt_taxon, match_frame
from training.core.metrics.understanding import canon_attr

__all__ = ["evaluate_behaviour", "canon_behaviour", "canon_content", "BEHAV_FINE", "BEHAV_COARSE",
           "CONTENT_FINE", "CONTENT_MAIN", "MIN_SUPPORT"]

MIN_SUPPORT = 10                       # classes below this are excluded from macro recall

BEHAV_FINE = ("cruising", "accelerating", "decelerating", "stopping", "parking", "crossing",
              "turn left", "turn right", "lane change", "backing", "other")
# coarse grouping; "moving" deliberately covers cruising + accelerating
BEHAV_COARSE_MAP = {
    "cruising": "moving", "accelerating": "moving",
    "decelerating": "slowing/stopped", "stopping": "slowing/stopped",
    "parking": "parked",
    "turn left": "turning/merging", "turn right": "turning/merging", "lane change": "turning/merging",
    "crossing": "crossing", "backing": "other", "other": "other",
}
BEHAV_COARSE = ("moving", "slowing/stopped", "parked", "turning/merging", "crossing", "other")

# first match wins -> put the specific patterns first
_BEHAV_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (lab, re.compile(rx)) for lab, rx in (
        ("turn left",    r"\bu[\s-]?turn|\bleft turn\b|\bturn(?:ing|s)? left\b"),
        ("turn right",   r"\bright turn\b|\bturn(?:ing|s)? right\b"),
        ("lane change",  r"lane[\s-]?chang|\blang change|lane[\s-]?borrow|\bmerg|\bnudge\b|pull[\s-]?out"),
        ("parking",      r"\bpark"),                      # parking / prepare to park / pull over and …
        ("crossing",     r"\bcross"),                     # crossing / may crossing / cross the road
        ("backing",      r"\bbacking\b|\bback(?:ing)? up\b"),
        ("accelerating", r"\bacceler"),
        ("decelerating", r"\bdeceler|\bgive way\b|\byield"),
        ("stopping",     r"\bstopp|\bstanding\b|\bstationary\b|\bhalt"),
        # locomotion ALONG the road (agent keeps moving in its own path) -> cruising
        ("cruising",     r"\bcruis|\bdriv|\bride\b|\briding\b|\bplanning to drive\b"
                         r"|\b(?:walk|walking|walkingn|run|running)\b(?!\s*(?:to|towards|near|aroud|around))"),
    ))


def canon_behaviour(v: Any) -> str:
    """Canonical fine behaviour label of a raw ``state`` / ``intention`` value."""
    s = canon_attr(v)
    if not s:
        return "other"
    for lab, rx in _BEHAV_RULES:
        if rx.search(s):
            return lab
    return "other"


CONTENT_FINE = ("red", "green", "yellow", "stop", "keep left", "keep right", "lane guiding",
                "speed limit", "no turn", "yield", "other")
CONTENT_MAIN = ("red", "green", "yellow", "stop", "other")     # coarse confusion matrix
_CONTENT_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (lab, re.compile(rx)) for lab, rx in (
        ("keep left",    r"\bkeep left\b"),
        ("keep right",   r"\bkeep right\b"),
        ("no turn",      r"\bno (?:left |right )?turn"),
        ("speed limit",  r"speed limit"),
        ("lane guiding", r"lane guiding|guiding"),
        ("yield",        r"\byield\b"),
        ("red",          r"\bred\b"),                      # "stop here on red" -> see below
        ("green",        r"\bgreen\b"),
        ("yellow",       r"\byellow\b|\bamber\b"),
        ("stop",         r"\bstop\b"),
    ))


def canon_content(v: Any) -> str:
    """Canonical traffic-control content label.  ``"stop here on red"`` counts as ``stop`` (it is a
    sign legend, not a signal state), so ``stop`` is tested before the colour words for that case."""
    s = canon_attr(v)
    if not s:
        return "other"
    if "stop" in s and "red" in s:
        return "stop"
    for lab, rx in _CONTENT_RULES:
        if rx.search(s):
            return lab
    return "other"


def _to_main(lab: str) -> str:
    return lab if lab in CONTENT_MAIN else "other"


class _Conf:
    """Confusion accumulator over a closed label set."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = tuple(labels)
        self.m: Dict[str, Dict[str, int]] = {}
        self.n = 0
        self.ok = 0

    def add(self, gt: str, pred: str) -> None:
        self.m.setdefault(gt, {})
        self.m[gt][pred] = self.m[gt].get(pred, 0) + 1
        self.n += 1
        self.ok += int(gt == pred)

    def report(self, min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
        per: Dict[str, Dict[str, Any]] = {}
        for g in self.labels:
            row = self.m.get(g, {})
            n = sum(row.values())
            col = sum(self.m.get(x, {}).get(g, 0) for x in self.labels)
            per[g] = {"n": n, "recall": (row.get(g, 0) / n) if n else None,
                      "precision": (row.get(g, 0) / col) if col else None}
        kept = [g for g in self.labels if per[g]["n"] >= min_support]
        macro = ([per[g]["recall"] for g in kept] or None)
        maj = max(((per[g]["n"], g) for g in self.labels), default=(0, None))
        return {
            "n": self.n,
            "acc": (self.ok / self.n) if self.n else None,
            "macro_recall": (sum(macro) / len(macro)) if macro else None,
            "macro_classes": kept,
            "majority_baseline": (maj[0] / self.n) if self.n else None,
            "majority_class": maj[1],
            "confusion": self.m,
            "per_class": per,
        }


def evaluate_behaviour(records: Sequence[Dict[str, Any]],
                       pre: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """``state`` / ``intention`` / ``content`` metrics on the objects the model found."""
    conf = {
        "state_fine": _Conf(BEHAV_FINE), "state_coarse": _Conf(BEHAV_COARSE),
        "intention_fine": _Conf(BEHAV_FINE), "intention_coarse": _Conf(BEHAV_COARSE),
        "content_fine": _Conf(CONTENT_FINE), "content_main": _Conf(CONTENT_MAIN),
    }
    collapse: Dict[str, int] = {}          # GT class -> counted as "moving" by the model
    rg = {"red_as_green": 0, "green_as_red": 0, "n_signal": 0}
    cov = {"state_gt": 0, "state_scored": 0, "content_gt": 0, "content_scored": 0}
    pre = list(pre) if pre is not None else [None] * len(records)    # type: ignore[list-item]
    for rec, mf in zip(records, pre):
        mf = mf if mf is not None else match_frame(rec)
        for i, g in enumerate(mf["gts"]):
            at = g.at
            klass = gt_taxon(g.ann)
            j = mf["match"][i]
            po = mf["pobjs"][j] if j is not None else None
            if klass in ("vehicle", "vulnerable user"):
                for field in ("state", "intention"):
                    if not at.get(field):
                        continue
                    cov[f"{field[:5]}_gt" if field == "state" else "state_gt"] += 0  # counted below
                    cov["state_gt"] += 1 if field == "state" else 0
                    if po is None:
                        continue
                    cov["state_scored"] += 1 if field == "state" else 0
                    gt_l = canon_behaviour(at.get(field))
                    pr_l = canon_behaviour(po.get(field))
                    conf[f"{field}_fine"].add(gt_l, pr_l)
                    gc, pc = BEHAV_COARSE_MAP[gt_l], BEHAV_COARSE_MAP[pr_l]
                    conf[f"{field}_coarse"].add(gc, pc)
                    if field == "state" and pc == "moving" and gc != "moving":
                        collapse[gc] = collapse.get(gc, 0) + 1
            elif klass == "traffic control" and at.get("content"):
                cov["content_gt"] += 1
                if po is None:
                    continue
                cov["content_scored"] += 1
                gt_l, pr_l = canon_content(at.get("content")), canon_content(po.get("content"))
                conf["content_fine"].add(gt_l, pr_l)
                conf["content_main"].add(_to_main(gt_l), _to_main(pr_l))
                if gt_l in ("red", "green"):
                    rg["n_signal"] += 1
                    if gt_l == "red" and pr_l == "green":
                        rg["red_as_green"] += 1
                    if gt_l == "green" and pr_l == "red":
                        rg["green_as_red"] += 1
    out: Dict[str, Any] = {k: v.report() for k, v in conf.items()}
    out["state_collapse_to_moving"] = collapse
    out["red_green"] = dict(rg, rate=((rg["red_as_green"] + rg["green_as_red"]) / rg["n_signal"])
                            if rg["n_signal"] else None)
    out["coverage"] = cov
    return out


if __name__ == "__main__":
    # ---- self-test: canonicalisation table + oracle / majority predictors on synthetic frames ----
    cases = [("left turn", "turn left"), ("turn left", "turn left"), ("u turn", "turn left"),
             ("the car wants to make an U turn", "turn left"), ("right lane change", "lane change"),
             ("merge into the main road", "lane change"), ("prepare to park on the roadside", "parking"),
             ("may crossing", "crossing"), ("He may cross the road.", "crossing"),
             ("give way to the ego vehicle", "decelerating"), ("standing", "stopping"),
             ("walking along the road", "cruising"), ("driving towards the ego vehicle", "cruising"),
             ("walking to the black car", "other"), ("openning the car's door", "other")]
    for raw, want in cases:
        got = canon_behaviour(raw)
        assert got == want, f"{raw!r} -> {got!r}, want {want!r}"
    for raw, want in [("green", "green"), ("red", "red"), ("stop", "stop"),
                      ("stop here on red", "stop"), ("keep left", "keep left"),
                      ("speed limit 25 mph", "speed limit"), ("no right turn", "no turn")]:
        got = canon_content(raw)
        assert got == want, f"{raw!r} -> {got!r}, want {want!r}"
    print("canonicalisation table OK (%d cases)" % (len(cases) + 7))

    import numpy as np
    from training.core.metrics.objects import _synthetic_records
    from training.core.metrics.understanding import _GtObj

    recs = _synthetic_records()

    def pred_from_gt(rec, const=None):
        objs = []
        for a in (rec["pano"].get("annotations") or []):
            g = _GtObj(a, rec["coco_anns"])
            at = a.get("attributes") or {}
            if g.mask_xy is None:
                pt = g.center
            else:
                xs, ys = g.mask_xy
                k = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
                pt = (xs[k], ys[k])
            o = {"type": at.get("type"), "point": pt}
            for f in ("state", "intention", "content"):
                o[f] = (const.get(f) if const else at.get(f))
            objs.append(o)
        return {"has_objects": bool(objs), "objects": objs}

    for r in recs:
        r["pred"] = pred_from_gt(r)
    m = evaluate_behaviour(recs)
    print("ORACLE   state_coarse acc=%.4f macro=%.4f | content_main acc=%.4f | red<->green=%.4f (n=%d)"
          % (m["state_coarse"]["acc"], m["state_coarse"]["macro_recall"],
             m["content_main"]["acc"], m["red_green"]["rate"], m["red_green"]["n_signal"]))
    assert m["state_coarse"]["acc"] == 1.0 and m["content_main"]["acc"] == 1.0
    assert m["red_green"]["rate"] == 0.0

    for r in recs:
        r["pred"] = pred_from_gt(r, {"state": "cruising", "intention": "cruising", "content": "green"})
    m2 = evaluate_behaviour(recs)
    s, c = m2["state_coarse"], m2["content_main"]
    print("MAJORITY state_coarse acc=%.4f macro=%.4f (majority %s %.4f, n=%d)"
          % (s["acc"], s["macro_recall"], s["majority_class"], s["majority_baseline"], s["n"]))
    print("         content_main acc=%.4f macro=%.4f (majority %s %.4f, n=%d) red->green=%d/%d"
          % (c["acc"], c["macro_recall"], c["majority_class"], c["majority_baseline"], c["n"],
             m2["red_green"]["red_as_green"], m2["red_green"]["n_signal"]))
    print("         state collapse-to-moving:", m2["state_collapse_to_moving"])
    print("         macro classes (state_coarse):", s["macro_classes"])
    print("         coverage:", m2["coverage"])
    assert abs(s["acc"] - s["majority_baseline"]) < 1e-9, "majority predictor must hit the baseline"
    print("behaviour.py self-test OK")
