"""Data-driven capability detection.

The rule is the same for every model: if its free-form thinking text reads as a reason, score it as
a reason; if it names object behaviour, score behaviour and consistency; otherwise report '—'.

A field is scored only if the model's OWN OUTPUT shows it: enough frames carry it (``MIN_COVERAGE``)
and the texts are not one constant sentence (``MIN_DISTINCT``).  A model that repeats the same
sentence on every frame therefore has coverage 1.0 but a single distinct text -> reason = '—'.
The static profile only supplies the coordinate system and the display name; it never forces a
capability on or off.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

MIN_COVERAGE = 0.20        # >= 20 % of frames must carry (or explicitly answer) the field
MIN_DISTINCT = 0.05        # distinct texts / frames-with-field  (one text over many frames -> constant -> absent)
MIN_CONTENT = 0.05         # object-level fields: >= 5 % of frames must actually carry objects (not only "none")
MIN_DISTINCT_STRUCT = 0.01 # types / points / trajectory: a closed vocabulary (~20 classes) is not a constant sentence
STRUCT_FIELDS = ("types", "points", "trajectory")
FIELDS = ("points", "types", "behaviour", "implication", "reason", "plan", "trajectory")
OBJECT_FIELDS = ("points", "types", "behaviour", "implication")


def _norm(s: Any) -> str:
    return " ".join(str(s).lower().split())


def detect_capabilities(records: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    """Per field: coverage, n_distinct, decision (True/False) and the rule that decided it."""
    n = max(len(records), 1)
    cov: Dict[str, int] = {f: 0 for f in FIELDS}
    none_ans: Dict[str, int] = {f: 0 for f in FIELDS}   # explicit "no objects" answers: a model that says
    texts: Dict[str, set] = {f: set() for f in FIELDS}  # <has_objects>no has still answered the question
    for rec in records:
        p = rec.get("pred") or {}
        objs = [o for o in (p.get("objects") or []) if isinstance(o, dict)]
        if not objs and (p.get("has_objects") is False or p.get("n_objects") == 0):
            for f in OBJECT_FIELDS:
                none_ans[f] += 1
        if objs:
            cov["types"] += 1; texts["types"].add(_norm(sorted(str(o.get("type")) for o in objs)))
        if any(o.get("point") is not None for o in objs):
            cov["points"] += 1; texts["points"].add(_norm([o.get("point") for o in objs]))
        if any(o.get("state") or o.get("intention") or o.get("content") for o in objs):
            cov["behaviour"] += 1
            texts["behaviour"].add(_norm([(o.get("state"), o.get("intention"), o.get("content")) for o in objs]))
        impl = [o.get("implication") for o in objs if o.get("implication")]
        if impl:
            cov["implication"] += 1; texts["implication"].add(_norm(impl))
        if p.get("reason"):
            cov["reason"] += 1; texts["reason"].add(_norm(p["reason"]))
        if p.get("final_plan"):
            cov["plan"] += 1; texts["plan"].add(_norm(p["final_plan"]))
        if rec.get("pred_xy") is not None or p.get("traj"):
            cov["trajectory"] += 1; texts["trajectory"].add(_norm(rec.get("pred_xy") or p.get("traj")))
    out: Dict[str, Dict[str, Any]] = {}
    for f in FIELDS:
        c = (cov[f] + none_ans[f]) / n                    # answered = carried the field OR said "no objects"
        content = cov[f] / n
        d = (len(texts[f]) / cov[f]) if cov[f] else 0.0
        if cov[f] == 0:
            decision, rule = False, "absent (0 frames)"
        elif c < MIN_COVERAGE:
            decision, rule = False, f"coverage {c:.2f} < {MIN_COVERAGE}"
        elif f in OBJECT_FIELDS and content < MIN_CONTENT:
            decision, rule = False, f"objects on only {cov[f]} frames ({content:.2f} < {MIN_CONTENT}); 'none' on {none_ans[f]}"
        elif d < (MIN_DISTINCT_STRUCT if f in STRUCT_FIELDS else MIN_DISTINCT):
            decision, rule = False, f"degenerate: {len(texts[f])} distinct over {cov[f]} frames"
        else:
            decision, rule = True, (f"coverage {c:.2f} ({cov[f]} with objects + {none_ans[f]} explicit none), "
                                    f"{len(texts[f])} distinct" if none_ans[f] else f"coverage {c:.2f}, {len(texts[f])} distinct")
        out[f] = {"coverage": round(c, 4), "n_frames": cov[f], "n_explicit_none": none_ans[f], "n_distinct": len(texts[f]),
                  "distinct_ratio": round(d, 4), "decision": decision, "rule": rule}
    # a plan is only worth scoring for consistency if there is also a trajectory to be consistent with;
    # behaviour only makes sense on objects that exist
    if out["behaviour"]["decision"] and not out["types"]["decision"]:
        out["behaviour"].update(decision=False, rule="no objects to attach behaviour to")
    return out


def caps_true(det: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    return {f: bool(det[f]["decision"]) for f in FIELDS}
