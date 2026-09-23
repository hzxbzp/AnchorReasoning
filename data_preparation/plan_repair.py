#!/usr/bin/env python3
"""Final-plan consistency repair -> ``plan_repair.json`` + ``plan_repair_v2_stats.json``.

For every index frame of p7-p21 the longitudinal trend class of the ground-truth future trajectory
(``motion_lon``) is compared with the class the annotated ``final_plan`` implies (``plan_to_motion``).
``unspecified`` (lateral-only plan) / ``unmatched`` plans are never counted as contradictions.  A frame is a
contradiction only when ``labels.plan_consistent`` (tolerant default = the *compatibility matrix*, "not
contradictory" == consistent) returns ``False`` for the longitudinal part; the matrix needs ``v0`` (history end
speed), ``v_end`` (mean speed of the last future second) and ``start_t`` (first future time with v >= 0.5 m/s),
which the worker records carry.  A plan whose class merely differs from ``motion_lon`` but is compatible
("decelerate" while the car holds speed and ends slower, "stop and wait" with a start 2 s later, "cruise" while
slowing within max(3, 0.5*v0) m/s, ...) is counted as ``tolerated`` and left alone.  The stricter class-equality
rule (``plan_consistent(..., strict=True)``) is still reported in the statistics as ``lon_inconsistent_strict``
/ ``inconsistency_rate_strict`` for comparison.  Lateral conflicts (plan turn vs motion straight / opposite) are
counted (``lateral_conflicts_with_lon_ok``) but never rewritten.

When the frame IS a contradiction the plan's longitudinal verb phrase is rewritten to the standard verb of
``motion_lon`` (lateral wording such as "keep lane" / "turn left" is preserved):

    -> keep: "cruise"   -> accelerate: "accelerate" (v0 < 0.5: "proceed")   -> decelerate: "decelerate"
    -> stop: "stop and wait" (moving) / "remain stopped" (already stopped)

Rewritten frames get ``reason_weight = 0.5``. The overlay is ``{fdir: {final_plan, reason_weight, orig_plan,
motion_lon, ...}}``; the original ``panorama_geo.json`` files are never modified.

``plan_to_motion`` / ``plan_consistent`` ARE ``training.core.labels.plan_to_motion`` / ``plan_consistent`` -- the
single implementation of the plan matcher and of the compatibility matrix, shared with training and evaluation;
the first-segment rule for two-stage plans also comes from ``labels.plan_first_segment``.  The lexicon below does
NOT classify plans: it only locates the longitudinal verb spans that ``rewrite_plan`` replaces, and every rewrite
is verified with ``labels.plan_to_motion`` (a candidate the shared matcher does not read as ``motion_lon`` is
reported as ``rewrite_failed``, never applied).

Usage::

    PYTHONPATH=<repo root> python -m data_preparation.plan_repair \
        [--index $ANCHOR_DATA/index_train.json] [--workers 32] [--limit 20 --out /tmp/x.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core import labels as L  # noqa: E402
from training.core.paths import DATA, S2_PARTITIONS, partition_of  # noqa: E402

V_STOP = L.STOP_SPEED
LON_CLASSES = L.MOTION_LON
UNDECIDED = ("unspecified", "unmatched")
REASON_WEIGHT_REWRITTEN = 0.5
LABEL_SOURCE = PLAN_FN_SOURCE = "training.core.labels"

plan_to_motion = L.plan_to_motion          # plan -> motion-class matcher: single implementation
plan_consistent = L.plan_consistent        # tolerant by default (compatibility matrix); strict=True = class equality
CONSISTENCY_RULE = "compatibility_matrix"  # consistency criterion recorded in the stats

# =============================================================================================================
# Rewrite lexicon: locates the verb spans ``rewrite_plan`` replaces (NOT a classifier -- classification is
# ``labels.plan_to_motion``; the ``cls`` field only tells longitudinal spans from lateral ones).
# Entries: (regex, cls, ext, tier)
#   cls : 'stop' | 'decelerate' | 'keep' | 'accelerate' | 'v0' (v0-dependent longitudinal verb)
#         | 'lat' (named group `dir`) | 'lat_left' | 'note:<name>'
#   ext : None | 'prep' (extend over a following prepositional object clause) | 'always' (extend over the object)
#   tier: 'table' (canonical plan verb phrase) | 'fallback' (extra keyword)
# Order matters: at the same start position the first matching entry wins (specific phrases first).
# =============================================================================================================
_CONNECT = r"(?:and|then|while|until|till|before|after|as|so|because|once|when|if|but|or)"
_LAT_STOP = r"(?:turn|turning|turns|change|changing|changes|merge|merging|nudge|nudging|pull|pulling|veer|veering|bear|bearing|keep|keeping|stay|staying|maintain|maintaining|hold|holding)"
_DELIM = rf"(?=,|;|\.|:|$|\s+{_CONNECT}\b|\s+(?:to\s+)?{_LAT_STOP}\b)"
_PREPS = r"(?:for|behind|at|before|until|till|to|in\s+front\s+of|near|by|with|through|past|along|into|onto|towards?|across|over|from|around|after|on|under|beyond|up\s+to)"
_EXT_PREP = rf"(?:\s+{_PREPS}\b[^,;.:]*?{_DELIM})?"
_EXT_ALWAYS = rf"(?:\s+(?!{_CONNECT}\b)(?!(?:to\s+)?{_LAT_STOP}\b)[^,;.:]*?{_DELIM})?"
_NOT_STOP_NOUN = r"(?!\s+(?:sign|signs|line|lines|light|lights|signal|signals|vehicle|vehicles|car|cars|truck|bus|traffic|lead))"

LEXICON: List[Tuple[str, str, Optional[str], str]] = [
    # ---- stop (table) ----
    (r"(?:decelerat\w*|slow\w*(?:\s+down)?|brak\w*|com(?:e|es|ing)|came)\s+to\s+a\s+(?:complete\s+|full\s+)?(?:stop|halt|standstill)", "stop", "prep", "table"),
    (r"(?:remain|remains|remaining|stay|stays|staying|stand|stands|standing|keep|keeps|keeping|continue|continues|continuing)\s+(?:stopped|still|stationary|halted|parked|waiting|at\s+a\s+standstill|to\s+wait)", "stop", "prep", "table"),
    (r"pull(?:s|ed|ing)?\s+over\s+and\s+stop(?:s|ped)?", "stop", "prep", "table"),
    (r"stand\s*still", "stop", None, "table"),
    (r"stay\s+put", "stop", None, "table"),
    (r"hold(?:s|ing)?\s+(?:the\s+|its\s+)?position", "stop", None, "table"),
    (r"do\s+not\s+(?:move|proceed|go|enter)", "stop", "prep", "table"),
    (rf"(?<!the )(?<!a )(?<!an )stop(?:s|ped|ping)?\b{_NOT_STOP_NOUN}", "stop", "prep", "table"),
    (r"wait(?:s|ed|ing)?\b", "stop", "prep", "table"),
    (r"hold(?:s|ing)?\b(?!\s+(?:the\s+)?lane)", "stop", "prep", "table"),
    (r"halt(?:s|ed|ing)?\b(?!\s+(?:sign|line))", "stop", "prep", "table"),
    (r"idl(?:e|es|ing)\b", "stop", None, "fallback"),
    # ---- decelerate (table) ----
    (r"decelerat\w*", "decelerate", "prep", "table"),
    (r"slow(?:n|s|ed|ing)?(?:\s+down)?\b(?!\s+(?:traffic|vehicle|vehicles|car|cars|moving|-moving|lane|speed))", "decelerate", "prep", "table"),
    (r"brak(?:e|es|ed|ing)\b(?!\s+light)", "decelerate", "prep", "table"),
    (r"yield(?:s|ed|ing)?\b(?!\s+sign)", "decelerate", "prep", "table"),
    (r"giv(?:e|es|ing)\s+way", "decelerate", "prep", "table"),
    (r"eas(?:e|es|ed|ing)\s+off", "decelerate", "prep", "table"),
    (r"reduc(?:e|es|ed|ing)\s+(?:the\s+|its\s+)?speed", "decelerate", "prep", "table"),
    (r"approach(?:es|ed|ing)?\s+(?:cautiously|slowly|carefully)", "decelerate", "prep", "table"),
    (r"approach(?:es|ed|ing)?\s+[^,;.:]*?\s+(?:cautiously|slowly|carefully)", "decelerate", None, "table"),
    (r"back(?:s|ed|ing)?\s+off", "decelerate", "prep", "table"),
    # ---- keep (table) ----
    (r"cruis(?:e|es|ed|ing)\b", "keep", "prep", "table"),
    (r"maintain(?:s|ed|ing)?\b(?!\s+(?:the\s+)?lane)", "keep", "always", "table"),
    (r"keep(?:s|ing)?\s+(?:the\s+|a\s+|its\s+|current\s+)?(?:pace|speed|moving|going|rolling|cruising|driving|(?:safe\s+)?distance|(?:safe\s+)?gap|following)\b", "keep", "prep", "table"),
    (r"(?<!the )(?<!a )follow(?:s|ed|ing)?\b", "keep", "always", "table"),
    (r"continu(?:e|es|ed|ing)\b(?!\s+straight)", "keep", "prep", "table"),
    (r"go(?:es|ing)?\s+through\b", "keep", "prep", "table"),
    (r"(?<!the )(?<!a )pass(?:es|ed|ing)?\b(?!enger)", "keep", "always", "table"),
    (r"coast(?:s|ed|ing)?\b", "keep", "prep", "table"),
    # ---- accelerate (table) ----
    (r"accelerat\w*", "accelerate", "prep", "table"),
    (r"speed(?:s|ed|ing)?\s+up\b", "accelerate", "prep", "table"),
    (r"start(?:s|ed|ing)?\b(?!\s+of\b)", "accelerate", "always", "table"),
    (r"mov(?:e|es|ed|ing)\s+(?:off|out|on)\b", "accelerate", "prep", "table"),
    (r"pull(?:s|ed|ing)?\s+away\b", "accelerate", "prep", "table"),
    (r"resum(?:e|es|ed|ing)\b", "accelerate", "always", "table"),
    (r"launch(?:es|ed|ing)?\b", "accelerate", "prep", "table"),
    (r"pull(?:s|ed|ing)?\s+out(?:\s+(?:to\s+the\s+)?(?:left|right))?\b", "accelerate", "prep", "table"),
    (r"get(?:s|ting)?\s+going\b", "accelerate", None, "table"),
    (r"set(?:s|ting)?\s+off\b", "accelerate", None, "table"),
    (r"depart\w*", "accelerate", "prep", "fallback"),
    # ---- v0-dependent (table): proceed / creep / go from stop -> accelerate, while moving -> keep ----
    (r"proceed(?:s|ed|ing)?\b", "v0", "prep", "table"),
    (r"creep(?:s|ed|ing)?(?:\s+forward)?\b", "v0", "prep", "table"),
    (r"go(?:es|ing)?\s+straight\b", "v0", "prep", "table"),
    (r"continu(?:e|es|ed|ing)\s+straight\b", "v0", "prep", "table"),
    (r"go(?:es|ing)?\b(?!\s+(?:through|around|past|left|right|straight))", "v0", "prep", "table"),
    (r"mov(?:e|es|ed|ing)\s+(?:forward|ahead)\b", "v0", "prep", "table"),
    (r"driv(?:e|es|ing)\b", "v0", "prep", "fallback"),
    (r"advanc(?:e|es|ed|ing)\b", "v0", "prep", "fallback"),
    (r"travel(?:s|led|ling|ed|ing)?\b", "v0", "prep", "fallback"),
    (r"roll(?:s|ed|ing)?\s+(?:forward|through|over|on)\b", "v0", "prep", "fallback"),
    # ---- lateral: turns ----
    (r"u-?\s?turn(?:s|ing)?\b", "lat_left", None, "table"),
    (r"turn(?:s|ed|ing)?\s+(?:to\s+the\s+)?(?P<dir>left|right)\b", "lat", None, "table"),
    (r"(?P<dir>left|right)(?:-|\s+)(?:hand\s+)?turn(?:s|ing)?\b", "lat", None, "table"),
    (r"veer(?:s|ed|ing)?\s+(?:to\s+the\s+)?(?P<dir>left|right)\b", "lat", None, "table"),
    (r"bear(?:s|ing)?\s+(?:to\s+the\s+)?(?P<dir>left|right)\b", "lat", None, "table"),
    (r"go(?:es|ing)?\s+(?P<dir>left|right)\b", "lat", None, "table"),
    # ---- lateral notes (recorded only) ----
    (r"lane\s+chang\w*|chang(?:e|es|ing)\s+(?:to\s+the\s+)?(?:left\s+|right\s+)?lanes?\b|chang(?:e|es|ing)\s+lanes?\b", "note:lane_change", None, "table"),
    (r"merg(?:e|es|ing)\b", "note:merge", None, "table"),
    (r"nudg(?:e|es|ing)\b", "note:nudge", None, "table"),
    (r"borrow\w*", "note:borrow", None, "table"),
    (r"pull(?:s|ed|ing)?\s+over\b", "note:pull_over", None, "table"),
    (r"go(?:es|ing)?\s+around\b|bypass\w*|overtak\w*", "note:go_around", None, "table"),
    (r"(?:keep|stay|remain|hold|maintain)\w*\s+(?:in\s+)?(?:the\s+|its\s+|current\s+)?lane\b", "note:lane_keep", None, "table"),
    (r"\bstraight\b", "note:straight", None, "table"),
]
_COMPILED = [(re.compile(p + (ext_pat if ext_pat else ""), re.I), cls, ext, tier)
             for (p, cls, ext, tier) in LEXICON
             for ext_pat in [{"prep": _EXT_PREP, "always": _EXT_ALWAYS, None: ""}[ext]]]

def normalize_plan(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "").replace("_", " ")).strip().lower()
    return t.rstrip(" .")


def split_first_segment(text: str) -> Tuple[str, str]:
    """('A', '<delimiter>B') for two-stage plans "A, then B" / "A and then B" (delegated to
    ``labels.plan_first_segment``); ('A', '') otherwise.  ``text`` is expected to be ``normalize_plan``-ed."""
    first = L.plan_first_segment(text)
    if first and len(first) < len(text) and text.startswith(first):
        return first, text[len(first):]
    return text, ""


def scan_plan(text: str) -> List[dict]:
    """All lexicon matches in reading order, each ``{start, end, cls, phrase, tier, dir}``; spans never overlap
    (matching resumes after the previous span, so an object clause absorbed by a verb is not re-scanned)."""
    out, pos = [], 0
    while pos < len(text):
        best = None
        for pat, cls, _ext, tier in _COMPILED:
            m = pat.search(text, pos)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, cls, tier)
        if best is None:
            break
        m, cls, tier = best
        d = m.groupdict().get("dir") if "dir" in pat_groups(m) else None
        out.append({"start": m.start(), "end": m.end(), "cls": cls, "phrase": m.group(0), "tier": tier, "dir": d})
        pos = max(m.end(), pos + 1)
    return out


def pat_groups(m: re.Match) -> Dict[str, Optional[str]]:
    try:
        return m.groupdict()
    except Exception:
        return {}




# =============================================================================================================
# Rewrite
# =============================================================================================================
def target_verb(motion_lon: str, v0: float) -> str:
    if motion_lon == "keep":
        return "cruise"
    if motion_lon == "accelerate":
        return "proceed" if v0 < V_STOP else "accelerate"
    if motion_lon == "decelerate":
        return "decelerate"
    if motion_lon == "stop":
        return "remain stopped" if v0 < V_STOP else "stop and wait"
    raise ValueError(f"unknown motion_lon {motion_lon!r}")


_TRAIL_CONNECTOR = re.compile(r"\s*(?:,\s*)?(?:and|to|then|while|,)\s*$", re.I)
_LEAD_CONNECTOR = re.compile(r"^\s*(?:and|then|to|while|,)\s+", re.I)


def _cleanup(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ",", s)
    s = _LEAD_CONNECTOR.sub("", s)
    s = re.sub(r"\s+(?:and|to|then|while|,)\s*$", "", s, flags=re.I)
    s = re.sub(r"\b(and|then|while)\s+(and|then|while)\b", r"\1", s)
    return s.strip(" ,")


def rewrite_plan(plan_text: str, motion_lon: str, v0: float) -> Tuple[Optional[str], dict]:
    """Rewrite the longitudinal verb phrase(s) of the first segment to the standard verb of ``motion_lon``.

    The first longitudinal phrase is replaced, any further longitudinal phrases of the same segment are deleted
    (with their connector), everything else (lateral wording, the second stage of a two-stage plan) is kept.
    The candidate is then verified with ``labels.plan_to_motion`` (the shared matcher).
    Returns ``(new_plan, info)``; ``new_plan`` is None when no longitudinal phrase could be located or when the
    candidate does not read as ``motion_lon`` under the shared matcher (``info['reason']`` says which).
    """
    text = normalize_plan(plan_text)
    first, rest = split_first_segment(text)
    ms = [m for m in scan_plan(first) if m["cls"] in LON_CLASSES or m["cls"] == "v0"]
    if not ms:
        return None, {"reason": "no longitudinal phrase", "first_segment": first}
    verb = target_verb(motion_lon, v0)
    pieces, cursor = [], 0
    for i, m in enumerate(ms):
        before = first[cursor:m["start"]]
        if i == 0:
            pieces.append(before)
            pieces.append(verb)
        else:
            pieces.append(_TRAIL_CONNECTOR.sub(" ", before))
        cursor = m["end"]
    pieces.append(first[cursor:])
    new_first = _cleanup("".join(pieces))
    if not new_first:
        new_first = verb
    new_text = new_first + rest
    info = {"target_verb": verb, "replaced": [m["phrase"] for m in ms], "first_segment": first}
    got = plan_to_motion(new_text, v0)["lon"]
    if got != motion_lon:
        info.update(reason=f"rewrite not confirmed by labels.plan_to_motion (reads as {got!r})", candidate=new_text)
        return None, info
    return new_text, info


# =============================================================================================================
# Driver
# =============================================================================================================
def _load(path: str):
    with open(path) as fh:
        return json.load(fh)


def frame_plan_record(fdir: str) -> dict:
    """Worker: final_plan + GT motion class + v0 / v_end / start_t (compatibility-matrix inputs) of one frame."""
    rec = {"fdir": fdir, "ok": False, "partition": partition_of(fdir), "plan": "", "has_reason": False,
           "motion_lon": None, "motion_lat": None, "v0": 0.0, "v_end": None, "start_t": None}
    try:
        fr = _load(os.path.join(fdir, "frame.json"))
        pg = _load(os.path.join(fdir, "panorama_geo.json"))
    except Exception as e:
        rec["error"] = f"load: {e}"
        return rec
    intent = fr.get("intent_corrected") or fr.get("intent") or "GO_STRAIGHT"
    rec["plan"] = str(pg.get("final_plan") or "").strip()
    rec["has_reason"] = bool(pg.get("reason"))
    try:
        ml = L.motion_label(fr, intent)
        rec["motion_lon"], rec["motion_lat"] = ml["lon"], ml["lat"]
        rec["v0"] = float(L.v0_of(fr))
        ve, st = L.future_v_end(fr), L.future_start_time(fr)
        rec["v_end"] = None if ve is None else float(ve)
        rec["start_t"] = None if st is None else float(st)
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"labels: {e}"
    return rec


def repair_records(records: Sequence[dict], max_examples: int = 60) -> Tuple[Dict[str, dict], Dict]:
    """Apply the repair rule to worker records; returns ``(overlay, stats)``.

    Contradiction = ``labels.plan_consistent`` (tolerant compatibility matrix) on the longitudinal part is
    ``False``; ``None`` (unspecified / unmatched) and ``True`` (equal or compatible class) are never rewritten.
    Records may omit ``v_end`` / ``start_t`` (then only the class-set cells of the matrix apply).
    """
    overlay: Dict[str, dict] = {}
    st = Counter()
    confusion = Counter()
    transitions = Counter()
    tolerated_tr = Counter()
    unmatched = Counter()
    unspecified = Counter()
    by_part: Dict[str, Counter] = {}
    examples, failed, tolerated_ex = [], [], []
    lat_conflicts = 0
    for r in records:
        st["frames"] += 1
        bp = by_part.setdefault(r["partition"], Counter())
        bp["frames"] += 1
        if not r["ok"]:
            st["unreadable_or_label_error"] += 1
            continue
        plan = r["plan"]
        if not plan:
            st["missing_plan"] += 1
            bp["missing_plan"] += 1
            continue
        v0, mlon = r["v0"], r["motion_lon"]
        v_end, start_t = r.get("v_end"), r.get("start_t")
        pm = plan_to_motion(plan, v0)
        pcls = pm["lon"]
        st[f"plan_class_{pcls}"] += 1
        confusion[f"{pcls}->{mlon}"] += 1
        if pcls == "unmatched":
            unmatched[normalize_plan(plan)] += 1
            continue
        if pcls == "unspecified":
            unspecified[normalize_plan(plan)] += 1
            continue
        # longitudinal judgement (bare string motion -> lateral not judged) + full judgement for the lateral count
        pc_lon = plan_consistent(plan, mlon, v0, v_end=v_end, start_t=start_t)
        pc_full = plan_consistent(plan, {"lon": mlon, "lat": r["motion_lat"]}, v0, v_end=v_end, start_t=start_t)
        if pc_full is False and pc_lon is True:
            lat_conflicts += 1
        if pcls != mlon:
            st["lon_inconsistent_strict"] += 1
            bp["lon_inconsistent_strict"] += 1
        if pc_lon is not False:
            st["lon_consistent"] += 1
            bp["lon_consistent"] += 1
            if pcls != mlon:                                    # compatible under the matrix, not equal
                st["tolerated"] += 1
                bp["tolerated"] += 1
                tolerated_tr[f"{pcls}->{mlon}"] += 1
                if len(tolerated_ex) < max_examples and not any(e["plan"] == plan and e["motion_lon"] == mlon for e in tolerated_ex):
                    tolerated_ex.append({"plan": plan, "plan_lon": pcls, "motion_lon": mlon, "v0": round(v0, 2),
                                         "v_end": None if v_end is None else round(v_end, 2), "start_t": start_t})
            continue
        st["lon_inconsistent"] += 1
        bp["lon_inconsistent"] += 1
        new, info = rewrite_plan(plan, mlon, v0)
        if new is None:
            st["rewrite_failed"] += 1
            if len(failed) < max_examples:
                failed.append({"fdir": r["fdir"], "plan": plan, "plan_lon": pcls, "motion_lon": mlon, "info": info})
            continue
        st["rewritten"] += 1
        bp["rewritten"] += 1
        transitions[f"{pcls}->{mlon}"] += 1
        overlay[r["fdir"]] = {"final_plan": new, "reason_weight": REASON_WEIGHT_REWRITTEN, "orig_plan": plan,
                              "motion_lon": mlon, "plan_lon": pcls, "matched_by": pm["matched_by"], "v0": round(v0, 2),
                              "v_end": None if v_end is None else round(v_end, 2), "start_t": start_t}
        if len(examples) < max_examples and not any(e["orig"] == plan and e["motion_lon"] == mlon for e in examples):
            examples.append({"orig": plan, "new": new, "plan_lon": pcls, "motion_lon": mlon, "v0": round(v0, 2)})
    n_judged = st["lon_consistent"] + st["lon_inconsistent"]
    stats = {
        "counts": dict(sorted(st.items())),
        "n_judged": n_judged,
        "consistency_rule": CONSISTENCY_RULE,
        "inconsistency_rate": round(st["lon_inconsistent"] / max(n_judged, 1), 4),
        "inconsistency_rate_strict": round(st["lon_inconsistent_strict"] / max(n_judged, 1), 4),
        "tolerated_rate": round(st["tolerated"] / max(n_judged, 1), 4),
        "rewrite_rate_of_frames": round(st["rewritten"] / max(st["frames"], 1), 4),
        "lateral_conflicts_with_lon_ok": lat_conflicts,
        "confusion_plan_to_motion": dict(sorted(confusion.items())),
        "rewrite_transitions": dict(sorted(transitions.items())),
        "tolerated_transitions": dict(sorted(tolerated_tr.items())),
        "tolerated_examples": tolerated_ex,
        "by_partition": {p: dict(sorted(c.items())) for p, c in sorted(by_part.items(), key=lambda t: int(t[0][1:]) if t[0][1:].isdigit() else 999)},
        "unmatched_plans_top": [{"plan": p, "n": n} for p, n in unmatched.most_common(80)],
        "n_unmatched_distinct": len(unmatched),
        "unspecified_plans_top": [{"plan": p, "n": n} for p, n in unspecified.most_common(40)],
        "rewrite_examples": examples,
        "rewrite_failed_examples": failed,
    }
    return overlay, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(DATA, "index_train.json"))
    ap.add_argument("--partitions", nargs="*", default=S2_PARTITIONS, help="frames of these partitions only ('all' = no filter)")
    ap.add_argument("--out", default=os.path.join(DATA, "plan_repair.json"))
    ap.add_argument("--stats-out", default=None, help="default: <out dir>/plan_repair_v2_stats.json")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N (filtered) index rows only")
    args = ap.parse_args(argv)

    with open(args.index) as fh:
        index = json.load(fh)
    parts = None if (not args.partitions or args.partitions == ["all"]) else set(args.partitions)
    frames = [f for f in index if parts is None or partition_of(f) in parts]
    if args.limit:
        frames = frames[: args.limit]
    print(f"[plan_repair] frames={len(frames)} (of {len(index)}) partitions={sorted(parts) if parts else 'all'} "
          f"labels={LABEL_SOURCE} plan_fn={PLAN_FN_SOURCE}", flush=True)
    t0 = time.time()
    workers = max(1, min(args.workers, max(len(frames), 1)))
    if workers == 1:
        records = [frame_plan_record(f) for f in frames]
    else:
        with Pool(workers) as pool:
            records = list(pool.imap(frame_plan_record, frames, chunksize=64))
    overlay, stats = repair_records(records)
    stats.update({"created": _dt.datetime.now().isoformat(timespec="seconds"), "index_file": os.path.abspath(args.index),
                  "partitions": sorted(parts) if parts else "all", "limit": args.limit, "label_source": LABEL_SOURCE,
                  "plan_fn_source": PLAN_FN_SOURCE, "reason_weight_rewritten": REASON_WEIGHT_REWRITTEN,
                  "elapsed_s": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(overlay, fh, indent=0)
    stats_out = args.stats_out or os.path.join(os.path.dirname(os.path.abspath(args.out)), "plan_repair_v2_stats.json")
    with open(stats_out, "w") as fh:
        json.dump(stats, fh, indent=1)
    c = stats["counts"]
    print(f"frames={c.get('frames', 0)} missing_plan={c.get('missing_plan', 0)} unmatched={c.get('plan_class_unmatched', 0)} "
          f"unspecified={c.get('plan_class_unspecified', 0)} judged={stats['n_judged']} inconsistent={c.get('lon_inconsistent', 0)} "
          f"({stats['inconsistency_rate']:.1%}; strict class-equality {c.get('lon_inconsistent_strict', 0)} = "
          f"{stats['inconsistency_rate_strict']:.1%}, tolerated {c.get('tolerated', 0)}) "
          f"rewritten={c.get('rewritten', 0)} failed={c.get('rewrite_failed', 0)}")
    print("transitions:", stats["rewrite_transitions"])
    print("tolerated:", stats["tolerated_transitions"])
    for e in stats["rewrite_examples"][:12]:
        print(f"  [{e['plan_lon']}->{e['motion_lon']} v0={e['v0']}] {e['orig']!r} -> {e['new']!r}")
    print(f"overlay -> {args.out} ({len(overlay)} frames)\nstats -> {stats_out}  ({stats['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
