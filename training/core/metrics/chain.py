"""Chain-quality metrics for S2 answers.

Record format (one per evaluated frame)::

    {'pred':   parse_output(...) dict (or the raw generated text),
     'frame':  frame.json dict (past_states / future_states / ego_behavior / intent*),
     'pred20': (20, 2) up-sampled predicted trajectory in the ego frame, or None (no valid <traj>),
     # optional -------------------------------------------------------------------------
     'text':   raw generated text (degeneracy heuristics),
     'n_tokens': int, 'max_new_tokens': int, 'finish_reason': 'length'|'stop'   (cap statistics),
     'v0':     float (defaults to |past_states.vel[-1]|),
     'gt_ego_state': labels.ego_state_label(frame), 'gt_motion': labels.motion_label(frame, intent),
     'gt_sample_class': labels.sample_class(frame)}     # if absent -> derived with labels.py

Key aliases (the records built by ``evaluation.eval_dev.make_record`` are accepted as-is):
``pred_xy`` = ``pred20``; ``gt20`` (else derived from ``frame``); ``ego_state_gt`` = ``gt_ego_state``;
``motion_gt`` = ``gt_motion``; ``sample_class`` = ``gt_sample_class``; ``hit_cap`` (bool) and
``n_new_tokens`` = ``n_tokens`` for the cap statistics; ``pano['final_plan']`` for the GT reference row.

GT labels: values supplied in the record win; otherwise they are derived from ``frame`` with
``training.core.labels`` -- the single implementation of the label rules and of the plan matcher
(``plan_to_motion`` / ``plan_consistent``; this module carries no copy of it).

The plan consistency rates use ``labels.plan_consistent`` with its tolerant default (the
plan/motion compatibility matrix, where "not contradictory" counts as consistent); the stricter
class-equality rule is reported alongside as ``plan_motion_consistency_strict`` /
``plan_traj_consistency_strict`` / ``gt_plan_motion_consistency_strict``
(+ ``plan_traj_inconsistency_rate_strict``).  The matrix inputs ``v_end`` / ``start_t`` come from
the *predicted* trajectory (``pred20``) for the pred-plan rates (only the class-set cells apply
when no trajectory was emitted) and from the GT future (``gt20`` / ``frame``) for the GT reference
row; ``v0`` is the shared history end speed.  When a record supplies ``gt20`` but the frame has no
``future_states``, the GT motion / sample class fall back to the trajectory-derived rules of
``metrics.traj`` (the functions that classify *predicted* trajectories).  The only local rules kept
here are ``ego_state_rule`` (a self-contained restatement of the ego-state rule, readable on its own
and cross-checkable against ``labels.ego_state_label``), the trajectory-derived classes and the
degeneracy heuristics.

Metrics (all rates over the frames for which the quantity is defined; ``None`` = no support):
  ego_state_present_rate, ego_state_lon_acc (lon class vs rule), ego_state_lat_acc (vs human
  ego_behavior.lateral), ego_state_acc (both; a missing segment counts as wrong), ego_state_lon_confusion,
  ego_state_stop_time_mae, ego_state_speed_mae
  motion_present_rate, motion_lon_acc, motion_lat_acc, motion_acc, motion_lon_confusion
  plan_motion_consistency (pred plan vs pred motion.lon, plan matcher + compatibility matrix;
  unmatched/unspecified excluded), plan_traj_consistency (pred plan vs class of the predicted
  trajectory), plan_unspecified_rate, plan_unmatched_rate, plan_traj_inconsistency_rate
  (1 - plan_traj_consistency; the quality gate on plan/trajectory contradictions),
  plan_motion_consistency_strict / plan_traj_consistency_strict / plan_traj_inconsistency_rate_strict (class equality),
  motion_traj_consistency (pred motion.lon == lon class of the predicted trajectory),
  motion_traj_lat_consistency (heading proxy), gt_plan_motion_consistency (reference row: GT plan vs GT motion,
  matrix inputs from the GT future) and gt_plan_motion_consistency_strict
  plan_tax_* (closed-set final_plan accuracy over the dataset's own plan vocabulary --
  10 lateral x 4 longitudinal via ``labels.plan_to_taxonomy``; per axis acc / macro_recall /
  macro_recall_min10 / majority_baseline / per_class / confusion, plus plan_tax_joint_acc,
  plan_tax_pred_rate and the counts excluded for stating no longitudinal intent.  This is the
  REFERENCE-BASED plan metric and it has a noise floor, because the GT plan and the GT trajectory
  do not always agree, so report it beside the reference-free plan_*_consistency rates),
  reason_missing_rate, plan_missing_rate, implication_missing_rate (per emitted object),
  ego_state_missing_rate, motion_missing_rate, traj_missing_rate, missing_rate (mean of reason/plan/implication)
  start_frame_confusion, start_frame_acc, start_pred_static_rate, start_frame_motion_acc, stay_frame_acc,
  n_start_frames, n_stay_frames
  cap_hit_rate, traj_unclosed_rate, unclosed_obj_rate, text_repeat_rate, traj_arith_rate, traj_flat_rate,
  obj_dup_rate, degenerate_rate
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from training.core import labels as L
from training.core.paths import DT
from training.core.parse_output import parse_output, parse_ego_lon, norm_motion_lon, norm_motion_lat
from training.core.metrics.traj import (
    as_xy20, speeds_from_xy, sample_class_from_xy, motion_lon_from_xy, motion_lat_from_xy, SAMPLE_CLASSES,
)

__all__ = [
    "chain_metrics", "ego_state_rule", "v0_of_frame", "gt_future_xy", "text_repeat", "traj_is_arith",
]

STOP_V = 0.5
DV_THR = 0.6
STOP_CAP = 4.0


# --------------------------------------------------------------------------- frame helpers
def v0_of_frame(frame: Dict[str, Any]) -> float:
    """Current speed |vel[-1]| (m/s) from ``past_states``."""
    ps = frame.get("past_states") or {}
    vx, vy = ps.get("vel_x") or [], ps.get("vel_y") or []
    if not vx or not vy:
        return 0.0
    return float(math.hypot(vx[-1], vy[-1]))


def gt_future_xy(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    """GT future (20, 2) from ``future_states`` (``None`` if absent)."""
    fs = frame.get("future_states") or {}
    px, py = fs.get("pos_x") or [], fs.get("pos_y") or []
    n = min(len(px), len(py))
    if n == 0:
        return None
    return as_xy20([(px[i], py[i]) for i in range(n)])


def ego_state_rule(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based ego_state, a self-contained restatement of ``labels.ego_state_label``.

    ``chain_metrics`` derives GT ego_state with ``labels.ego_state_label``; this copy spells the same
    rule out in one place so the two can be compared.  Returns ``{'lon_class', 'lat', 'T' (stopped
    seconds, capped 4.0), 'v0', 'v1', 'lon', 'text'}``.
    """
    ps = frame.get("past_states") or {}
    vx, vy = ps.get("vel_x") or [], ps.get("vel_y") or []
    sp = [math.hypot(a, b) for a, b in zip(vx, vy)]
    v0 = sp[-1] if sp else 0.0
    v1 = sp[-5] if len(sp) >= 5 else (sp[0] if sp else 0.0)
    lat = (frame.get("ego_behavior") or {}).get("lateral") or "n/a"
    out: Dict[str, Any] = {"lat": lat, "v0": v0, "v1": v1, "T": None}
    if v0 < STOP_V:
        k = 0
        for s in reversed(sp):
            if s < STOP_V:
                k += 1
            else:
                break
        T = k * DT
        out["lon_class"] = "stopped"
        out["T"] = min(T, STOP_CAP)
        out["lon"] = "stopped 4.0s+" if T >= STOP_CAP else f"stopped {T:.1f}s"
    else:
        dv = v0 - v1
        if dv > DV_THR:
            out["lon_class"] = "accelerating"; out["lon"] = f"accelerating {v1:.1f}->{v0:.1f} m/s"
        elif dv < -DV_THR:
            out["lon_class"] = "decelerating"; out["lon"] = f"decelerating {v1:.1f}->{v0:.1f} m/s"
        else:
            out["lon_class"] = "cruising"; out["lon"] = f"cruising {v0:.1f} m/s"
    out["text"] = f"lon={out['lon']} | lat={lat}"
    return out


# --------------------------------------------------------------------------- degeneracy heuristics
_REPEAT = re.compile(r"(.{25,200}?)\1\1", re.S)


def text_repeat(text: Optional[str]) -> bool:
    """True when a chunk of >= 25 chars is repeated >= 3 times consecutively (verbatim looping)."""
    if not text:
        return False
    return _REPEAT.search(text) is not None


def traj_is_arith(points5: Optional[Sequence[Sequence[float]]], tol: float = 0.05) -> bool:
    """True when the 5 waypoints (from the origin) advance by the SAME non-zero step every second
    (an arithmetic progression)."""
    if not points5 or len(points5) < 2:
        return False
    P = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in points5]
    d = [(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]) for i in range(len(P) - 1)]
    if math.hypot(*d[0]) < tol:
        return False
    return all(abs(a[0] - d[0][0]) <= tol and abs(a[1] - d[0][1]) <= tol for a in d[1:])


def _gt_deltas_vary(gt20: Optional[np.ndarray], thr: float = 0.5) -> bool:
    """Whether the GT 1 Hz steps differ by more than ``thr`` m (i.e. GT is NOT itself arithmetic)."""
    if gt20 is None:
        return False
    idx = [3, 7, 11, 15, 19]
    P = [np.zeros(2)] + [gt20[i] for i in idx]
    steps = [float(np.hypot(*(P[i + 1] - P[i]))) for i in range(5)]
    return (max(steps) - min(steps)) > thr


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _first(rec: Dict[str, Any], *keys: str) -> Any:
    """First non-None value among alias keys of a record."""
    for k in keys:
        v = rec.get(k)
        if v is not None:
            return v
    return None


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


class _Tax:
    """final_plan closed-set accumulator: 10 lateral x 4 longitudinal.

    GT ``unspecified`` / ``unmatched`` longitudinals are EXCLUDED: a lateral-only plan states
    nothing about speed, and folding it into ``cruise`` would assert something the plan did not
    say; the excluded count is reported.  Same for a GT lateral the parser could not place
    (``other``).
    """

    def __init__(self) -> None:
        self.conf: Dict[str, Dict[str, Dict[str, int]]] = {"lat": {}, "lon": {}}
        self.tot = {"lat": 0, "lon": 0}
        self.ok = {"lat": 0, "lon": 0}
        self.joint_tot = self.joint_ok = 0
        self.excluded_lon = self.other_lat = 0
        self.n_gt = self.n_pred = 0

    def add(self, gt: Dict[str, Any], pr: Optional[Dict[str, Any]]) -> None:
        self.n_gt += 1
        self.n_pred += int(pr is not None)
        axes: Dict[str, Optional[bool]] = {}
        for ax, key, bad in (("lat", "lateral", ("other", None)),
                             ("lon", "longitudinal", ("unspecified", "unmatched", None))):
            g = gt.get(key)
            if g in bad:
                if ax == "lon":
                    self.excluded_lon += 1
                else:
                    self.other_lat += 1
                axes[ax] = None
                continue
            pv = (pr or {}).get(key)
            row = self.conf[ax].setdefault(g, {})
            lab = pv if pv is not None else "<missing>"
            row[lab] = row.get(lab, 0) + 1
            self.tot[ax] += 1
            hit = pv == g
            self.ok[ax] += int(hit)
            axes[ax] = hit
        if axes.get("lat") is not None and axes.get("lon") is not None:
            self.joint_tot += 1
            self.joint_ok += int(bool(axes["lat"]) and bool(axes["lon"]))

    def report(self, ax: str, labels: Sequence[str]) -> Dict[str, Any]:
        per: Dict[str, Dict[str, Any]] = {}
        for g in labels:
            row = self.conf[ax].get(g, {})
            ng = sum(row.values())
            if ng:
                per[g] = {"n": ng, "recall": row.get(g, 0) / ng}
        rec_all = [v["recall"] for v in per.values()]
        rec_10 = [v["recall"] for v in per.values() if v["n"] >= 10]
        maj = max(((v["n"], g) for g, v in per.items()), default=(0, None))
        tot = self.tot[ax]
        return {
            "n": tot,
            "acc": _rate(self.ok[ax], tot),
            "macro_recall": (sum(rec_all) / len(rec_all)) if rec_all else None,
            "macro_recall_min10": (sum(rec_10) / len(rec_10)) if rec_10 else None,
            "majority_class": maj[1],
            "majority_baseline": _rate(maj[0], tot),
            "per_class": per,
            "confusion": self.conf[ax],
        }


# --------------------------------------------------------------------------- main entry
def chain_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute chain-quality metrics over ``records`` (format and metric list in the module docstring)."""
    n = 0
    ego_present = ego_lon_ok = ego_lat_ok = ego_ok = 0
    ego_lon_conf: Counter = Counter()
    stop_T_err: List[float] = []; speed_err: List[float] = []
    mot_present = mot_lon_ok = mot_lat_ok = mot_ok = 0
    mot_lon_conf: Counter = Counter()
    n_lat_judged = 0
    pm_ok = pm_tot = pm_ok_s = 0; pt_ok = pt_tot = pt_ok_s = 0
    plan_unspec = plan_unmatch = plan_judgeable = 0
    gt_pm_ok = gt_pm_tot = gt_pm_ok_s = 0
    tax = _Tax()
    mt_ok = mt_tot = 0; mtl_ok = mtl_tot = 0
    miss_reason = miss_plan = miss_ego = miss_motion = miss_traj = 0
    impl_tot = impl_miss = 0
    start_conf: Counter = Counter(); stay_conf: Counter = Counter()
    n_start = n_stay = start_ok = stay_ok = start_static = 0
    start_mot_ok = start_mot_tot = 0
    cap_hit = cap_tot = 0
    traj_unclosed = unclosed_obj = rep = arith = flat = dup = degenerate = 0
    for rec in records:
        pred = rec["pred"]
        if isinstance(pred, str):
            pred = parse_output(pred)
        frame = rec.get("frame") or {}
        text = rec.get("text")
        n += 1
        v0 = float(rec["v0"]) if rec.get("v0") is not None else v0_of_frame(frame)
        intent = frame.get("intent_corrected") or frame.get("intent")
        gt20 = as_xy20(rec["gt20"]) if rec.get("gt20") is not None else gt_future_xy(frame)
        p20 = as_xy20(_first(rec, "pred20", "pred_xy"))

        # ---------------- GT labels (record > labels.py > trajectory-derived rules of metrics.traj)
        g_ego = _first(rec, "gt_ego_state", "ego_state_gt")
        if g_ego is None:
            g_ego = L.ego_state_label(frame)
        g_lon_cls = g_ego.get("lon_class") or parse_ego_lon(g_ego.get("lon")).get("cls")
        g_lat = str(g_ego.get("lat") or "n/a")
        g_ego_num = parse_ego_lon(g_ego.get("lon"))
        has_future = bool((frame.get("future_states") or {}).get("pos_x"))
        g_mot = _first(rec, "gt_motion", "motion_gt")
        if g_mot is None and gt20 is not None:
            if has_future:
                g_mot = L.motion_label(frame, intent)
            else:   # gt20 supplied without the frame's future: heading-only lateral proxy
                g_mot = {"lon": motion_lon_from_xy(gt20, v0), "lat": motion_lat_from_xy(gt20), "_lat_proxy": True}
        g_cls = _first(rec, "gt_sample_class", "sample_class")
        if g_cls is None and gt20 is not None:
            g_cls = L.sample_class(frame) if has_future else sample_class_from_xy(gt20, v0)
        p2m, pcons = L.plan_to_motion, L.plan_consistent          # plan matcher: single implementation
        # compatibility-matrix inputs (v_end = mean speed of the last 1 s, start_t = first t with v >= 0.5 m/s)
        p_sp = speeds_from_xy(p20) if p20 is not None else []
        p_vend, p_start = L.v_end_from_speeds(p_sp), L.start_time_from_speeds(p_sp)
        g_sp = speeds_from_xy(gt20) if gt20 is not None else []
        g_vend, g_start = L.v_end_from_speeds(g_sp), L.start_time_from_speeds(g_sp)

        # ---------------- ego_state
        es = pred.get("ego_state")
        if es is None:
            miss_ego += 1
            ego_lon_conf[f"{g_lon_cls}->none"] += 1
        else:
            ego_present += 1
            p_lon_cls = es.get("lon_class") or parse_ego_lon(es.get("lon")).get("cls")
            lon_ok = p_lon_cls is not None and p_lon_cls == g_lon_cls
            lat_ok = str(es.get("lat") or "n/a").strip().upper() == g_lat.strip().upper()
            ego_lon_ok += int(lon_ok); ego_lat_ok += int(lat_ok); ego_ok += int(lon_ok and lat_ok)
            ego_lon_conf[f"{g_lon_cls}->{p_lon_cls}"] += 1
            pn = parse_ego_lon(es.get("lon"))
            if lon_ok and g_lon_cls == "stopped" and pn["T"] is not None and g_ego_num["T"] is not None:
                stop_T_err.append(abs(pn["T"] - g_ego_num["T"]))
            if lon_ok and g_lon_cls in ("accelerating", "decelerating", "cruising"):
                pv = pn["v_to"] if g_lon_cls != "cruising" else pn["v"]
                gv = g_ego_num["v_to"] if g_lon_cls != "cruising" else g_ego_num["v"]
                if gv is None:
                    gv = g_ego.get("v0")
                if pv is not None and gv is not None:
                    speed_err.append(abs(pv - float(gv)))

        # ---------------- motion
        mo = pred.get("motion")
        p_mlon = norm_motion_lon((mo or {}).get("lon")) if mo else None
        p_mlat = norm_motion_lat((mo or {}).get("lat")) if mo else None
        if mo is None:
            miss_motion += 1
        else:
            mot_present += 1
        if g_mot is not None:
            g_mlon, g_mlat = g_mot.get("lon"), g_mot.get("lat")
            lon_ok = p_mlon is not None and p_mlon == g_mlon
            mot_lon_ok += int(lon_ok)
            mot_lon_conf[f"{g_mlon}->{p_mlon}"] += 1
            lat_ok = None
            if g_mlat is not None:
                n_lat_judged += 1
                lat_ok = p_mlat is not None and p_mlat == g_mlat
                mot_lat_ok += int(lat_ok)
            mot_ok += int(lon_ok and (lat_ok is not False))

        # ---------------- plan consistency
        plan = pred.get("final_plan")
        if plan:
            c = p2m(plan, v0)
            if c.get("lon") == "unspecified":
                plan_unspec += 1
            elif c.get("lon") == "unmatched":
                plan_unmatch += 1
            else:
                plan_judgeable += 1
            if mo is not None and p_mlon is not None:
                pmot = {"lon": p_mlon, "lat": p_mlat}
                ok = pcons(plan, pmot, v0, v_end=p_vend, start_t=p_start)
                if ok is not None:
                    pm_tot += 1; pm_ok += int(ok)
                    pm_ok_s += int(bool(pcons(plan, pmot, v0, strict=True)))
            if p20 is not None:
                tmot = {"lon": motion_lon_from_xy(p20, v0), "lat": motion_lat_from_xy(p20)}
                ok = pcons(plan, tmot, v0, v_end=p_vend, start_t=p_start)
                if ok is not None:
                    pt_tot += 1; pt_ok += int(ok)
                    pt_ok_s += int(bool(pcons(plan, tmot, v0, strict=True)))
        gt_plan = (rec.get("pano") or {}).get("final_plan")
        if gt_plan:                                   # ---- final_plan closed-set accuracy
            tax.add(L.plan_to_taxonomy(gt_plan, v0),
                    L.plan_to_taxonomy(plan, v0) if plan else None)
        if gt_plan and g_mot is not None:
            gmot = {"lon": g_mot.get("lon"), "lat": g_mot.get("lat")}
            ok = pcons(gt_plan, gmot, v0, v_end=g_vend, start_t=g_start)
            if ok is not None:
                gt_pm_tot += 1; gt_pm_ok += int(ok)
                gt_pm_ok_s += int(bool(pcons(gt_plan, gmot, v0, strict=True)))

        # ---------------- motion vs predicted trajectory
        if p20 is not None and p_mlon is not None:
            mt_tot += 1; mt_ok += int(motion_lon_from_xy(p20, v0) == p_mlon)
        if p20 is not None and p_mlat is not None:
            mtl_tot += 1; mtl_ok += int(motion_lat_from_xy(p20) == p_mlat)

        # ---------------- missing
        miss_reason += int(not pred.get("reason"))
        miss_plan += int(not plan)
        miss_traj += int(p20 is None)
        for o in pred.get("objects") or []:
            impl_tot += 1
            impl_miss += int(not o.get("implication"))

        # ---------------- start-frame confusion
        if g_cls is not None and p20 is not None:
            p_cls = sample_class_from_xy(p20, v0)
        else:
            p_cls = "none"
        if g_cls == "start":
            n_start += 1; start_conf[p_cls] += 1
            start_ok += int(p_cls == "start"); start_static += int(p_cls == "stay")
            if p_mlon is not None:
                start_mot_tot += 1; start_mot_ok += int(p_mlon == "accelerate")
        elif g_cls == "stay":
            n_stay += 1; stay_conf[p_cls] += 1; stay_ok += int(p_cls == "stay")

        # ---------------- cap / degeneracy
        fr = rec.get("finish_reason"); hc = rec.get("hit_cap")
        nt = _first(rec, "n_tokens", "n_new_tokens"); mx = rec.get("max_new_tokens")
        if fr is not None or hc is not None or (nt is not None and mx is not None):
            cap_tot += 1
            cap_hit += int(fr == "length" or bool(hc) or (nt is not None and mx is not None and nt >= mx))
        traj_unclosed += int(pred.get("traj_n_points", 0) > 0 and not pred.get("traj_closed", False))
        unclosed_obj += int(pred.get("n_unclosed_obj", 0) > 0)
        r = text_repeat(text)
        a = traj_is_arith(pred.get("traj")) and _gt_deltas_vary(gt20)
        pts5 = pred.get("traj")
        f = False
        if pts5:
            sp5 = speeds_from_xy(pts5, dt=1.0)
            f = (max(sp5) - min(sp5)) < 0.5 and max(sp5) >= STOP_V
        objs = pred.get("objects") or []
        keys = [(str(o.get("type")).lower(), o.get("point")) for o in objs if o.get("point") is not None]
        d = len(keys) != len(set(keys))
        rep += int(r); arith += int(a); flat += int(f); dup += int(d)
        degenerate += int(r or a or d)

    m: Dict[str, Any] = {"n_frames": n}
    m["ego_state_present_rate"] = _rate(ego_present, n)
    m["ego_state_lon_acc"] = _rate(ego_lon_ok, n)
    m["ego_state_lat_acc"] = _rate(ego_lat_ok, n)
    m["ego_state_acc"] = _rate(ego_ok, n)
    m["ego_state_lon_confusion"] = dict(ego_lon_conf)
    m["ego_state_stop_time_mae"] = _mean(stop_T_err)
    m["ego_state_speed_mae"] = _mean(speed_err)
    m["motion_present_rate"] = _rate(mot_present, n)
    m["motion_lon_acc"] = _rate(mot_lon_ok, n)
    m["motion_lat_acc"] = _rate(mot_lat_ok, n_lat_judged)
    m["motion_acc"] = _rate(mot_ok, n)
    m["motion_lon_confusion"] = dict(mot_lon_conf)
    m["plan_motion_consistency"] = _rate(pm_ok, pm_tot)
    m["plan_traj_consistency"] = _rate(pt_ok, pt_tot)
    m["plan_traj_inconsistency_rate"] = None if not pt_tot else 1.0 - pt_ok / pt_tot
    m["plan_motion_consistency_strict"] = _rate(pm_ok_s, pm_tot)
    m["plan_traj_consistency_strict"] = _rate(pt_ok_s, pt_tot)
    m["plan_traj_inconsistency_rate_strict"] = None if not pt_tot else 1.0 - pt_ok_s / pt_tot
    n_plans = plan_unspec + plan_unmatch + plan_judgeable
    m["plan_unspecified_rate"] = _rate(plan_unspec, n_plans)
    m["plan_unmatched_rate"] = _rate(plan_unmatch, n_plans)
    m["plan_tax_n_with_gt"] = tax.n_gt
    m["plan_tax_pred_rate"] = _rate(tax.n_pred, tax.n_gt)
    m["plan_tax_lon_excluded"] = tax.excluded_lon       # GT plans that state no longitudinal intent
    m["plan_tax_lat_gt_other"] = tax.other_lat
    m["plan_tax_joint_acc"] = _rate(tax.joint_ok, tax.joint_tot)
    for _ax, _rep in (("lat", tax.report("lat", L.LAT_TAXONOMY)),
                      ("lon", tax.report("lon", L.LON_TAXONOMY))):
        for _k, _v in _rep.items():
            m[f"plan_tax_{_ax}_{_k}"] = _v
    m["gt_plan_motion_consistency"] = _rate(gt_pm_ok, gt_pm_tot)
    m["gt_plan_motion_consistency_strict"] = _rate(gt_pm_ok_s, gt_pm_tot)
    m["motion_traj_consistency"] = _rate(mt_ok, mt_tot)
    m["motion_traj_lat_consistency"] = _rate(mtl_ok, mtl_tot)
    m["reason_missing_rate"] = _rate(miss_reason, n)
    m["plan_missing_rate"] = _rate(miss_plan, n)
    m["implication_missing_rate"] = _rate(impl_miss, impl_tot)
    m["ego_state_missing_rate"] = _rate(miss_ego, n)
    m["motion_missing_rate"] = _rate(miss_motion, n)
    m["traj_missing_rate"] = _rate(miss_traj, n)
    m["missing_rate"] = _mean([m["reason_missing_rate"], m["plan_missing_rate"], m["implication_missing_rate"]])
    m["n_start_frames"] = n_start
    m["n_stay_frames"] = n_stay
    m["start_frame_confusion"] = {c: start_conf.get(c, 0) for c in SAMPLE_CLASSES + ["none"]}
    m["stay_frame_confusion"] = {c: stay_conf.get(c, 0) for c in SAMPLE_CLASSES + ["none"]}
    m["start_frame_acc"] = _rate(start_ok, n_start)
    m["start_pred_static_rate"] = _rate(start_static, n_start)
    m["start_frame_motion_acc"] = _rate(start_mot_ok, start_mot_tot)
    m["stay_frame_acc"] = _rate(stay_ok, n_stay)
    m["cap_hit_rate"] = _rate(cap_hit, cap_tot)
    m["traj_unclosed_rate"] = _rate(traj_unclosed, n)
    m["unclosed_obj_rate"] = _rate(unclosed_obj, n)
    m["text_repeat_rate"] = _rate(rep, n)
    m["traj_arith_rate"] = _rate(arith, n)
    m["traj_flat_rate"] = _rate(flat, n)
    m["obj_dup_rate"] = _rate(dup, n)
    m["degenerate_rate"] = _rate(degenerate, n)
    return m
