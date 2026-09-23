"""Proxy composites used to rank dev checkpoints.

S1:  0.15*ctx + 0.15*events_f1 + 0.10*presence + 0.20*recall_r1 + 0.20*loc_relaxed
     + 0.10*attr(macro of intention/state/location) + 0.10*rank_acc
S2:  0.35*traj + 0.25*decision + 0.25*understanding + 0.15*chain, with
     traj          = mean exp(-ADE5 / 2 m)                       (``traj_metrics()['traj_score']``)
     decision      = 0.5*start_frame_acc + 0.5*(1 - missing_rate) (``chain_metrics()``)
     understanding = S1 composite of the same run                (``evaluate_understanding()``)
     chain         = 0.5*motion_traj_consistency + 0.5*ego_state_acc (``chain_metrics()``)

Both functions accept one flat dict (the three metric dicts merged) or a dict that nests them under
``'understanding'`` / ``'traj'`` (alias ``'trajectory'``, the key used by
``evaluation.eval_dev.score_records``) / ``'chain'`` (plus eval_dev's ``'traj_extra'`` /
``'chain_extra'`` as last resort). Terms whose value is ``None`` (no support in the evaluated
subset) are skipped and the remaining weights re-normalised; ``None`` is returned when nothing is
available.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["s1_composite", "s2_composite", "s1_parts", "s2_parts", "weighted"]

S1_WEIGHTS: Dict[str, float] = {
    "ctx_macro_acc": 0.15, "events_f1": 0.15, "presence_f1": 0.10, "obj_recall_rank1": 0.20,
    "loc_hit_relaxed": 0.20, "attr_slot_macro_acc": 0.10, "rank_acc": 0.10,
}
S2_WEIGHTS: Dict[str, float] = {"traj": 0.35, "decision": 0.25, "understanding": 0.25, "chain": 0.15}
_SUBDICTS = ("understanding", "traj", "trajectory", "chain", "traj_extra", "chain_extra")


def _get(m: Dict[str, Any], key: str) -> Optional[float]:
    """Look ``key`` up in ``m`` and in its optional nested sub-dicts; non-numeric -> None."""
    v = m.get(key)
    if v is None:
        for sub in _SUBDICTS:
            d = m.get(sub)
            if isinstance(d, dict) and d.get(key) is not None:
                v = d[key]
                break
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def weighted(parts: Sequence[Tuple[Optional[float], float]]) -> Optional[float]:
    """Weighted mean of ``(value, weight)`` pairs, skipping ``None`` values and re-normalising."""
    keep = [(v, w) for v, w in parts if v is not None and w > 0]
    if not keep:
        return None
    return float(sum(v * w for v, w in keep) / sum(w for _, w in keep))


def s1_parts(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The seven S1 terms (value or None) read from an ``evaluate_understanding`` dict."""
    attr = _get(m, "attr_slot_macro_acc")
    if attr is None:      # fall back to the three per-field accuracies
        attr = _mean([_get(m, "attr_intention_acc"), _get(m, "attr_state_acc"), _get(m, "attr_location_acc")])
    return {
        "ctx_macro_acc": _get(m, "ctx_macro_acc"),
        "events_f1": _get(m, "events_f1"),
        "presence_f1": _get(m, "presence_f1"),
        "obj_recall_rank1": _get(m, "obj_recall_rank1"),
        "loc_hit_relaxed": _get(m, "loc_hit_relaxed"),
        "attr_slot_macro_acc": attr,
        "rank_acc": _get(m, "rank_acc"),
    }


def s1_composite(m: Dict[str, Any]) -> Optional[float]:
    """S1 proxy composite. ``m`` = ``evaluate_understanding()`` output (or a superset)."""
    parts = s1_parts(m)
    return weighted([(parts[k], S1_WEIGHTS[k]) for k in S1_WEIGHTS])


def s2_parts(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The four S2 terms: traj / decision / understanding / chain (value or None)."""
    traj = _get(m, "traj_score")
    if traj is None:
        ade5 = _get(m, "ade@5s")
        if ade5 is not None:            # approximation when only the mean ADE is available
            import math
            traj = math.exp(-ade5 / 2.0)
    missing = _get(m, "missing_rate")
    if missing is None:
        missing = _mean([_get(m, "reason_missing_rate"), _get(m, "plan_missing_rate"),
                         _get(m, "implication_missing_rate")])
    decision = weighted([(_get(m, "start_frame_acc"), 0.5),
                         (None if missing is None else 1.0 - missing, 0.5)])
    understanding = _get(m, "s1_composite")
    if understanding is None:
        understanding = s1_composite(m)
    chain = weighted([(_get(m, "motion_traj_consistency"), 0.5), (_get(m, "ego_state_acc"), 0.5)])
    return {"traj": traj, "decision": decision, "understanding": understanding, "chain": chain}


def s2_composite(m: Dict[str, Any]) -> Optional[float]:
    """S2 proxy composite. ``m`` = merged ``evaluate_understanding`` + ``traj_metrics`` +
    ``chain_metrics`` dicts (flat, or nested under 'understanding'/'traj'/'chain')."""
    parts = s2_parts(m)
    return weighted([(parts[k], S2_WEIGHTS[k]) for k in S2_WEIGHTS])
