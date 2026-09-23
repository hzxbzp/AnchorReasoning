#!/usr/bin/env python3
"""Assistant-target builder.

The target is returned as an ordered list of *segments* ``(text, field, weight_mult)`` so the
dataset/collator can assign a per-token loss weight ``W[field] * weight_mult`` and a per-token
field id (for per-field loss logging and the Stage-1 curriculum).  Concatenating the ``text``
parts in order gives the exact assistant string; no whitespace/newlines are inserted between
segments.

Grammar per task (all values are emitted without surrounding whitespace):

``s1``::

    <context>weather=W;daytime=D;visibility=V;scenario=S;road=R</context>
    <events>E1,E2|none</events><has_objects>yes|no</has_objects>
    [<obj type=T><point>x,y</point><rank>k</rank>[<location>..</location>][<intention>..</intention>]
       [<state>..</state>][<content>..</content>]</obj> x N  <n_objects>N</n_objects>]

``s2``::

    <ego_state>lon=.. | lat=..</ego_state> + s1 (each <obj> additionally ends with <implication>..</implication>)
    + <reason>..</reason><final_plan>..</final_plan><motion>lon=.. | lat=..</motion>
    + <traj>[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]</traj>

``attrqa``::

    a single <obj ...>..</obj> block (point, rank, attributes; no implication)

Weight multipliers (default 1.0):
  * ``intention``/``state`` value segments: ``class_balance_factor(value, freq_table, field)``;
  * each of the 5 ``<traj>`` waypoint segments: ``waypoint_weights(points5, frame)[i] / 3.0``
    (so the effective weight ``W['traj'] * mult`` equals ``3.0 * clip(1 + d_i/2, 1, 3)``);
  * ``reason`` on a plan-repaired frame: ``overlay['reason_weight'] / 1.5``
    (effective weight = ``overlay['reason_weight']``, i.e. 0.5 by default).

Object ordering / rank: objects are sorted by ``attributes.impact_rank`` ascending (missing rank
-> 99, stable w.r.t. annotation order).  ``<rank>`` is the raw ``impact_rank`` when present,
otherwise the object's 1-based position in the sorted list.

Points: ``pano['annotations'][i]['_point']`` (already encoded by the dataset through the adapter's
point codec) is used verbatim.  If it is absent and the annotation carries a ``bbox``, the bbox
center is encoded with ``point_codec.encode(cx, cy, pano.get('_scale', (1, 1)))`` as a fallback;
if neither is possible the ``<point>`` tags are omitted.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from training.core.normalize import CONTEXT_KEYS, fields_for_type, norm_context, norm_type
from training.core.labels import class_balance_factor, ego_state_label, motion_label, traj_points_1hz
from training.core.traj_codec import encode_traj_text, waypoint_weights

try:  # optional helper of the codec (per-waypoint texts, ", " prefixed); regex fallback below if absent
    from training.core.traj_codec import encode_traj_segments
except ImportError:  # pragma: no cover - depends on the codec implementation
    encode_traj_segments = None

# ---- field vocabulary (order fixed; new fields are APPENDED so ids stay stable) ----
FIELDS: List[str] = [
    "struct", "context", "events", "presence", "type", "location", "intention", "state", "content",
    "point", "implication", "reason", "final_plan", "traj", "count", "obj_open", "ego_state", "rank",
    "motion",
]
FIELD_ID: Dict[str, int] = {f: i for i, f in enumerate(FIELDS)}

# ---- per-field loss weights ----
W: Dict[str, float] = {
    "struct": 0.2,
    "ego_state": 1.0,
    "context": 1.0,
    "events": 1.0,
    "presence": 1.0,
    "count": 2.0,
    "obj_open": 3.0,
    "type": 2.0,
    "location": 2.0,
    "content": 2.0,
    "intention": 1.5,
    "state": 1.5,
    "rank": 1.5,
    "point": 3.0,
    "implication": 1.5,
    "reason": 1.5,
    "final_plan": 1.5,
    "motion": 2.0,
    "traj": 3.0,
}

TASKS: Tuple[str, ...] = ("s1", "s2", "attrqa")
# attribute fields whose value segment is multiplied by the class-balance factor
BALANCED_FIELDS: Tuple[str, ...] = ("intention", "state")
# base weights the multipliers above are expressed against (independent of config overrides of W)
TRAJ_BASE_WEIGHT = 3.0
REASON_BASE_WEIGHT = 1.5
MISSING_RANK = 99
DEFAULT_INTENT = "GO_STRAIGHT"

Segment = Tuple[str, str, float]
_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _attr_val(at: dict, field: str) -> Optional[str]:
    """Attribute value as a string; list-valued attributes take the first element; empty -> None."""
    v = at.get(field)
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def intent_of(frame: dict) -> str:
    """Intent used for label derivation: ``intent_corrected`` -> ``intent`` -> ``GO_STRAIGHT``."""
    return frame.get("intent_corrected") or frame.get("intent") or DEFAULT_INTENT


def _rank_value(ann: dict) -> Optional[int]:
    """Raw ``impact_rank`` as an int, or None when missing / non-numeric."""
    r = (ann.get("attributes") or {}).get("impact_rank")
    if isinstance(r, bool):
        return None
    if isinstance(r, (int, float)):
        return int(round(r))
    if isinstance(r, str):
        try:
            return int(round(float(r)))
        except ValueError:
            return None
    return None


def rank_objects(anns: Sequence[dict]) -> List[Tuple[dict, int]]:
    """Sort annotations by impact_rank (missing -> 99, stable) and attach the rank to write.

    Returns ``[(annotation, rank_text_int)]`` where the rank is the raw ``impact_rank`` when present
    and otherwise the object's 1-based position in the sorted list.
    """
    keyed = [(MISSING_RANK if _rank_value(a) is None else _rank_value(a), i, a) for i, a in enumerate(anns)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    out: List[Tuple[dict, int]] = []
    for pos, (_, _, a) in enumerate(keyed, start=1):
        rv = _rank_value(a)
        out.append((a, rv if rv is not None else pos))
    return out


def _point_of(ann: dict, pano: dict, point_codec: Any) -> Optional[Tuple[int, int]]:
    """Encoded integer point for an annotation: ``_point`` first, bbox-center via codec as fallback."""
    pt = ann.get("_point")
    if pt is not None:
        return int(round(float(pt[0]))), int(round(float(pt[1])))
    bb = ann.get("bbox")
    if bb and point_codec is not None and hasattr(point_codec, "encode"):
        try:
            cx = float(bb["x"]) + float(bb["w"]) / 2.0
            cy = float(bb["y"]) + float(bb["h"]) / 2.0
        except (KeyError, TypeError, ValueError):
            return None
        scale = pano.get("_scale") or (1.0, 1.0)
        x, y = point_codec.encode(cx, cy, scale)
        return int(round(float(x))), int(round(float(y)))
    return None


def _balance(value: str, field: str, freq_table: Optional[dict]) -> float:
    """Class-balance multiplier for an attribute value segment (1.0 unless intention/state + table)."""
    if freq_table is None or field not in BALANCED_FIELDS:
        return 1.0
    return float(class_balance_factor(value, freq_table, field))


def _object_block(ann: dict, rank: int, *, pano: dict, point_codec: Any, freq_table: Optional[dict],
                  with_implication: bool) -> List[Segment]:
    """Segments of one ``<obj type=T>...</obj>`` block (point -> rank -> attributes [-> implication])."""
    at = ann.get("attributes") or {}
    ty = norm_type(at.get("type", "Other"))
    seg: List[Segment] = [("<obj type=", "obj_open", 1.0), (ty, "type", 1.0), (">", "struct", 1.0)]
    pt = _point_of(ann, pano, point_codec)
    if pt is not None:
        seg += [("<point>", "struct", 1.0), (f"{pt[0]},{pt[1]}", "point", 1.0), ("</point>", "struct", 1.0)]
    seg += [("<rank>", "struct", 1.0), (str(int(rank)), "rank", 1.0), ("</rank>", "struct", 1.0)]
    for f in fields_for_type(ty):
        v = _attr_val(at, f)
        if v:
            seg += [(f"<{f}>", "struct", 1.0), (v, f, _balance(v, f, freq_table)), (f"</{f}>", "struct", 1.0)]
    if with_implication:
        impl = ann.get("driving_implication")
        if impl and str(impl).strip():
            seg += [("<implication>", "struct", 1.0), (str(impl).strip(), "implication", 1.0),
                    ("</implication>", "struct", 1.0)]
    seg.append(("</obj>", "struct", 1.0))
    return seg


def waypoint_texts(pts: Sequence[Tuple[float, float]]) -> List[str]:
    """Per-waypoint texts whose concatenation equals ``encode_traj_text(pts)``.

    Segment i is ``"[x_i, y_i]"`` for i = 0 and ``", [x_i, y_i]"`` afterwards (the ``", "`` separator is
    attached to the waypoint it introduces, same convention as ``traj_codec.encode_traj_segments``).
    The codec's own segmenter is used when it exists; otherwise the codec string is re-split on its
    bracket groups.  Raises ``ValueError`` if the codec output cannot be segmented consistently.
    """
    full = encode_traj_text(pts)
    if encode_traj_segments is not None:
        parts = [str(p) for p in encode_traj_segments(pts)]
        if len(parts) == len(pts) and "".join(parts) == full:
            return parts
    groups = _BRACKET_RE.findall(full)
    if len(groups) == len(pts) and ", ".join(groups) == full:
        return [g if i == 0 else ", " + g for i, g in enumerate(groups)]
    raise ValueError(f"cannot split codec trajectory text into {len(pts)} waypoint segments: {full!r}")


def traj_segments(frame: dict) -> List[Segment]:
    """``<traj>`` block split per waypoint (see ``waypoint_texts``), each with its own multiplier.

    ``weight_mult[i] = waypoint_weights(points5, frame)[i] / 3.0`` so that the effective weight
    ``W['traj'] * mult`` is ``3.0 * clip(1 + d_i / 2, 1, 3)``.  The tags are ``struct``.
    """
    pts = [tuple(p) for p in traj_points_1hz(frame)]
    parts = waypoint_texts(pts)
    ww = [float(w) for w in waypoint_weights(pts, frame)]
    if len(ww) != len(parts):
        raise ValueError(f"waypoint_weights returned {len(ww)} weights for {len(parts)} waypoints")
    seg: List[Segment] = [("<traj>", "struct", 1.0)]
    for text, w in zip(parts, ww):
        seg.append((text, "traj", w / TRAJ_BASE_WEIGHT))
    seg.append(("</traj>", "struct", 1.0))
    return seg


def _resolve_attrqa_obj(anns: Sequence[dict], attrqa_obj: Union[int, dict, None]) -> dict:
    """Map ``attrqa_obj`` (index into ``pano['annotations']`` or the annotation dict itself) to the dict."""
    if attrqa_obj is None:
        raise ValueError("task='attrqa' requires attrqa_obj (annotation index or annotation dict)")
    if isinstance(attrqa_obj, bool):
        raise ValueError("attrqa_obj must be an int index or an annotation dict")
    if isinstance(attrqa_obj, int):
        if not 0 <= attrqa_obj < len(anns):
            raise ValueError(f"attrqa_obj index {attrqa_obj} out of range for {len(anns)} annotations")
        return anns[attrqa_obj]
    if isinstance(attrqa_obj, dict):
        for a in anns:
            if a is attrqa_obj:
                return a
        for a in anns:
            if a == attrqa_obj:
                return a
        return attrqa_obj  # not part of pano -> still build its block (rank from its own impact_rank)
    raise ValueError("attrqa_obj must be an int index or an annotation dict")


# ----------------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------------
def build_target(frame: dict, pano: dict, task: str, *, point_codec: Any, freq_table: Optional[dict] = None,
                 overlay: Optional[dict] = None, attrqa_obj: Union[int, dict, None] = None) -> List[Segment]:
    """Build the assistant target as ``[(text, field, weight_mult)]`` for one frame.

    Args:
        frame: ``frame.json`` dict (past_states / future_states / intent / ego_behavior ...).
        pano: ``panorama_geo.json`` dict; ``annotations[i]['_point']`` is expected to be pre-encoded
            by the dataset (integer target coordinates).
        task: ``'s1'`` | ``'s2'`` | ``'attrqa'``.
        point_codec: adapter ``PointCodec`` (used only as a fallback when ``_point`` is missing).
        freq_table: ``{'intention': {value: count}, 'state': {...}}`` for class balancing (None -> 1.0).
        overlay: plan-repair entry ``{'final_plan': str, 'reason_weight': float}`` or None (S2 only).
        attrqa_obj: for ``'attrqa'``: index into ``pano['annotations']`` or the annotation dict.

    Returns:
        Ordered segment list; ``''.join(text)`` is the exact assistant string.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")

    anns = list(pano.get("annotations") or [])
    ranked = rank_objects(anns)

    if task == "attrqa":
        target = _resolve_attrqa_obj(anns, attrqa_obj)
        for a, rk in ranked:
            if a is target:
                return _object_block(a, rk, pano=pano, point_codec=point_codec, freq_table=freq_table,
                                     with_implication=False)
        rv = _rank_value(target)
        return _object_block(target, rv if rv is not None else len(anns) + 1, pano=pano, point_codec=point_codec,
                             freq_table=freq_table, with_implication=False)

    seg: List[Segment] = []

    def s(text: str, field: str, mult: float = 1.0) -> None:
        seg.append((text, field, float(mult)))

    # ---- S2 prefix: current ego state (from history) ----
    if task == "s2":
        s("<ego_state>", "struct"); s(ego_state_label(frame)["text"], "ego_state"); s("</ego_state>", "struct")

    # ---- context ----
    ctx = norm_context(pano.get("context"))
    s("<context>", "struct")
    s(";".join(f"{k}={ctx[k]}" for k in CONTEXT_KEYS), "context")
    s("</context>", "struct")

    # ---- traffic events ----
    evs: List[str] = []
    for e in (pano.get("traffic_events") or []):
        evs += [str(t) for t in (e.get("types") or []) if t]
    s("<events>", "struct"); s(",".join(evs) if evs else "none", "events"); s("</events>", "struct")

    # ---- presence ----
    s("<has_objects>", "struct"); s("yes" if anns else "no", "presence"); s("</has_objects>", "struct")

    # ---- objects (by impact_rank), then the count as a checksum ----
    if anns:
        for a, rk in ranked:
            seg.extend(_object_block(a, rk, pano=pano, point_codec=point_codec, freq_table=freq_table,
                                     with_implication=(task == "s2")))
        s("<n_objects>", "struct"); s(str(len(anns)), "count"); s("</n_objects>", "struct")

    if task == "s1":
        return seg

    # ---- S2 chain: reason -> final_plan -> motion -> traj ----
    reason = pano.get("reason")
    if reason and str(reason).strip():
        reason_mult = 1.0
        if overlay and overlay.get("reason_weight") is not None:
            reason_mult = float(overlay["reason_weight"]) / REASON_BASE_WEIGHT
        s("<reason>", "struct"); s(str(reason).strip(), "reason", reason_mult); s("</reason>", "struct")

    plan = overlay.get("final_plan") if overlay and overlay.get("final_plan") else pano.get("final_plan")
    if plan and str(plan).strip():
        s("<final_plan>", "struct"); s(str(plan).strip(), "final_plan"); s("</final_plan>", "struct")

    s("<motion>", "struct"); s(motion_label(frame, intent_of(frame))["text"], "motion"); s("</motion>", "struct")

    seg.extend(traj_segments(frame))
    return seg


# ----------------------------------------------------------------------------------------------
# conveniences for consumers of the segment list
# ----------------------------------------------------------------------------------------------
def segments_text(segs: Sequence[Segment]) -> str:
    """Concatenate segment texts into the assistant string."""
    return "".join(t for t, _, _ in segs)


def segment_weights(segs: Sequence[Segment], weights: Optional[Dict[str, float]] = None) -> List[float]:
    """Effective per-segment loss weight ``weights[field] * weight_mult`` (``weights`` defaults to ``W``)."""
    tbl = W if weights is None else {**W, **weights}
    return [float(tbl[f]) * float(m) for _, f, m in segs]


def segment_field_ids(segs: Sequence[Segment]) -> List[int]:
    """Per-segment integer field ids (``FIELD_ID``)."""
    return [FIELD_ID[f] for _, f, _ in segs]


def traj_mask(segs: Sequence[Segment]) -> List[bool]:
    """Per-segment flag: True for the waypoint segments inside ``<traj>`` (field == 'traj')."""
    return [f == "traj" for _, f, _ in segs]
