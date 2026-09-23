"""Parse a generated Stage-1 / Stage-2 answer back into a structured dict.

Tolerant regex parsing of the target grammar::

    <ego_state>lon=stopped 3.2s | lat=LANE_KEEPING</ego_state>
    <context>weather=W;daytime=D;visibility=V;scenario=S;road=R</context>
    <events>E1,E2|none</events>
    <has_objects>yes|no</has_objects>
    <obj type=T><point>x,y</point><rank>k</rank><location>..</location><intention>..</intention>
        <state>..</state><content>..</content><implication>..</implication></obj> x N
    <n_objects>N</n_objects>
    <reason>..</reason><final_plan>..</final_plan>
    <motion>lon=accelerate | lat=straight</motion>
    <traj>[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]</traj>

Segments an answer does not contain simply come back as ``None`` (Stage-1 answers, for example,
stop after the understanding part and have no ``motion`` or ``traj``).

The dict returned by :func:`parse_output` is the *only* representation the metric modules
(``training.core.metrics``) consume; every key is always present.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "parse_output", "parse_traj", "parse_kv", "parse_ego_lon", "ego_lon_class",
    "norm_motion_lon", "norm_motion_lat", "OBJ_FIELDS", "EGO_LON_CLASSES", "MOTION_LON", "MOTION_LAT",
]

OBJ_FIELDS = ("location", "intention", "state", "content", "implication")
EGO_LON_CLASSES = ("stopped", "accelerating", "decelerating", "cruising")
MOTION_LON = ("accelerate", "decelerate", "keep", "stop")
MOTION_LAT = ("straight", "left_turn", "right_turn")
TRAJ_POINTS = 5


def _tag(name: str) -> "re.Pattern[str]":
    return re.compile(rf"<{name}>(.*?)</{name}>", re.S)


_CTX = _tag("context")
_EVT = _tag("events")
_EGO = _tag("ego_state")
_MOTION = _tag("motion")
_REASON = _tag("reason")
_PLAN = _tag("final_plan")
_HAS = re.compile(r"<has_objects>\s*(yes|no)", re.I)
_NOBJ = re.compile(r"<n_objects>\s*(\d+)")
# tolerate optional quotes around the type and whitespace before '>'
_OBJ = re.compile(r"<obj\s+type=[\"']?(.*?)[\"']?\s*>(.*?)</obj>", re.S)
_OBJ_OPEN = re.compile(r"<obj\s+type=")
_PT = re.compile(r"<point>\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*</point>")
_RANK = re.compile(r"<rank>\s*(\d+)\s*</rank>")
# <traj> ... </traj>; an unclosed <traj> (generation cap) is still parsed up to the end of text
_TRAJ = re.compile(r"<traj>(.*?)(?:</traj>|$)", re.S)
_BRACKET_PAIR = re.compile(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]")
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _field(body: str, name: str) -> Optional[str]:
    m = re.search(rf"<{name}>(.*?)</{name}>", body, re.S)
    return m.group(1).strip() if m else None


def parse_kv(s: Optional[str], sep: str = "|") -> Dict[str, str]:
    """``"lon=stopped 3.2s | lat=LANE_KEEPING"`` -> ``{'lon': 'stopped 3.2s', 'lat': 'LANE_KEEPING'}``.

    Keys are lower-cased and stripped; entries without ``=`` are ignored.
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    for kv in s.split(sep):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def parse_traj(text: str) -> Optional[List[Tuple[float, float]]]:
    """Parse the ``<traj>`` segment into exactly 5 ``(x, y)`` tuples.

    Returns ``None`` when the segment is absent, has fewer than 5 bracketed ``[x, y]`` pairs or
    contains non-numeric pairs; such an answer counts as "no valid trajectory" in the metrics.
    Extra pairs beyond the 5th are ignored.
    """
    m = _TRAJ.search(text or "")
    if not m:
        return None
    pairs = _BRACKET_PAIR.findall(m.group(1))
    if len(pairs) < TRAJ_POINTS:
        return None
    try:
        pts = [(float(a), float(b)) for a, b in pairs[:TRAJ_POINTS]]
    except ValueError:  # the bracket regex already guarantees numeric pairs
        return None
    return pts


def ego_lon_class(lon_text: Optional[str]) -> Optional[str]:
    """Longitudinal class word of an ego_state ``lon=`` value (``stopped|accelerating|decelerating|cruising``)."""
    if not lon_text:
        return None
    s = str(lon_text).strip().lower()
    for cls in EGO_LON_CLASSES:
        if s.startswith(cls[:4]):        # 'stop', 'acce', 'dece', 'crui' (tolerates 'stop 3.2s')
            return cls
    return None


def parse_ego_lon(lon_text: Optional[str]) -> Dict[str, Any]:
    """Decompose an ego_state ``lon`` value.

    Returns ``{'cls': str|None, 'T': float|None, 'plus': bool, 'v_from': float|None, 'v_to': float|None,
    'v': float|None}`` where ``T`` is the stopped time (``'4.0s+'`` -> T=4.0, plus=True), ``v_from/v_to``
    the speeds of ``accelerating v1->v2 m/s`` and ``v`` the cruising speed.
    """
    out: Dict[str, Any] = {"cls": ego_lon_class(lon_text), "T": None, "plus": False,
                           "v_from": None, "v_to": None, "v": None}
    if not lon_text:
        return out
    nums = [float(x) for x in _NUM.findall(str(lon_text))]
    cls = out["cls"]
    if cls == "stopped":
        out["T"] = nums[0] if nums else None
        out["plus"] = "+" in str(lon_text)
    elif cls in ("accelerating", "decelerating"):
        if len(nums) >= 2:
            out["v_from"], out["v_to"] = nums[0], nums[1]
        elif nums:
            out["v_to"] = nums[0]
    elif cls == "cruising":
        out["v"] = nums[0] if nums else None
    return out


def norm_motion_lon(v: Optional[str]) -> Optional[str]:
    """Map a free-form ``motion.lon`` value onto the closed set ``accelerate|decelerate|keep|stop`` (else None)."""
    if not v:
        return None
    s = str(v).strip().lower()
    if s.startswith("acc"):
        return "accelerate"
    if s.startswith("dec"):
        return "decelerate"
    if s.startswith("keep") or s.startswith("maintain") or s.startswith("cruis"):
        return "keep"
    if s.startswith("stop"):
        return "stop"
    return None


def norm_motion_lat(v: Optional[str]) -> Optional[str]:
    """Map a free-form ``motion.lat`` value onto ``straight|left_turn|right_turn`` (else None)."""
    if not v:
        return None
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    if s.startswith("straight"):
        return "straight"
    if s.startswith("left"):
        return "left_turn"
    if s.startswith("right"):
        return "right_turn"
    return None


def parse_output(text: Optional[str]) -> Dict[str, Any]:
    """Parse one generated answer into a dict.

    Keys (always present)::

        ego_state    {'lon': str|None, 'lat': str|None, 'lon_class': str|None} | None (segment absent)
        context      {dim: value}
        events       [str]          ('none' -> [])
        has_objects  bool|None
        n_objects    int|None       predicted <n_objects>
        objects      [{'type', 'point': (x,y)|None, 'rank': int|None, 'location', 'intention', 'state',
                       'content', 'implication'}]   (only CLOSED <obj ...>...</obj> blocks, in order)
        n_unclosed_obj int          '<obj type=' openings without a matching </obj> (truncation signal)
        reason       str|None
        final_plan   str|None
        motion       {'lon': str|None, 'lat': str|None} | None   (values normalised to the closed sets;
                                                                  unknown words -> None)
        motion_raw   {'lon': str, 'lat': str} | None            (verbatim values)
        traj         [(x,y)]*5 | None
        traj_n_points int           number of bracketed pairs found inside <traj> (0 if absent)
        traj_closed  bool           whether </traj> was emitted

    ``point`` is returned in the model's own coordinate system; the metric code maps it back to
    panorama pixels through the adapter's ``PointCodec.decode``.
    """
    text = text or ""
    out: Dict[str, Any] = {
        "ego_state": None, "context": {}, "events": [], "has_objects": None,
        "n_objects": None, "objects": [], "n_unclosed_obj": 0, "reason": None, "final_plan": None,
        "motion": None, "motion_raw": None, "traj": None, "traj_n_points": 0, "traj_closed": False,
    }
    m = _EGO.search(text)
    if m:
        kv = parse_kv(m.group(1))
        lon = kv.get("lon")
        out["ego_state"] = {"lon": lon, "lat": kv.get("lat"), "lon_class": ego_lon_class(lon)}
    m = _CTX.search(text)
    if m:
        for kv_ in m.group(1).split(";"):
            if "=" in kv_:
                k, v = kv_.split("=", 1)
                out["context"][k.strip()] = v.strip()
    m = _EVT.search(text)
    if m:
        ev = m.group(1).strip()
        out["events"] = [] if ev.lower() in ("", "none") else [e.strip() for e in ev.split(",") if e.strip()]
    m = _HAS.search(text)
    if m:
        out["has_objects"] = (m.group(1).lower() == "yes")
    m = _NOBJ.search(text)
    out["n_objects"] = int(m.group(1)) if m else None
    n_open = 0
    for ty, body in _OBJ.findall(text):
        n_open += 1
        o: Dict[str, Any] = {"type": ty.strip()}
        for f in OBJ_FIELDS:
            o[f] = _field(body, f)
        pm = _PT.search(body)
        o["point"] = (float(pm.group(1)), float(pm.group(2))) if pm else None
        rm = _RANK.search(body)
        o["rank"] = int(rm.group(1)) if rm else None
        out["objects"].append(o)
    out["n_unclosed_obj"] = max(0, len(_OBJ_OPEN.findall(text)) - n_open)
    m = _REASON.search(text)
    out["reason"] = m.group(1).strip() if m else None
    m = _PLAN.search(text)
    out["final_plan"] = m.group(1).strip() if m else None
    m = _MOTION.search(text)
    if m:
        kv = parse_kv(m.group(1))
        out["motion_raw"] = {"lon": kv.get("lon"), "lat": kv.get("lat")}
        out["motion"] = {"lon": norm_motion_lon(kv.get("lon")), "lat": norm_motion_lat(kv.get("lat"))}
    tm = _TRAJ.search(text)
    if tm:
        out["traj_n_points"] = len(_BRACKET_PAIR.findall(tm.group(1)))
        out["traj_closed"] = "</traj>" in text
    out["traj"] = parse_traj(text)
    return out
