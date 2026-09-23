"""Deterministic label derivation for the two-stage SFT.

Every function here is pure (no IO, no randomness) and works on the raw ``frame.json`` /
``panorama_geo.json`` dicts written by the dataset decoding step:

* ``frame['past_states']``   : ``pos_x/pos_y/vel_x/vel_y`` lists, 16 steps @ 4 Hz, ego frame
                               (last point is the origin, heading is +x).
* ``frame['future_states']`` : ``pos_x/pos_y`` lists, 20 steps @ 4 Hz (t = 0.25 .. 5.0 s).
* ``frame['ego_behavior']``  : ``{'longitudinal': .., 'lateral': ..}`` (only ``lateral`` is read,
                               and it may be a placeholder when the source records carry no
                               ego-behaviour label; a missing value becomes ``'n/a'``).
* ``pano['annotations']``    : ``[{'attributes': {'intention': [..]|str, 'state': ..}, ..}]``.

Main label API: ``history_text, ego_state_label, future_speeds, sample_class, motion_label,
lat_class, plan_to_motion, plan_consistent, traj_points_1hz, minority_attr_frame,
class_balance_factor``.  Additional helpers (``v0_of, frame_intent, future_xy, lon_class,
heading_profile, future_v_end, future_start_time, plan_lon_compatible, attr_first,
minority_attr_object, sorted_with_ranks``) are exported for the dataset / target-builder / cache /
metrics modules.

Design notes:

* ``plan_consistent(plan_text, motion, v0, *, strict=False, frame=None, v_end=None, start_t=None)``
  judges a plan against the motion trend with the *compatibility matrix* implemented in
  ``plan_lon_compatible`` ("not contradictory" == consistent).  ``strict=True`` falls back to plain
  class equality (``class(plan) == motion.lon``).  Class equality is too aggressive in practice
  because of the semantic gap between the plan wording and the realised motion ("decelerate"
  meaning caution while the car holds speed, "stop and wait" followed by a start 2-3 s later).

* ``plan_consistent`` (tolerant, default): the speed-based relaxations of the matrix need ``v_end``
  (mean speed of the last future second) and ``start_t`` (first future time with v >= 0.5 m/s).
  They are taken from ``frame['future_states']`` when a frame is given, else from the explicit
  keyword arguments; when neither is available only the class-set rows of the matrix apply (a plan
  ``decelerate`` still tolerates motion ``stop``, but ``keep`` vs ``accelerate`` is a contradiction
  because the |v_end - v0| band cannot be checked).  ``motion`` may be the ``motion_label`` dict or
  a bare ``motion.lon`` string (lateral not judged).

* ``motion_label`` evaluates the longitudinal rules in a fixed order (``stop`` first), so a frame
  that starts and stops again within 5 s is ``stop`` while its ``sample_class`` is ``start``.
* ``lat_class`` truncates the future path at the first heading reversal (>120 deg between
  consecutive segments).  Reversals only occur when the ego rolls backwards; without the guard a
  "creep forward then roll back" frame would be labelled a 180 deg turn.
* ``plan_to_motion`` maps several longitudinal words to the last action *result*, in two tiers:
  explicit speed-change / end-state verbs (stop, decelerate, accelerate, cruise, ...) always win
  over continuation verbs (proceed, creep, go, follow, ...), and among the explicit verbs the last
  one wins.  This keeps ``"decelerate and stop" -> stop`` and ``"wait, then proceed" -> stop``
  while avoiding ``"accelerate and proceed through the intersection" -> keep``.  Masking happens in
  two tiers: object noun phrases (``"stop line"``, ``"parked car"``, ``"the vehicle turning
  left"``, ``"wait for the car to pass"``, ``"right turn lane"``) are removed first and the plan's
  own ``lat`` is read from what is left, then clause masks (purpose infinitives ``"to follow the
  lead vehicle"``, ``"keep lane"``, ``"keep right"``, ``"keep a safe distance"``, ``"without
  stopping"``) are removed for the longitudinal match.  ``"keep lane"`` / ``"keep right"`` on their
  own are lateral-only (``unspecified``, notes ``keep_lane`` / ``keep_side``), never longitudinal
  ``keep``.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.core.paths import DT, FUT_STEPS, HIST_STEPS

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
MOTION_LON = ('accelerate', 'decelerate', 'keep', 'stop')
MOTION_LAT = ('straight', 'left_turn', 'right_turn')
SAMPLE_CLASSES = ('stay', 'start', 'stop', 'decel', 'accel', 'keep')
PLAN_LON_CLASSES = MOTION_LON + ('unspecified', 'unmatched')
EGO_LON_CLASSES = ('stopped', 'accelerating', 'decelerating', 'cruising')
TURN_INTENTS = frozenset({'GO_LEFT', 'GO_RIGHT'})
MAJORITY_ATTR = frozenset({'cruising', 'stopping', 'parking', 'crossing', ''})

STOP_SPEED = 0.5          # m/s: below this the ego counts as stopped
START_VMAX = 1.0          # m/s: max future speed that turns a stopped ego into a "start"
EGO_DV = 0.6              # m/s: |v0 - v(-1 s)| beyond which ego_state is accelerating/decelerating
STOPPED_CAP_S = 4.0       # "stopped 4.0s+"
TURN_DEG = 25.0           # |heading change| needed for a turn
CURV_RESID_DEG = 5.0      # constant-curvature fit residual that separates a turn from a road curve
MIN_STEP_M = 0.3          # future points closer than this to the previous kept point are skipped
REVERSAL_DEG = 120.0      # consecutive-segment heading jump treated as a backing reversal
TRAJ_1HZ_IDX = (3, 7, 11, 15, 19)   # 0-based indices of the t = 1..5 s waypoints
BALANCE_CLIP = (0.5, 3.0)

# plan/motion compatibility matrix used by the tolerant plan_consistent
TOL_KEEP_DV_ABS = 3.0     # plan keep vs motion accelerate/decelerate: |v_end - v0| <= max(3.0, 0.5*v0) (moving ego)
TOL_KEEP_DV_REL = 0.5
TOL_ACC_KEEP_DV = 0.5     # plan accelerate vs motion keep: v_end >= v0 - 0.5
TOL_STOP_START_S = 2.0    # plan stop vs motion accelerate: stopped ego (v0 < 0.5) that starts at start_t >= 2.0 s
TOL_STOP_KEEP_VEND = 1.5  # plan stop vs motion keep: v_end < 1.5 (creeping / crawling)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _fmt1(v: float) -> str:
    """Format with one decimal, mapping -0.0 to 0.0 (labels never emit '-0.0')."""
    r = round(float(v), 1)
    if r == 0:
        r = 0.0
    return f'{r:.1f}'


def _round1(v: float) -> float:
    r = round(float(v), 1)
    return 0.0 if r == 0 else r


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _past_speeds(frame: Dict[str, Any]) -> List[float]:
    """|vel| per past step (last HIST_STEPS).  Falls back to position differences if vel is absent."""
    ps = frame.get('past_states') or {}
    vx, vy = ps.get('vel_x') or [], ps.get('vel_y') or []
    n = min(len(vx), len(vy))
    if n > 0:
        return [math.hypot(vx[i], vy[i]) for i in range(max(0, n - HIST_STEPS), n)]
    px, py = ps.get('pos_x') or [], ps.get('pos_y') or []
    n = min(len(px), len(py))
    if n < 2:
        return [0.0] * n
    sp = [math.hypot(px[i] - px[i - 1], py[i] - py[i - 1]) / DT for i in range(1, n)]
    sp = [sp[0]] + sp
    return sp[max(0, len(sp) - HIST_STEPS):]


def v0_of(frame: Dict[str, Any]) -> float:
    """Current speed |vel[-1]| in m/s (0.0 when the frame has no past states)."""
    sp = _past_speeds(frame)
    return sp[-1] if sp else 0.0


def frame_intent(frame: Dict[str, Any]) -> str:
    """``intent_corrected`` with fallback to ``intent`` (and finally GO_STRAIGHT)."""
    return frame.get('intent_corrected') or frame.get('intent') or 'GO_STRAIGHT'


def future_xy(frame: Dict[str, Any], n: int = FUT_STEPS) -> List[Tuple[float, float]]:
    """Future (x, y) points (ego frame) as a list of at most ``n`` tuples."""
    fs = frame.get('future_states') or {}
    px, py = fs.get('pos_x') or [], fs.get('pos_y') or []
    m = min(len(px), len(py), n)
    return [(float(px[i]), float(py[i])) for i in range(m)]


# --------------------------------------------------------------------------------------
# 1. history text
# --------------------------------------------------------------------------------------
def history_text(frame: Dict[str, Any]) -> str:
    """History string ``"[x, y], [x, y], ..., [0.0, 0.0]"`` — the last 16 past positions, 1 decimal.

    The last past position is the origin by construction, so the string always ends with
    ``[0.0, 0.0]`` (``-0.0`` is normalised to ``0.0``).
    """
    ps = frame.get('past_states') or {}
    px, py = ps.get('pos_x') or [], ps.get('pos_y') or []
    n = min(len(px), len(py))
    lo = max(0, n - HIST_STEPS)
    return ', '.join(f'[{_fmt1(px[i])}, {_fmt1(py[i])}]' for i in range(lo, n))


# --------------------------------------------------------------------------------------
# 2. ego state (current behaviour, from past states)
# --------------------------------------------------------------------------------------
def ego_state_label(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Ego-state label derived from the past states.

    lon: ``v0=|vel[-1]|``, ``v1=|vel[-5]|`` (1 s ago; ``vel[0]`` if fewer than 5 steps),
    ``dv=v0-v1``.  ``v0<0.5`` -> ``stopped Ts`` (T = trailing steps with |vel|<0.5 x 0.25 s,
    ``4.0s+`` when T >= 4.0); ``dv>+0.6`` -> ``accelerating v1->v0 m/s``; ``dv<-0.6`` ->
    ``decelerating v1->v0 m/s``; else ``cruising v0 m/s``.  Speeds use one decimal.

    lat: ``frame['ego_behavior']['lateral']`` verbatim, ``'n/a'`` when missing.

    Returns ``{'lon', 'lat', 'lon_class', 'text', 'v0', 'v1'}`` (``v0``/``v1`` are the raw
    floats used, exported for the motion-meta cache).
    """
    sp = _past_speeds(frame)
    if sp:
        v0 = sp[-1]
        v1 = sp[-5] if len(sp) >= 5 else sp[0]
    else:
        v0 = v1 = 0.0
    dv = v0 - v1
    if v0 < STOP_SPEED:
        n_stopped = 0
        for s in reversed(sp):
            if s < STOP_SPEED:
                n_stopped += 1
            else:
                break
        t_stopped = n_stopped * DT
        lon = f'stopped {STOPPED_CAP_S:.1f}s+' if t_stopped >= STOPPED_CAP_S else f'stopped {t_stopped:.1f}s'
        lon_class = 'stopped'
    elif dv > EGO_DV:
        lon, lon_class = f'accelerating {v1:.1f}->{v0:.1f} m/s', 'accelerating'
    elif dv < -EGO_DV:
        lon, lon_class = f'decelerating {v1:.1f}->{v0:.1f} m/s', 'decelerating'
    else:
        lon, lon_class = f'cruising {v0:.1f} m/s', 'cruising'
    eb = frame.get('ego_behavior') or {}
    lat = eb.get('lateral') if isinstance(eb, dict) else None
    lat = str(lat).strip() if lat else 'n/a'
    return {'lon': lon, 'lat': lat, 'lon_class': lon_class, 'text': f'lon={lon} | lat={lat}',
            'v0': float(v0), 'v1': float(v1)}


# --------------------------------------------------------------------------------------
# 3. future speeds, sample class, motion trend
# --------------------------------------------------------------------------------------
def future_speeds(frame: Dict[str, Any]) -> List[float]:
    """20 future speeds (m/s) = consecutive displacement / 0.25 s; the first one is relative to the origin."""
    prev = (0.0, 0.0)
    out: List[float] = []
    for p in future_xy(frame):
        out.append(math.hypot(p[0] - prev[0], p[1] - prev[1]) / DT)
        prev = p
    return out


def v_end_from_speeds(speeds: Sequence[float]) -> Optional[float]:
    """``v_end`` = mean of the last 1 s (last 4 steps @ 4 Hz) of a per-step speed list; ``None`` if empty."""
    if not speeds:
        return None
    tail = speeds[-4:]
    return float(sum(tail) / len(tail))


def start_time_from_speeds(speeds: Sequence[float], dt: float = DT) -> Optional[float]:
    """``start_t`` = time (s) of the first step with speed >= 0.5 m/s (step k -> (k+1)*dt).

    ``None`` when the list is empty or the ego never reaches 0.5 m/s within the horizon.
    """
    for k, s in enumerate(speeds):
        if s >= STOP_SPEED:
            return (k + 1) * dt
    return None


def future_v_end(frame: Dict[str, Any]) -> Optional[float]:
    """``v_end`` of the frame's future (mean speed over the last 1 s); ``None`` without future states."""
    return v_end_from_speeds(future_speeds(frame))


def future_start_time(frame: Dict[str, Any]) -> Optional[float]:
    """``start_t`` of the frame's future: first t with v >= 0.5 m/s; ``None`` if never / no future."""
    return start_time_from_speeds(future_speeds(frame))


def _speed_summary(frame: Dict[str, Any]) -> Tuple[float, float, float, List[float]]:
    """(v0, v_end, v_max, speeds); v_end = mean speed of the last 1 s (last 4 steps)."""
    sp = future_speeds(frame)
    v0 = v0_of(frame)
    if not sp:
        return v0, 0.0, 0.0, sp
    return v0, v_end_from_speeds(sp), max(sp), sp


def sample_class(frame: Dict[str, Any]) -> str:
    """Fine-grained motion class used for sampling / statistics.

    ``stay``  : v0<0.5 and max(v)<0.5
    ``start`` : v0<0.5 and max(v)>=1.0
    ``stop``  : v0>=0.5 and v_end<0.5
    ``decel`` : v_end < v0 - max(1, 0.2*v0)
    ``accel`` : v_end > v0 + max(1, 0.2*v0)
    ``keep``  : otherwise (includes a stopped ego that only creeps: 0.5<=max(v)<1.0)
    """
    v0, v_end, v_max, sp = _speed_summary(frame)
    if not sp:
        return 'stay' if v0 < STOP_SPEED else 'keep'
    if v0 < STOP_SPEED and v_max < STOP_SPEED:
        return 'stay'
    if v0 < STOP_SPEED and v_max >= START_VMAX:
        return 'start'
    if v0 >= STOP_SPEED and v_end < STOP_SPEED:
        return 'stop'
    thr = max(1.0, 0.2 * v0)
    if v_end < v0 - thr:
        return 'decel'
    if v_end > v0 + thr:
        return 'accel'
    return 'keep'


def lon_class(frame: Dict[str, Any]) -> str:
    """Longitudinal trend (closed set ``accelerate|decelerate|keep|stop``); the rules are tried in
    the order listed below.

    ``stop``       : v_end < 0.5 (stays stopped, brakes to a stop, or starts and stops again)
    ``accelerate`` : (v0<0.5 and max(v)>=1.0) or v_end - v0 > max(1 m/s, 0.2*v0)
    ``decelerate`` : v0 - v_end > max(1 m/s, 0.2*v0) (v_end >= 0.5 is implied by the order)
    ``keep``       : otherwise (constant speed, creeping, ...)
    """
    v0, v_end, v_max, sp = _speed_summary(frame)
    if not sp:
        return 'stop' if v0 < STOP_SPEED else 'keep'
    if v_end < STOP_SPEED:
        return 'stop'
    thr = max(1.0, 0.2 * v0)
    if (v0 < STOP_SPEED and v_max >= START_VMAX) or (v_end - v0 > thr):
        return 'accelerate'
    if v0 - v_end > thr and v_end >= STOP_SPEED:
        return 'decelerate'
    return 'keep'


def heading_profile(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """(theta_deg, resid_deg) of a path that starts at ``pts[0]`` (public; ``_heading_profile`` is an alias).

    Points closer than MIN_STEP_M to the previously kept point are skipped; the path is truncated
    at the first backing reversal (consecutive heading jump > REVERSAL_DEG).  theta = heading of
    the last kept segment minus heading of the first; resid = max |residual| (deg) of a
    least-squares constant-curvature fit heading(s) = a + b*s over the kept segments.
    ``metrics.traj.motion_lat_from_xy`` reuses it for the heading-only lateral proxy.
    """
    kept = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= MIN_STEP_M:
            kept.append(p)
    headings: List[float] = []     # unwrapped
    s_mid: List[float] = []
    s = 0.0
    for i in range(1, len(kept)):
        dx, dy = kept[i][0] - kept[i - 1][0], kept[i][1] - kept[i - 1][1]
        h = math.atan2(dy, dx)
        if headings:
            d = _wrap(h - headings[-1])
            if abs(d) > math.radians(REVERSAL_DEG):
                break                       # backing reversal -> ignore the rest of the path
            h = headings[-1] + d
        seg = math.hypot(dx, dy)
        s_mid.append(s + seg / 2.0)
        s += seg
        headings.append(h)
    if len(headings) < 2:
        return 0.0, 0.0
    theta = math.degrees(_wrap(headings[-1] - headings[0]))
    n = len(headings)
    mx, my = sum(s_mid) / n, sum(headings) / n
    sxx = sum((x - mx) ** 2 for x in s_mid)
    b = sum((x - mx) * (y - my) for x, y in zip(s_mid, headings)) / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = max(abs(math.degrees(y - (a + b * x))) for x, y in zip(s_mid, headings))
    return theta, resid


_heading_profile = heading_profile      # private alias of the public name


def lat_class(frame: Dict[str, Any], intent: Optional[str] = None) -> str:
    """Lateral trend ``straight|left_turn|right_turn`` from the future 5 s path.

    theta = heading difference between the first and last future segments (origin prepended,
    points with <0.3 m displacement skipped, path truncated at a backing reversal).
    ``|theta| >= 25 deg`` and (constant-curvature heading residual >= 5 deg or intent in
    GO_LEFT/GO_RIGHT) -> ``left_turn`` (theta > 0, +y is left) / ``right_turn``; else ``straight``.
    A large but smooth heading change with a GO_STRAIGHT intent is a road curve -> ``straight``.
    Lane changes / nudges / pull-outs are not represented (they map to ``straight``).
    """
    intent = intent if intent is not None else frame_intent(frame)
    pts = [(0.0, 0.0)] + future_xy(frame)
    theta, resid = heading_profile(pts)
    if abs(theta) >= TURN_DEG and (resid >= CURV_RESID_DEG or intent in TURN_INTENTS):
        return 'left_turn' if theta > 0 else 'right_turn'
    return 'straight'


def motion_label(frame: Dict[str, Any], intent: Optional[str] = None) -> Dict[str, str]:
    """Training label ``<motion>``: ``{'lon', 'lat', 'text': 'lon=<lon> | lat=<lat>'}``.

    ``intent`` defaults to ``frame_intent(frame)`` (intent_corrected, then intent).
    """
    lon = lon_class(frame)
    lat = lat_class(frame, intent)
    return {'lon': lon, 'lat': lat, 'text': f'lon={lon} | lat={lat}'}


# --------------------------------------------------------------------------------------
# 4. plan text -> trend class
# --------------------------------------------------------------------------------------
_OBJ_NOUN = (r'(?:vehicles?|cars?|trucks?|bus(?:es)?|vans?|suvs?|sedans?|pickups?|trailers?|traffic|'
             r'objects?|cyclists?|motorcyclists?|motorcycles?|bicycl\w*|bikes?|riders?|pedestrians?|'
             r'scooters?|lead|leader|queue|convoy|ambulance|taxis?|cabs?|animals?|dogs?|persons?|people)')
_OBJ_ADJ = (r'(?:stopped|stationary|parked|halted|idling|slow|slow-moving|slowly[\s-]moving|braking|'
            r'yielding|merging|crossing|oncoming|approaching|waiting|queued|queuing|turning|'
            r'left-turning|right-turning|accelerating|decelerating|cruising|moving|passing|following|'
            r'creeping|reversing|backing|departing|starting|stalled|broken[\s-]down|double[\s-]parked)')
_OBJ_MOD = (r'(?:lead|front|ego|white|red|black|silver|blue|gray|grey|green|yellow|orange|brown|large|'
            r'small|big|dark|light|nearby|adjacent|distant|first|second|other|another|the|a|an)')
_ARTICLES = r'(?:a|an|the|its|his|her|their|our|this|that|these|those|my|your)'
_KEEP_VERBS = r'(?:keep|keeps|keeping|kept|maintain|maintains|maintaining|hold|holds|holding|leave|leaving|preserve|preserving)'
_LANE_VERBS = _KEEP_VERBS + r'|stay|stays|staying|remain|remains|remaining'
# verbs that are still actions when they follow "to" (everything else after "to" is a purpose clause)
_TO_ACTIONS = (r'(?:stop|stops|wait|waits|hold|halt|accelerat\w*|decelerat\w*|slow|slows|brake|brakes|speed|'
               r'cruise|maintain|keep|proceed|creep|continue|resume|yield|come|reduce|ease|back|go|move|'
               r'start|remain|stay|stand|creep|coast|pick|get|queue|park)')

_LANE_KEEP_RX = (rf'\b(?:{_LANE_VERBS})\s+(?:in\s+|to\s+|within\s+|on\s+)?(?:{_ARTICLES}\s+)?(?:current\s+|same\s+|own\s+|'
                 r'left\s+|right\s+|center\s+|centre\s+|middle\s+|travel\s+|through\s+)?lanes?\b')     # "keep lane"
_SIDE_KEEP_RX = (rf'\b(?:{_LANE_VERBS})\s+(?:to\s+the\s+|to\s+|on\s+the\s+)?(?:far\s+|slightly\s+)?'
                 r'(?:left|right|center|centre|middle)(?:[\s-]hand)?(?:\s+side)?\b(?!\s+lanes?\b)')     # "keep right"

# Object noun phrases: masked BEFORE the lateral extraction so that another road user's action
# ("the vehicle turning left", "wait for the car to pass") never reads as the ego's own plan.
_OBJ_MASKS = [re.compile(p) for p in (
    rf'\b{_OBJ_ADJ}\s+(?:{_OBJ_MOD}\s+)*{_OBJ_NOUN}\b',            # "stopped car", "turning vehicle"
    rf'\b{_OBJ_NOUN}\s+(?:(?:that|which|who)\s+(?:is|are|was|were)\s+)?{_OBJ_ADJ}'
    r'(?:\s+(?:to\s+the\s+)?(?:left|right))?\b',                    # "vehicle turning left", "truck waiting"
    rf'\b{_OBJ_NOUN}\s+to\s+(?!(?:{_ARTICLES})\s+)(?!(?:stop|wait|halt|stand|come|remain|stay)\b)[a-z]+\b[^,;]*',
                                                                    # "wait for the car to pass / to turn left"
    r'\bstop\s+(?:signs?|lines?|signals?|lights?|bars?)\b',         # "stop sign", "stop line"
    r'\bbus\s+stops?\b',
    r'\byield\s+signs?\b',
    r'\b(?:brake|hazard|turn|tail|head|traffic|red|green|yellow|amber)\s*-?\s*lights?\b',
    r'\bcross\s*-?\s*(?:walks?|traffic|streets?|roads?|lines?)\b',
    r'\b(?:(?:left|right|u)[\s-]?)?turn(?:ing)?\s+(?:lanes?|signals?|areas?|pockets?|arrows?|bays?|only)\b',
                                                                    # "right turn lane" is a place, not a turn
)]
# Clause masks: applied after the object masks, for the longitudinal match only.
_CLAUSE_MASKS = [re.compile(p) for p in (
    rf'\b{_KEEP_VERBS}\s+(?:{_ARTICLES}\s+)?(?:safe\s+|proper\s+|adequate\s+|comfortable\s+|larger\s+|greater\s+|'
    r'sufficient\s+|close\s+|short\s+|normal\s+)?(?:following\s+|trailing\s+)?(?:distance|gap|space|clearance|buffer|headway)\b',
    _LANE_KEEP_RX,
    _SIDE_KEEP_RX,
    r'\bwithout\s+(?:stopping|waiting|slowing|braking|accelerating|decelerating|yielding|halting|pausing)\b',
    rf'\bto\s+(?!{_ARTICLES}\b)(?!{_TO_ACTIONS}\b)[a-z]+\b[^,;]*',   # purpose infinitive "to follow the lead vehicle"
)]
_MASKS = _OBJ_MASKS + _CLAUSE_MASKS

_SEQ_SPLIT = re.compile(
    r'\s*[,;]?\s*\b(?:and\s+)?then\b'
    r'|\s*[,;]?\s*\bfollowed\s+by\b'
    r'|\s*[,;]?\s*\bafter\s+(?:that|which)\b'
    r'|\s*[,;]?\s*\bafterwards?\b'
    r'|\s*[,;]?\s*\bsubsequently\b'
    r'|\s*[,;]?\s*\b(?:and\s+)?finally\b'
    r'|\s*[,;]?\s*\bbefore\s+[a-z]+ing\b')

_STRONG: List[Tuple[str, re.Pattern]] = [(lab, re.compile(rx)) for lab, rx in (
    ('stop',
     r'\bstops?\b|\bstopp(?:ed|ing)\b|\bwait(?:s|ed|ing)?\b'
     r'|\bhold(?:s|ing)?\b(?!\s+(?:the\s+|current\s+|a\s+|steady\s+|its\s+)?(?:speed|pace|lane))'
     r'|\bhalt(?:s|ed|ing)?\b|\bstand(?:s|ing)?\s+still\b|\bstandstill\b'
     r'|\b(?:remain|remains|remaining|stay|stays|staying)\s+(?:stopped|stationary|halted|still|put|parked|at\s+rest|in\s+place|motionless)\b'
     r'|\bstationary\b|\bmotionless\b|\bqueue(?:s|d)?\b|\bqueuing\b|\bqueueing\b|\bidl(?:e|es|ing)\b'
     r'|\bpark(?:s|ed|ing)?\b(?!\s+(?:lot|lots|space|spaces|spot|spots|area|areas|garage|structure))'
     rf'|\blet(?:s|ting)?\s+(?:{_ARTICLES}\s+)?(?:[a-z-]+\s+){{0,4}}?'
     r'(?:cross|pass|clear|go|through|merge|proceed|turn|enter|exit|finish|complete|move)\b'),   # "let the pedestrian cross"
    ('decelerate',
     r'\bdec+eler\w*|\bdecelar\w*|\bslow(?:s|ed|ing)?\b|\bbrak(?:e|es|ed|ing)\b|\byield(?:s|ed|ing)?\b'
     r'|\bgiv(?:e|es|ing)\s+way\b|\bgave\s+way\b|\beas(?:e|es|ed|ing)\s+(?:off|up)\b'
     r'|\breduc(?:e|es|ing)\s+(?:the\s+|its\s+|your\s+)?speed\b|\bback(?:s|ed|ing)?\s+off\b'
     r'|\bapproach(?:es|ed|ing)?\s+(?:cautiously|slowly|carefully|with\s+caution)\b|\bcautiously\s+approach\w*'),
    ('accelerate',
     r'\bac+eler\w*|\bspeed(?:s|ing)?\s+up\b'
     r'|\bstart(?:s|ed|ing)?\b(?!\s+(?:to\s+)?(?:brak|slow|decel|stopp|wait|yield))'
     r'|\bmov(?:e|es|ed|ing)\s+off\b|\bpull(?:s|ed|ing)?\s+away\b|\bresum(?:e|es|ed|ing)\b'
     r'|\blaunch(?:es|ed|ing)?\b|\bpull(?:s|ed|ing)?\s+out\b|\bget(?:s|ting)?\s+going\b|\bset(?:s|ting)?\s+off\b'
     r'|\btak(?:e|es|ing)\s+off\b|\bpick(?:s|ed|ing)?\s+up\s+speed\b|\bincreas(?:e|es|ed|ing)\s+(?:the\s+|its\s+)?speed\b'
     r'|\bgo\s+faster\b|\bdepart(?:s|ed|ing)?\b|\bbegin(?:s|ning)?\s+(?:moving|to\s+move|rolling)\b'),
    ('keep',
     r'\bcruis(?:e|es|ed|ing)\b|\bmaintain(?:s|ed|ing)?\b'
     r'|\bkeep(?:s|ing)?\s+(?:the\s+|its\s+|current\s+|a\s+|your\s+)?(?:pace|speed|momentum)\b'
     r'|\bkeep(?:s|ing)?\s+(?:going|moving|driving|rolling|cruising|up)\b'
     r'|\bhold(?:s|ing)?\s+(?:the\s+|current\s+|a\s+|steady\s+|its\s+)?(?:speed|pace)\b|\bcoast(?:s|ed|ing)?\b'
     r'|\b(?:steady|constant|same|current|low|moderate|slow|reduced|walking|crawling)\s+(?:speed|pace)\b'),
)]

# continuation verbs: only count when no explicit verb is present.
# 'split' -> accelerate when v0<0.5 (from stop) else keep (while moving); 'keep' -> keep regardless of v0.
_WEAK: List[Tuple[str, re.Pattern]] = [(lab, re.compile(rx)) for lab, rx in (
    ('split',
     r'\bproceed(?:s|ed|ing)?\b|\bgo(?:es|ing)?\b|\bwent\b|\bcreep(?:s|ing)?\b|\bcrept\b'
     r'|\bmov(?:e|es|ed|ing)\b|\bdriv(?:e|es|ing)\b|\bdrove\b|\badvanc(?:e|es|ed|ing)\b|\broll(?:s|ed|ing)?\b'
     r'|\bhead(?:s|ed|ing)?\b|\btravel(?:s|ed|led|ing|ling)?\b|\bcross(?:es|ed|ing)?\b|\benter(?:s|ed|ing)?\b'
     r'|\btravers\w*|\bnavigat\w*|\bnegotiat\w*|\binch(?:es|ed|ing)?\s+forward\b'),
    ('keep',
     r'\bcontinu(?:e|es|ed|ing)\b|\bfollow(?:s|ed|ing)?\b|\bpass(?:es|ed|ing)?\b|\bovertak\w*|\bbypass\w*'),
)]

# keyword fallback for out-of-table phrasing (only when nothing above matched)
_FALLBACK: List[Tuple[str, re.Pattern]] = [(lab, re.compile(rx)) for lab, rx in (
    ('stop', r'\bstill\b|\bstand\b|\bno\s+movement\b|\bfreeze\w*|\bpaus\w*'),
    ('decelerate', r'\bcautious\w*|\bcareful\w*|\bgentl\w*|\beas(?:e|y|ing)\b|\breduc\w*|\blower\w*|\blet\b|\bsoft\w*|\bback\s+away'),
    ('accelerate', r'\bfaster\b|\bquick\w*|\bincreas\w*|\bpick\w*\s+up\b|\bget\s+moving\b|\bcommenc\w*|\bbegin\w*|\bboost\w*|\bpush\w*'),
    ('keep', r'\bsteady\b|\bconstant\b|\bsame\s+speed\b|\bmomentum\b|\bpace\b|\btrack\w*\b|\bkeep\w*\b|\bcontinu\w*'),
)]

_TURN_RE = re.compile(
    r'\bturn(?:s|ed|ing)?\s+(?:to\s+the\s+|to\s+|towards?\s+the\s+|towards?\s+)?(?P<a>left|right)\b'
    r'|\b(?P<b>left|right)(?:[\s-]hand)?[\s-]turns?\b'
    r'|\b(?:veer|veers|veered|veering|bear|bears|bearing|steer|steers|steering|swing|swings|swinging|curve|curves|curving)'
    r'\s+(?:to\s+the\s+|to\s+)?(?P<c>left|right)\b'
    r'|\b(?P<u>u)[\s-]?turns?\b')

_LAT_NOTES: List[Tuple[str, re.Pattern]] = [(lab, re.compile(rx)) for lab, rx in (
    ('lane_change', r'\blane[\s-]chang\w*|\bchang(?:e|es|ed|ing)\s+(?:the\s+|a\s+|to\s+the\s+)?lanes?\b|\bmerg\w*'),
    ('nudge', r'\bnudg\w*'),
    ('borrow', r'\bborrow\w*|\blane[\s-]borrow\w*'),
    ('pull_over', r'\bpull(?:s|ed|ing)?\s+over\b'),
    ('pull_out', r'\bpull(?:s|ed|ing)?\s+out\b'),
    ('reverse', r'\brevers\w*|\bback(?:s|ed|ing)?\s+up\b|\bbacking\b|\bbackwards?\b'),
    ('keep_lane', _LANE_KEEP_RX),
    ('keep_side', _SIDE_KEEP_RX),
    ('straight', r'\bstraight\b'),
)]


def _find_hits(patterns: Sequence[Tuple[str, re.Pattern]], text: str) -> List[Tuple[int, int, str, str]]:
    """All non-overlapping (start, end, label, matched_text) hits; earliest start wins, then longest."""
    hits = [(m.start(), m.end(), lab, m.group(0)) for lab, rx in patterns for m in rx.finditer(text)]
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for h in hits:
        if h[0] >= last_end:
            out.append(h)
            last_end = h[1]
    return out


def _normalize_plan(plan_text: Any) -> str:
    s = '' if plan_text is None else str(plan_text)
    s = s.lower().replace('\n', ' ').replace('_', ' ')
    return re.sub(r'\s+', ' ', s).strip(' .')


def plan_first_segment(plan_text: Any) -> str:
    """The first action of a two-part plan ("A, then B" -> "A")."""
    s = _normalize_plan(plan_text)
    parts = [p.strip(' ,;') for p in _SEQ_SPLIT.split(s)]
    parts = [p for p in parts if p]
    return parts[0] if parts else s


def _mask_plan(seg: str, masks: Sequence[re.Pattern] = _MASKS) -> str:
    for rx in masks:
        seg = rx.sub(' ', seg)
    return re.sub(r'\s+', ' ', seg).strip()


def _plan_lat(seg: str) -> Optional[str]:
    m = _TURN_RE.search(seg)
    if not m:
        return None
    if m.group('u'):
        return 'left_turn'
    d = m.group('a') or m.group('b') or m.group('c')
    return 'left_turn' if d == 'left' else 'right_turn'


def plan_to_motion(plan_text: Any, v0: Optional[float]) -> Dict[str, Any]:
    """Map a ``final_plan`` string to the trend closed set.

    Returns ``{'lon': accelerate|decelerate|keep|stop|unspecified|unmatched,
    'lat': left_turn|right_turn|None, 'matched_by': str, 'lat_notes': tuple, 'segment': str}``.

    Steps: normalise -> take the first segment of a two-part plan -> mask object noun phrases
    (another road user's action, "stop sign", "right turn lane", ...) -> ``lat`` = the first turn
    mention left (U-turn -> left_turn) -> mask clauses (purpose infinitives, "keep lane", "keep a
    safe distance", "without stopping") -> the last explicit longitudinal verb
    (stop/decelerate/accelerate/keep tables) wins; otherwise the last continuation verb
    (proceed/creep/go/... are ``accelerate`` from stop (v0<0.5) and ``keep`` while moving;
    continue/follow/pass -> keep); otherwise the keyword fallback; otherwise ``unspecified``
    (a lateral-only plan) or ``unmatched``.
    ``lat_notes`` records lateral phrases that do not enter the closed set (lane_change, nudge,
    borrow, pull_over, pull_out, reverse, keep_lane, keep_side, straight).  ``matched_by`` says
    which table/word decided.  ``v0=None`` is treated as moving.
    """
    seg = plan_first_segment(plan_text)
    obj_masked = _mask_plan(seg, _OBJ_MASKS)
    masked = _mask_plan(obj_masked, _CLAUSE_MASKS)
    moving = v0 is None or float(v0) >= STOP_SPEED
    v0_tag = 'v0=?' if v0 is None else ('v0>=0.5' if moving else 'v0<0.5')
    lat = _plan_lat(obj_masked)
    notes = tuple(dict.fromkeys(lab for lab, rx in _LAT_NOTES if rx.search(obj_masked)))
    out: Dict[str, Any] = {'lon': 'unmatched', 'lat': lat, 'matched_by': 'unmatched',
                           'lat_notes': notes, 'segment': seg}
    if not masked:
        out['matched_by'] = 'empty_plan' if not seg else 'unmatched'
        if lat or notes:
            out['lon'], out['matched_by'] = 'unspecified', 'unspecified:lateral_only'
        return out
    strong = _find_hits(_STRONG, masked)
    if strong:
        _, _, lab, word = strong[-1]
        out['lon'], out['matched_by'] = lab, f'table:{word}'
        return out
    weak = _find_hits(_WEAK, masked)
    if weak:
        _, _, lab, word = weak[-1]
        if lab == 'split':
            out['lon'] = 'keep' if moving else 'accelerate'
            out['matched_by'] = f'table:{word}@{v0_tag}'
        else:
            out['lon'], out['matched_by'] = 'keep', f'table:{word}'
        return out
    fb = _find_hits(_FALLBACK, masked)
    if fb:
        _, _, lab, word = fb[-1]
        out['lon'], out['matched_by'] = lab, f'fallback:{word}'
        return out
    if lat or notes:
        out['lon'], out['matched_by'] = 'unspecified', 'unspecified:lateral_only'
    return out


# ------------------------------------------------------------------- plan -> plan-vocabulary labels
# The *plan vocabulary* of the dataset (used by the closed-set final_plan metric): 10 lateral x
# 4 longitudinal.  It is NOT the same closed set as ``<motion>``: motion is the model's trend field
# (lon = accelerate|decelerate|keep|stop), the plan vocabulary keeps the dataset's own word
# (``cruise`` where motion says ``keep``).  Everything else -- first segment, object masking, last
# explicit verb, v0 splitting, the keyword fallback -- is inherited verbatim from
# ``plan_to_motion`` so there is exactly ONE plan lexicon in the repo.
LAT_TAXONOMY = ("keep lane", "turn left", "turn right", "lane change left", "lane change right",
                "nudge left", "nudge right", "pull out", "pull over", "bypass/overtake")
LON_TAXONOMY = ("cruise", "accelerate", "decelerate", "stop")
_MOTION_LON_TO_PLAN = {"keep": "cruise", "accelerate": "accelerate",
                       "decelerate": "decelerate", "stop": "stop"}
_LEFT_RX, _RIGHT_RX = re.compile(r'\bleft\b'), re.compile(r'\bright\b')


def _side_of(seg: str) -> Optional[str]:
    """'left' / 'right' / None -- whichever direction word comes first in the (object-masked) text."""
    ml, mr = _LEFT_RX.search(seg), _RIGHT_RX.search(seg)
    if ml and mr:
        return 'left' if ml.start() < mr.start() else 'right'
    if ml:
        return 'left'
    return 'right' if mr else None


def plan_to_taxonomy(plan_text: Any, v0: Optional[float]) -> Dict[str, Any]:
    """Map a ``final_plan`` string to the dataset plan vocabulary.

    Returns ``{'lateral': one of LAT_TAXONOMY or 'other', 'longitudinal': one of LON_TAXONOMY or
    'unspecified'/'unmatched', 'motion': the raw plan_to_motion dict}``.

    Lateral priority: explicit turn (U-turn -> turn left) > lane change > nudge > pull out >
    pull over > lane borrowing/bypass > lane/side keeping or "straight" > fall back to
    ``keep lane`` when a longitudinal action was recognised, else ``other``.  ``reverse`` is in
    the lexicon but never occurs in this dataset, so it is folded into ``other``.
    """
    res = plan_to_motion(plan_text, v0)
    seg = _mask_plan(plan_first_segment(plan_text), _OBJ_MASKS)
    notes = set(res['lat_notes'])
    if res['lat'] == 'left_turn':
        lat = 'turn left'
    elif res['lat'] == 'right_turn':
        lat = 'turn right'
    elif 'lane_change' in notes:
        lat = 'lane change ' + (_side_of(seg) or 'left')
    elif 'nudge' in notes:
        lat = 'nudge ' + (_side_of(seg) or 'left')
    elif 'pull_out' in notes:
        lat = 'pull out'
    elif 'pull_over' in notes:
        lat = 'pull over'
    elif 'borrow' in notes:
        lat = 'bypass/overtake'
    elif notes & {'keep_lane', 'keep_side', 'straight'}:
        lat = 'keep lane'
    elif res['lon'] in _MOTION_LON_TO_PLAN:
        lat = 'keep lane'                       # longitudinal-only plan -> lane keeping
    else:
        lat = 'other'
    lon = _MOTION_LON_TO_PLAN.get(res['lon'], res['lon'])
    return {'lateral': lat, 'longitudinal': lon, 'motion': res}


def plan_lon_compatible(plan_lon: str, motion_lon: Optional[str], v0: Optional[float],
                        v_end: Optional[float] = None, start_t: Optional[float] = None) -> bool:
    """Plan/motion compatibility matrix (P = plan class, M = ``motion.lon``): True when P does not
    contradict M.  Class equality is always compatible; the other cells are

    * P=keep       : M in {accelerate, decelerate} and v0 >= 0.5 and |v_end - v0| <= max(3.0, 0.5*v0)
    * P=decelerate : M == stop; or M == keep and v_end < v0
    * P=accelerate : M == keep and v_end >= v0 - 0.5
    * P=stop       : M == decelerate; or (M == accelerate and v0 < 0.5 and start_t >= 2.0);
                     or (M == keep and v_end < 1.5)

    ``v_end`` / ``start_t`` / ``v0`` = None means "unknown": every cell that needs the missing
    quantity is a contradiction (only the pure class-set cells survive).  ``plan_lon`` outside the
    closed set (unspecified / unmatched) is never compatible here -- ``plan_consistent`` returns
    ``None`` for those before calling this function.
    """
    if plan_lon not in MOTION_LON or motion_lon not in MOTION_LON:
        return False
    if plan_lon == motion_lon:
        return True
    v0f = None if v0 is None else float(v0)
    ve = None if v_end is None else float(v_end)
    if plan_lon == 'keep':
        return (motion_lon in ('accelerate', 'decelerate') and v0f is not None and ve is not None
                and v0f >= STOP_SPEED and abs(ve - v0f) <= max(TOL_KEEP_DV_ABS, TOL_KEEP_DV_REL * v0f))
    if plan_lon == 'decelerate':
        return motion_lon == 'stop' or (motion_lon == 'keep' and v0f is not None and ve is not None and ve < v0f)
    if plan_lon == 'accelerate':
        return motion_lon == 'keep' and v0f is not None and ve is not None and ve >= v0f - TOL_ACC_KEEP_DV
    if plan_lon == 'stop':
        if motion_lon == 'decelerate':
            return True
        if motion_lon == 'accelerate':
            return (v0f is not None and v0f < STOP_SPEED and start_t is not None
                    and float(start_t) >= TOL_STOP_START_S)
        return motion_lon == 'keep' and ve is not None and ve < TOL_STOP_KEEP_VEND
    return False


def plan_consistent(plan_text: Any, motion: Any, v0: Optional[float], *, strict: bool = False,
                    frame: Optional[Dict[str, Any]] = None, v_end: Optional[float] = None,
                    start_t: Optional[float] = None) -> Optional[bool]:
    """Consistency of a ``final_plan`` with the motion trend.

    Default (``strict=False``): the plan class must be *compatible* with ``motion.lon`` under the
    matrix of ``plan_lon_compatible`` (not contradictory == consistent) and there must be no lateral
    contradiction.  ``strict=True``: the former rule ``class(plan) == motion.lon`` and no lateral
    contradiction.

    ``motion`` is the dict from ``motion_label`` (``{'lon', 'lat'}``) or a bare ``motion.lon`` string
    (lateral unknown, not judged).  ``v_end`` (mean speed of the last future second) and ``start_t``
    (first future time with v >= 0.5 m/s) are derived from ``frame['future_states']`` when ``frame``
    is given (explicit keyword values win); ``v0`` falls back to ``v0_of(frame)`` when None.
    Returns ``None`` when the plan is ``unspecified``/``unmatched`` (not judged).  Lateral
    contradiction only when the plan names a turn and ``motion.lat`` is ``straight`` (or the
    opposite turn), or the plan explicitly says ``straight`` while ``motion.lat`` is a turn; a plan
    that does not mention the lateral direction never contradicts.
    """
    if frame is not None and v0 is None:
        v0 = v0_of(frame)
    pm = plan_to_motion(plan_text, v0)
    if pm['lon'] not in MOTION_LON:
        return None
    if isinstance(motion, dict):
        m_lon, m_lat = motion.get('lon'), motion.get('lat')
    else:
        m_lon, m_lat = motion, None
    if strict:
        lon_ok = pm['lon'] == m_lon
    else:
        if frame is not None:
            if v_end is None:
                v_end = future_v_end(frame)
            if start_t is None:
                start_t = future_start_time(frame)
        lon_ok = plan_lon_compatible(pm['lon'], m_lon, v0, v_end=v_end, start_t=start_t)
    if not lon_ok:
        return False
    if m_lat is None:
        return True
    if pm['lat'] in ('left_turn', 'right_turn'):
        return m_lat == pm['lat']
    if 'straight' in pm['lat_notes'] and m_lat in ('left_turn', 'right_turn'):
        return False
    return True


# --------------------------------------------------------------------------------------
# 5. trajectory target, 6. attribute helpers, class balance
# --------------------------------------------------------------------------------------
def traj_points_1hz(frame: Dict[str, Any]) -> List[Tuple[float, float]]:
    """GT future points at t = 1..5 s (indices 3/7/11/15/19), each rounded to 1 decimal.

    Raises ``ValueError`` if the frame has fewer than 20 future points.
    """
    pts = future_xy(frame)
    if len(pts) < FUT_STEPS:
        raise ValueError(f'traj_points_1hz: need {FUT_STEPS} future points, got {len(pts)}')
    return [(_round1(pts[i][0]), _round1(pts[i][1])) for i in TRAJ_1HZ_IDX]


def attr_first(attributes: Optional[Dict[str, Any]], field: str) -> str:
    """The attribute value that gets emitted: first element of a list, '' for None."""
    v = (attributes or {}).get(field)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return '' if v is None else str(v)


def _attr_norm(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return '' if v is None else str(v).strip().lower()


def minority_attr_object(ann: Dict[str, Any]) -> bool:
    """True when the object's intention or state is outside {cruising, stopping, parking, crossing, ''}."""
    at = ann.get('attributes') or {}
    return any(_attr_norm(at.get(f)) not in MAJORITY_ATTR for f in ('intention', 'state'))


def minority_attr_frame(pano: Dict[str, Any]) -> bool:
    """True when any annotated object has a minority intention/state (used for w_attr and Attr-QA)."""
    return any(minority_attr_object(a) for a in (pano.get('annotations') or []))


def class_balance_factor(value: Any, freq_table: Optional[Dict[str, Dict[str, float]]], field: str) -> float:
    """Attribute class-balance factor ``g(c) = clip(sqrt(f_med / f_c), 0.5, 3.0)``.

    ``freq_table = {'intention': {value: count}, 'state': {...}}``; ``f_med`` is the median count of
    the field.  Unknown value / unknown field / empty table / zero count -> 1.0.  Lookup is exact
    first, then case- and whitespace-insensitive; list values use their first element.
    """
    tbl = (freq_table or {}).get(field) or {}
    counts = [float(c) for c in tbl.values() if isinstance(c, (int, float)) and c > 0]
    if not counts:
        return 1.0
    key = _attr_norm(value) if isinstance(value, (list, tuple)) else value
    if key not in tbl:
        norm = _attr_norm(key)
        key = next((k for k in tbl if _attr_norm(k) == norm), None)
        if key is None:
            return 1.0
    f = tbl[key]
    if not isinstance(f, (int, float)) or f <= 0:
        return 1.0
    f_med = statistics.median(counts)
    lo, hi = BALANCE_CLIP
    return float(min(hi, max(lo, math.sqrt(f_med / float(f)))))


def sorted_with_ranks(anns: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    """Objects sorted by ``impact_rank`` (missing -> 99, stable by appearance).

    Returns ``[(annotation, rank_to_write)]``: the original ``impact_rank`` when present, otherwise
    the object's 1-based position in the sorted list.
    """
    def key(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, int]:
        r = (item[1].get('attributes') or {}).get('impact_rank')
        return (float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else 99.0, item[0])

    ordered = sorted(enumerate(anns), key=key)
    out: List[Tuple[Dict[str, Any], int]] = []
    for pos, (_, a) in enumerate(ordered, start=1):
        r = (a.get('attributes') or {}).get('impact_rank')
        out.append((a, int(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else pos))
    return out
