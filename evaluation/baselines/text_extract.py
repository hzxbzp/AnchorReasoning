"""Tolerant extraction of the chain fields from FREE TEXT: pick the fields out of whatever prose
the model wrote.

Released models asked with SYSTEM_S2 rarely honour the tag grammar, and native inference scripts
never see it; the strict ``parse_output`` then returns an empty answer and every metric reads 0.
``extract(text)`` first runs the strict parser and then fills ONLY the missing pieces from the prose:

* objects   -- type mentions from a synonym table over the dataset's 19 types.  Ego mentions
               ("the ego vehicle", "our car", "we") are masked out first; negated mentions ("no
               pedestrians", "don't see any cyclists") are skipped; a specific match ("school bus")
               consumes its span so the generic pattern ("bus") cannot double-count it; one object per
               (type, sentence) plus one per explicit coordinate.  Coordinates are recognised in every
               format the released models use (``<point>x,y</point>``, ``(x, y)``, ``[x, y]``,
               ``x=1420, y=600``, ``[x1, y1, x2, y2]`` boxes, Qwen ``{"bbox_2d": [...], "label": ...}``
               JSON, ``<|box_start|>(x1,y1),(x2,y2)<|box_end|>``) and attached to the NEAREST preceding
               mention.  A behaviour word in the mention's clause becomes ``state``; a colour word next
               to a traffic light becomes ``content``; the sentence is the object's ``implication``.
* reason    -- ``Reason:`` / ``**Reason:**`` / ``## Reason`` label, else a causal sentence
               (because / so / therefore); ``<think>`` text is used only as a last resort
* final_plan-- ``Plan:`` / ``Action:`` label, else "the ego / our vehicle / we should ..." sentence
* traj      -- a run of >= 5 ``[x, y]`` / ``(x, y)`` float pairs, preferring the run introduced by a
               trajectory keyword, else the LAST run (a restated history comes first); resampled
               to t = 1..5 s
* context   -- ``weather: Sunny`` style key-value mentions for the five dimensions

Every filled field is listed in ``pred['_provenance']``.  Nothing here touches ``parse_output``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.core.parse_output import parse_output
from training.core.metrics.behaviour import canon_behaviour

__all__ = ["extract", "extract_objects", "extract_traj5", "TYPE_PATTERNS", "CONTEXT_KEYS", "strip_think"]

# canonical dataset type -> regex over lowercase prose.  Specific before generic; a match consumes
# its span so later (generic) patterns cannot re-use it.
TYPE_PATTERNS: Sequence[Tuple[str, str]] = (
    ("School bus",           r"\bschool[\s-]?bus(?:es)?\b"),
    ("Emergency vehicle",    r"\b(?:ambulance|police (?:car|vehicle|cruiser|suv)|fire ?truck|fire engine|emergency vehicle)s?\b"),
    ("Construction vehicle", r"\b(?:excavator|bulldozer|construction (?:vehicle|truck)|cement mixer|crane truck|road roller|dump truck)s?\b"),
    ("Bus",                  r"\bbus(?:es)?\b(?![\s-]?(?:stop|lane|station|shelter))"),
    ("Truck",                r"\b(?:truck|pickup|lorry|semi[\s-]?trailer|tractor[\s-]?trailer|trailer)s?\b"),
    ("Car",                  r"\b(?:car|sedan|suv|hatchback|taxi|van|minivan|coupe|jeep)s?\b(?![\s-]?(?:park|lane|door))"),
    ("vehicle",              r"\b(?:motor )?vehicles?\b(?![\s-]?(?:lane|door))"),           # class-level only
    ("Motorcyclist",         r"\b(?:motorcycl(?:e|es|ist|ists)|motorbike(?:s)?|biker(?:s)?)\b"),
    ("Scooter rider",        r"\b(?:scooter|e[\s-]?scooter|moped)s?(?: rider)?\b"),
    ("Cyclist",              r"\b(?:cyclist|bicycl(?:e|es|ist|ists)|bike rider|bike)s?\b(?![\s-]?(?:lane|path|rack))"),
    ("Pedestrian",           r"\b(?:pedestrian|person|people|man|woman|walker|child|jogger|runner)s?\b(?![\s-]?crossing)"),
    ("Traffic light",        r"\b(?:traffic (?:light|signal)|(?:red|green|yellow|amber) (?:light|signal)|stoplight|"
                             r"(?:the )?(?:light|signal) (?:is|was|turns?|turned|remains|stays) (?:red|green|yellow|amber))s?\b"),
    ("Stop line",            r"\bstop[\s-]?line\b"),
    ("Cross walk",           r"\b(?:cross[\s-]?walk|zebra crossing|pedestrian crossing)s?\b"),
    ("Bump",                 r"\bspeed[\s-]?(?:bump|hump)s?\b"),
    ("Temporary control",    r"\b(?:temporary (?:traffic )?control|flagger|detour sign|road[\s-]?work sign|construction sign)s?\b"),
    ("Sign",                 r"\b(?:stop sign|yield sign|speed limit sign|no[\s-]?(?:left|right)[\s-]?turn sign|(?:traffic |road )?sign(?:age|s)?)\b"),
    ("Animal",               r"\b(?:animal|dog|deer|cat|bird|horse|cow)s?\b"),
    ("Object",               r"\b(?:cone|traffic cone|barrier|barricade|bollard|debris|obstacle|(?<!bounding )box|pothole|drum|pylon)s?\b"),
)
_TYPE_RX = [(t, re.compile(rx)) for t, rx in TYPE_PATTERNS]
_CONTROL = {"Traffic light", "Sign", "Stop line", "Cross walk", "Bump", "Temporary control"}
_SIGNAL = "Traffic light"

CONTEXT_KEYS = ("weather", "daytime", "visibility", "scenario", "road")
_CTX_RX = {k: re.compile(rf"\b{k}\**\s*[:=]\s*\**\s*([A-Za-z][A-Za-z\- ]{{0,24}}?)\**(?=[;,.\n)]|$)", re.I) for k in CONTEXT_KEYS}
_EVENTS_RX = re.compile(r"\btraffic[\s_]?events?\**\s*[:=]\s*\**\s*([^\n]+?)\**\s*(?=\n|$)", re.I)
_MOTION_RX = re.compile(r"\blon\s*=\s*([a-z_]+)\s*\|\s*lat\s*=\s*([a-z_]+)", re.I)
# structured attribute fields written inside an object line, e.g.
#   "**Car** (point: [1600, 740], impact rank: 1, attributes: location: right lane, intention: driving straight, state: moving)"
_ATTR_RX = {k: re.compile(rf"\b{k}\s*[:\-\u2013=]\s*([^,;)\n]+)", re.I) for k in ("location", "intention", "state", "content")}
_IMPL_RX = re.compile(r"\bimplication\s*[:\-\u2013=]\s*([^\n)]+)", re.I)
_RANK_RX = re.compile(r"\b(?:impact[\s_]?)?rank\s*[:\-\u2013=]\s*(\d+)", re.I)
# wording the released models use for the dataset's behaviour labels (canon_behaviour handles the rest)
_BEH_SYN = (("slowing down", "decelerating"), ("slowing", "decelerating"), ("slow down", "decelerating"),
            ("speeding up", "accelerating"), ("speed up", "accelerating"), ("moving", "cruising"),
            ("driving", "cruising"), ("going straight", "cruising"), ("proceeding", "cruising"),
            ("waiting", "stopping"), ("stationary", "stopping"), ("halted", "stopping"), ("idle", "stopping"),
            ("reversing", "backing"), ("merging", "lane change"))
_EFFECT = re.compile(r"\b(?:block|constrain|narrow|yield|slow|stop|wait|proceed|clear|release|allow|require|must|should|"
                     r"limit|force|prevent|occup|obstruct|give way|pass|overtake|follow)", re.I)
_SECTION_END = re.compile(r"(?:^|\n)\s*(?:#+\s*)?\**\s*(?:reason(?:ing)?|rationale|final[\s_-]?plan|plan|motion|trajectory)\b", re.I)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_EGO = re.compile(r"\b(?:the |our |my )?(?:ego|own|host)(?:[\s-]?(?:vehicle|car))?\b|\bour (?:vehicle|car)\b|\bwe\b|\bus\b|\bourselves\b", re.I)
_NEG_BEFORE = re.compile(r"(?:\bno\b|\bnot\b|\bwithout\b|\bany\b|n't\s+(?:see|detect|notice)|\bdo not (?:see|detect)|\babsence of\b|"
                         r"\bfree of\b|\bneither\b|\bnor\b|\bnone\b|\bzero\b|\bisn't\b|\baren't\b)[^.;,]{0,30}$", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_MD = re.compile(r"[*_#`>]+")

_NUM = r"-?\d+(?:\.\d+)?"
_PAIR_PAREN = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)")
_PAIR_BRACK = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_PAIR_XY = re.compile(rf"\bx\s*[=:]\s*({_NUM})\s*,?\s*y\s*[=:]\s*({_NUM})", re.I)
_BOX4 = re.compile(rf"[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*[\])]")
_POINT_TAG = re.compile(rf"<point>\s*({_NUM})\s*,\s*({_NUM})\s*</point>")
_BOX_TOKENS = re.compile(r"<\|box_start\|>\s*\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>")
_FLOAT_PAIR = re.compile(rf"[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*[\])]")
_JSON_ITEM = re.compile(r"\{[^{}]*\}")
_TRAJ_KW = re.compile(r"\b(?:trajectory|waypoints?|future (?:path|positions?|points)|traj|planned path|path)\b[^\[(]{0,60}$", re.I)

_LABEL = r"(?:[#*\-\d.)\s]*)"          # markdown / list prefixes allowed before a label
_REASON_RX = re.compile(rf"(?:^|(?<=[\n.!?;])){_LABEL}\**(?:reason(?:ing)?|rationale|why)\**\s*:\s*\**\s*([^\n]+?)\s*\**\s*"
                        rf"(?=\s*\**(?:final[\s_-]?plan|plan|action|decision|trajectory|waypoints?)\**\s*:|\n|$)", re.I)
_REASON_HDR = re.compile(r"(?:^|\n)\s*#+\s*(?:reason(?:ing)?|rationale)\s*\n+\s*([^\n]+)", re.I)
_PLAN_RX = re.compile(rf"(?:^|(?<=[\n.!?;])){_LABEL}\**(?:final[\s_-]?plan|plan|action|decision|maneuver|manoeuvre|recommended action)\**\s*:\s*\**\s*([^\n]+?)\s*\**\s*"
                      rf"(?=\s*\**(?:trajectory|traj|reason(?:ing)?|waypoints?|motion)\**\s*:|\n|$)", re.I)
_PLAN_HDR = re.compile(r"(?:^|\n)\s*#+\s*(?:final[\s_-]?plan|plan|action)\s*\n+\s*([^\n]+)", re.I)
_EGO_SHOULD = re.compile(r"\b(?:the )?(?:ego(?: vehicle| car)?|our (?:vehicle|car)|we|the car|the vehicle)\s+(?:should|will|must|needs? to|has to|have to|can|may|ought to)\s+([^.;\n]{3,160})", re.I)
_CAUSAL = re.compile(r"\b(?:because|so that|therefore|as a result|which means|so the ego|so it|so we|hence|due to)\b", re.I)
_NO_OBJECTS = re.compile(r"\b(?:no|there are no|without any|not any) (?:key |critical |relevant |notable |decision[\s-]critical )?(?:objects?|elements?|agents?|road users?|obstacles?)\b", re.I)
_COLOUR = re.compile(r"\b(red|green|yellow|amber)\b")


def strip_think(text: str) -> Tuple[str, str]:
    """``(answer without <think> blocks, concatenated think text)``."""
    thinks = " ".join(m.group(0)[7:-8] for m in _THINK.finditer(text or ""))
    return _THINK.sub(" ", text or ""), thinks


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None                         # NaN guard


def _to_mode(pt: Tuple[float, float], point_mode: str) -> Optional[Tuple[float, float]]:
    """Coerce a raw coordinate into the adapter's convention: 0-1 fractions and mismatched
    pixel/normalised ranges are converted rather than silently dropped."""
    x, y = pt
    if x < 0 or y < 0:
        return None
    if x <= 1.0 and y <= 1.0:                            # fraction of the image
        return (x * 1000, y * 1000) if point_mode == "norm1000" else (x * 2916, y * 1079)
    if point_mode == "norm1000":
        if x > 1000 or y > 1000:                          # looks like panorama pixels
            return (x / 2916 * 1000, y / 1079 * 1000) if (x <= 2916 and y <= 1079) else None
        return (x, y)
    if x <= 1000 and y <= 1000 and (x > 2916 or y > 1079):
        return None
    return (x, y) if (x <= 2916 and y <= 1079) else None


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Qwen-style grounding JSON: ``{"bbox_2d": [x1,y1,x2,y2], "label": "car"}`` / ``"point_2d"``."""
    out = []
    for m in _JSON_ITEM.finditer(text or ""):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        label = d.get("label") or d.get("type") or d.get("name") or d.get("category")
        pt = None
        bb = d.get("bbox_2d") or d.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            vals = [_num(v) for v in bb]
            if all(v is not None for v in vals):
                pt = ((vals[0] + vals[2]) / 2, (vals[1] + vals[3]) / 2)
        for k in ("point_2d", "point", "center"):
            v = d.get(k)
            if pt is None and isinstance(v, (list, tuple)) and len(v) == 2:
                a, b = _num(v[0]), _num(v[1])
                if a is not None and b is not None:
                    pt = (a, b)
        if label or pt:
            out.append({"label": str(label) if label else None, "point": pt, "span": (m.start(), m.end())})
    return out


def _type_of(label: Optional[str]) -> Optional[str]:
    s = _EGO.sub(" ", (label or "").lower())
    for t, rx in _TYPE_RX:
        if rx.search(s):
            return t
    return None


def _mentions(low: str) -> List[Tuple[str, int, int]]:
    """Non-overlapping (type, start, end) mentions; specific patterns consume their span first;
    negated mentions are dropped."""
    taken: List[Tuple[int, int]] = []
    found: List[Tuple[str, int, int]] = []
    for t, rx in _TYPE_RX:
        for m in rx.finditer(low):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            if _NEG_BEFORE.search(low[max(0, s - 40):s]):
                taken.append((s, e))                       # consume, but do not emit
                continue
            taken.append((s, e)); found.append((t, s, e))
    found.sort(key=lambda z: z[1])
    return found


def _coords_in(sent: str, point_mode: str) -> List[Tuple[int, Tuple[float, float]]]:
    """``[(position, point)]`` in the adapter's convention."""
    out: List[Tuple[int, Tuple[float, float]]] = []
    used: List[Tuple[int, int]] = []

    def add(m: "re.Match", pt: Tuple[float, float]) -> None:
        if any(m.start() < e and m.end() > s for s, e in used):
            return
        p = _to_mode(pt, point_mode)
        if p is not None:
            used.append(m.span()); out.append((m.start(), p))
    for m in _POINT_TAG.finditer(sent):
        add(m, (float(m.group(1)), float(m.group(2))))
    for m in _BOX_TOKENS.finditer(sent):
        x1, y1, x2, y2 = [float(g) for g in m.groups()]
        add(m, ((x1 + x2) / 2, (y1 + y2) / 2))
    for m in _BOX4.finditer(sent):
        x1, y1, x2, y2 = [float(g) for g in m.groups()]
        if x2 > x1 and y2 > y1:
            add(m, ((x1 + x2) / 2, (y1 + y2) / 2))
    for rx in (_PAIR_XY, _PAIR_PAREN, _PAIR_BRACK):
        for m in rx.finditer(sent):
            x, y = float(m.group(1)), float(m.group(2))
            if rx is _PAIR_BRACK and abs(x) < 30 and abs(y) < 30 and ("." in m.group(1) or "." in m.group(2)):
                continue                                  # a trajectory waypoint in metres, not a pixel
            if x > 30 or y > 30 or (x <= 1.0 and y <= 1.0):
                add(m, (x, y))
    out.sort()
    return out


def _clause(sent: str, start: int, end: int) -> str:
    """The clause around a mention: from the previous ',;' to the next ',;' ."""
    a = max(sent.rfind(",", 0, start), sent.rfind(";", 0, start)) + 1
    b = min([i for i in (sent.find(",", end), sent.find(";", end)) if i >= 0] or [len(sent)])
    return sent[a:b]


_TAG_START = re.compile(rf"<point>\s*\(?\s*({_NUM})\s*,\s*({_NUM})\s*\)?\s*</point>|<obj\s+type=[\"']?([^>\"']+?)[\"']?\s*>", re.I)
_TAG_FIELD = re.compile(r"<\s*(impact[\s_]?rank|rank|object[\s_]?type|type|location|intention\s+and\s+state|intention|state|"
                        r"content|implication)\s*>\s*(.*?)\s*</\s*\1\s*>", re.I | re.S)
_TAG_END = re.compile(r"</objects>|</scene>|<reason>|<final_plan>|<n_objects>|<motion>|<traj>", re.I)


def _tagged_objects(text: str, point_mode: str = "abs_pixel") -> List[Dict[str, Any]]:
    """Near-grammar object blocks that the strict parser rejects, e.g.:

        <objects><point>100, 500</point><impact_rank>1</impact_rank><object_type>Construction vehicle</object_type>
        <location>left lane, ahead</location><intention and state>stationary, blocking lane</intention and state>
        <implication>forces ego to slow</implication></objects><point>300, 600</point>...

    Every ``<point>x,y</point>`` (or an unclosed ``<obj type=T>``) opens a block that runs to the next
    opener or a section terminator; the tagged fields inside are read as the grammar's own fields.
    A degenerate multi-pair point tag does not open a block."""
    answer, _ = strip_think(text)
    starts = list(_TAG_START.finditer(answer))
    if not starts:
        return []
    objs: List[Dict[str, Any]] = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(answer)
        t_end = _TAG_END.search(answer, m.end(), end)
        if t_end:
            end = t_end.start()
        block = answer[m.end():end]
        fields: Dict[str, str] = {}
        for fm in _TAG_FIELD.finditer(block):
            key = re.sub(r"[\s_]+", " ", fm.group(1).lower())
            fields.setdefault(key, fm.group(2).strip())
        label = fields.get("object type") or fields.get("type") or m.group(3)
        t = _type_of(label) if label else None
        pt = _to_mode((float(m.group(1)), float(m.group(2))), point_mode) if m.group(1) is not None else None
        if t is None and pt is None:
            continue
        rank_s = fields.get("impact rank") or fields.get("rank")
        rank = int(re.search(r"\d+", rank_s).group(0)) if rank_s and re.search(r"\d+", rank_s) else None
        state = fields.get("state"); intention = fields.get("intention")
        if fields.get("intention and state") and not (state or intention):
            toks = [x.strip() for x in re.split(r",|;|\band\b|/", fields["intention and state"]) if x.strip()]
            st = [x for x in toks if _syn(x) != "other"]
            state = st[0] if st else None
            rest = [x for x in toks if x not in st]
            intention = ", ".join(rest) if rest else None
        t = t or "Other"
        beh = _syn(state) if state else None
        intent = _syn(intention) if intention else None
        col = _COLOUR.search(fields.get("content") or "") if t in _CONTROL else None
        content = (col.group(1) if col else fields.get("content")) or None
        if content == "amber":
            content = "yellow"
        objs.append({"type": t, "point": pt, "rank": rank, "location": fields.get("location"),
                     "intention": (intent if intent and intent != "other" else (intention or None)),
                     "state": (beh if (t not in _CONTROL and beh and beh != "other") else None),
                     "content": content, "implication": fields.get("implication") or None, "_from": "tagged"})
    return objs


def extract_objects(text: str, point_mode: str = "abs_pixel") -> List[Dict[str, Any]]:
    """Object mentions -> parse_output-shaped object dicts (type, point, state, content, implication)."""
    answer, _ = strip_think(text)
    objs: List[Dict[str, Any]] = []
    have: set = set()
    jitems = _json_objects(answer)
    for j in jitems:
        t = _type_of(j["label"])
        if t is None and j["point"] is None:
            continue
        pt = _to_mode(j["point"], point_mode) if j["point"] is not None else None
        objs.append({"type": t or "Other", "point": pt, "rank": None, "location": None, "intention": None,
                     "state": None, "content": None, "implication": j["label"] or None, "_from": "json"})
        have.add(t or "Other")
    prose = answer
    for j in reversed(jitems):                            # remove the JSON blocks before the prose scan
        s, e = j["span"]; prose = prose[:s] + " " + prose[e:]
    m_end = _SECTION_END.search(prose)                    # objects are listed BEFORE the reason/plan sections;
    if m_end and m_end.start() > 0:                       # re-mentions inside those sections are not new objects
        prose = prose[:m_end.start()]
    for si, sent in enumerate(_sentences(prose)):
        if _NO_OBJECTS.search(sent):
            continue
        low = _EGO.sub(lambda m: " " * len(m.group(0)), sent.lower())   # keep offsets, mask out the ego
        ments = _mentions(low)
        if not ments:
            continue
        coords = _coords_in(sent, point_mode)
        # nearest PRECEDING mention takes each coordinate; a mention keeps at most one
        owner: Dict[int, Tuple[float, float]] = {}
        extra: List[Tuple[str, Tuple[float, float]]] = []
        for pos, pt in coords:
            cands = [k for k, (_, s, _) in enumerate(ments) if s <= pos]
            k = cands[-1] if cands else 0
            if k in owner:
                extra.append((ments[k][0], pt))           # plural mention with several points
            else:
                owner[k] = pt
        fields = {k: (rx.search(sent).group(1).strip() if rx.search(sent) else None) for k, rx in _ATTR_RX.items()}
        impl_m = _IMPL_RX.search(sent); rank_m = _RANK_RX.search(sent)
        structured = any(fields.values()) or rank_m is not None
        for k, (t, s, e) in enumerate(ments):
            pt = owner.get(k)
            if t in have and pt is None and any(o["_from"] == "json" for o in objs):
                continue                                  # already grounded through JSON
            cl = _clause(low, s, e)
            beh = _syn(fields["state"]) if fields["state"] else canon_behaviour(cl) if t not in _CONTROL else "other"
            if beh == "other" and t not in _CONTROL and not fields["state"] and \
                    sum(1 for tt, _, _ in ments if tt not in _CONTROL) == 1:
                beh = canon_behaviour(low)                # single agent in the sentence -> whole sentence is about it
            intention = _syn(fields["intention"]) if fields["intention"] else None
            col = _COLOUR.search(fields["content"] or (cl if t == _SIGNAL else "")) if t in _CONTROL else None
            content = (fields["content"] if fields["content"] and not col else
                       col.group(1) if col else ("stop" if (t == "Sign" and "stop sign" in cl) else
                                                 ("yield" if (t == "Sign" and "yield" in cl) else None)))
            if content == "amber":
                content = "yellow"
            if impl_m:
                implication = impl_m.group(1).strip()
            elif structured:
                implication = None                        # an attribute record, not an explanation
            else:
                implication = sent if _EFFECT.search(sent) else None
            objs.append({"type": t, "point": pt, "rank": int(rank_m.group(1)) if rank_m else None,
                         "location": fields["location"], "intention": (intention if intention != "other" else None),
                         "state": (beh if (t not in _CONTROL and beh != "other") else None),
                         "content": content, "implication": implication, "_from": "prose"})
        for t, pt in extra:
            objs.append({"type": t, "point": pt, "rank": None, "location": None, "intention": None,
                         "state": None, "content": None, "implication": None, "_from": "prose"})
    grounded = {o["type"] for o in objs if o["point"] is not None}
    if grounded:                                          # "the vehicles ahead" after a pointed list = re-mention
        objs = [o for o in objs if o["point"] is not None or (o["type"] not in grounded and o["type"] != "vehicle")]
    return objs


def _syn(v: Optional[str]) -> str:
    """Released-model wording -> the dataset's behaviour label (then canon_behaviour's own rules)."""
    s = (v or "").lower().strip()
    for a, b in _BEH_SYN:
        if a in s:
            return b
    return canon_behaviour(s)


def _resample5(run: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    n = len(run)
    if n == 5:
        return run
    if n == 20:
        return [run[i] for i in (3, 7, 11, 15, 19)]
    if n == 10:
        return [run[i] for i in (1, 3, 5, 7, 9)]
    if n == 6 and abs(run[0][0]) < 1e-6 and abs(run[0][1]) < 1e-6:
        return run[1:]
    out = []
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        f = t / 5.0 * (n - 1); i = min(int(f), n - 2); w = f - i
        out.append((run[i][0] * (1 - w) + run[i + 1][0] * w, run[i][1] * (1 - w) + run[i + 1][1] * w))
    return out


def extract_traj5(text: str) -> Optional[List[Tuple[float, float]]]:
    """A run of >= 5 ``[x, y]`` / ``(x, y)`` float pairs -> 5 waypoints at t = 1..5 s.

    Runs introduced by a trajectory keyword win; otherwise the LAST run (a restated history comes
    before the prediction and lies at x <= 0)."""
    answer, _ = strip_think(text)
    pairs = [(m.start(), m.end(), float(m.group(1)), float(m.group(2))) for m in _FLOAT_PAIR.finditer(answer)]
    if len(pairs) < 5:
        return None
    runs: List[Tuple[int, List[Tuple[float, float]]]] = []
    cur: List[Tuple[float, float]] = []; start = pairs[0][0]; last_end = None
    for pos, end, x, y in pairs:
        gap = answer[last_end:pos] if last_end is not None else ""
        if last_end is not None and (pos - last_end > 24 or re.search(r"[A-Za-z]{4,}", gap)):
            runs.append((start, cur)); cur = []; start = pos    # "- t = 2 s: " is tolerated, "Trajectory:" splits
        cur.append((x, y)); last_end = end
    runs.append((start, cur))
    good = [(s, r) for s, r in runs if len(r) >= 5 and max(abs(v) for xy in r for v in xy) <= 400]
    if not good:
        return None
    keyed = [(s, r) for s, r in good if _TRAJ_KW.search(answer[max(0, s - 80):s])]
    future = [(s, r) for s, r in (keyed or good) if max(x for x, _ in r) > 0 or all(abs(x) < 1e-6 for x, _ in r)]
    s, run = (future or keyed or good)[-1]
    return _resample5(run)


def _reason_of(text: str, answer: str, think: str) -> Optional[str]:
    for rx in (_REASON_RX, _REASON_HDR):
        m = rx.search(text)
        if m:
            return _MD.sub("", m.group(1)).strip()
    for src in (answer, think):
        for s in _sentences(src):
            if _CAUSAL.search(s) and len(s) > 25:
                return s
    return None


def _plan_of(text: str, answer: str) -> Optional[str]:
    for rx in (_PLAN_RX, _PLAN_HDR):
        m = rx.search(text)
        if m:
            return _MD.sub("", m.group(1)).strip()
    for s in _sentences(answer):
        m2 = _EGO_SHOULD.search(s)
        if m2:
            return m2.group(1).strip()
    return None


def extract(text: str, point_mode: str = "abs_pixel") -> Dict[str, Any]:
    """Strict ``parse_output`` first, then prose fallbacks for whatever is missing."""
    text = text or ""
    pred = parse_output(text)
    prov: Dict[str, str] = {}
    answer, think = strip_think(text)
    if not pred.get("objects") and pred.get("has_objects") is not False:   # a strict "no" is final
        objs = _tagged_objects(text, point_mode)
        src = "tagged"
        if not objs:
            objs = extract_objects(text, point_mode); src = "prose/json"
        if objs:
            pred["objects"] = objs; pred["has_objects"] = True; pred["n_objects"] = len(objs)
            prov["objects"] = src
        elif pred.get("has_objects") is None and answer.strip():
            pred["has_objects"] = False; prov["has_objects"] = "prose:none-found"
    if not pred.get("reason"):
        r = _reason_of(text, answer, think)
        if r:
            pred["reason"] = r; prov["reason"] = "prose" if r in answer else "think"
    if not pred.get("final_plan"):
        p = _plan_of(text, answer)
        if p:
            pred["final_plan"] = p; prov["final_plan"] = "prose"
    if not pred.get("traj"):
        tr = extract_traj5(text)
        if tr:
            pred["traj"] = tr; pred["traj_n_points"] = 5; pred["traj_closed"] = True; prov["traj"] = "prose"
    if not pred.get("context"):
        ctx = {k: _MD.sub("", m.group(1)).strip() for k, rx in _CTX_RX.items() for m in [rx.search(text)] if m}
        if ctx:
            pred["context"] = ctx; prov["context"] = "prose"
    if not pred.get("events"):
        m = _EVENTS_RX.search(text)
        if m:
            ev = _MD.sub("", m.group(1)).strip()
            pred["events"] = [] if ev.lower() in ("none", "no", "n/a", "-", "") else [e.strip() for e in re.split(r"[,;|]", ev) if e.strip()]
            prov["events"] = "prose"
    if not pred.get("motion"):
        m = _MOTION_RX.search(answer)
        if m:
            from training.core.parse_output import norm_motion_lon, norm_motion_lat
            pred["motion"] = {"lon": norm_motion_lon(m.group(1)), "lat": norm_motion_lat(m.group(2))}
            pred["motion_raw"] = {"lon": m.group(1), "lat": m.group(2)}; prov["motion"] = "prose"
    pred["_provenance"] = prov
    return pred
