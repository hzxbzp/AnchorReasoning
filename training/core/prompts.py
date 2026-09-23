"""Prompt construction for this code base.

Everything the model sees BEFORE its own answer is produced here, and only here:

* ``SYSTEM_S1`` / ``SYSTEM_S2`` -- the two system prompts.  Both carry the placeholder
  ``{POINT_COORD_DESC}`` which the backbone adapter fills through :func:`system_for`
  (``PointCodec.desc``), because the point coordinate convention differs per family
  (Qwen2.5-VL: processed-image pixels; Qwen3-VL: 0-1000 normalized).
  ``SYSTEM_S2`` is a *strict prefix extension* of ``SYSTEM_S1``.
* ``INSTRUCTIONS`` -- the last line of the user turn for the three tasks
  (``s1`` / ``s2`` / ``attrqa``).
* ``USER_TEMPLATE`` -- the user turn: History (16 waypoints) + Intent + instruction.
* ``CHAT_TEMPLATE`` -- the Qwen chat scaffold with the image pads expanded manually.
  Qwen2.5-VL and Qwen3-VL render *identical* strings for a
  ``[system(text), user(image, text)]`` conversation, and both processors expand a single
  ``<|image_pad|>`` into ``image_grid_thw.prod() // merge_size**2`` pads, so
  :func:`render_prompt` reproduces ``processor.apply_chat_template(..., add_generation_prompt=True)``
  followed by ``processor(text, images)`` token-for-token (checked at runtime by the
  adapters' ``chat_template_check``).
* :func:`prompt_hash` -- a short fingerprint of every prompt constant (per point
  description) written into checkpoints and checked by the evaluation scripts.

The module is pure Python (no torch / transformers imports) so it can be used by data
scripts, adapters, trainers and evaluators alike.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from training.core.paths import HIST_STEPS

__all__ = [
    "SYSTEM_S1", "SYSTEM_S2", "INSTRUCTIONS", "USER_TEMPLATE", "CHAT_TEMPLATE",
    "TASKS", "INTENTS", "DEFAULT_INTENT", "POINT_COORD_PLACEHOLDER", "HISTORY_LABEL",
    "system_for", "instruction_for", "history_text_from_frame", "intent_of", "user_text",
    "chat_messages", "render_prompt", "prompt_hash",
]

# --------------------------------------------------------------------------------------
# Constants. Editing any of them changes prompt_hash and therefore silently invalidates
# every checkpoint trained with the previous wording -- change them only deliberately.
# --------------------------------------------------------------------------------------
POINT_COORD_PLACEHOLDER = "{POINT_COORD_DESC}"

_S1_PARAGRAPH_1 = (
    'You are an autonomous-driving scene understanding model. Inputs: one panorama stitched '
    "from three forward cameras, the ego vehicle's past 4 s trajectory as 16 (x, y) waypoints "
    "in meters (ego frame, current position at the origin, x forward), and the ego's "
    'high-level intent. Output strictly in the given tag format and order: (1) the 5 context '
    'dimensions; (2) traffic events; (3) whether key objects affecting driving exist; (4) if '
    'so, list ALL of them from highest to lowest impact, and only AFTER the list report how '
    'many you listed (<n_objects>). For each object: FIRST its point (<point>, '
    '{POINT_COORD_DESC}) on the object, THEN its impact rank, THEN only the attribute fields '
    'valid for its type. Ground every attribute in what is visible in the image: location '
    '(which lane relative to the ego and which direction), intention and state (what the '
    'object is doing right now, e.g. stopping, decelerating, turning, crossing), content (for '
    'traffic controls). A scene usually has SEVERAL such objects -- do NOT stop after one; '
    'include every object that matters for the intended maneuver. If none, output '
    '<has_objects>no</has_objects> and stop.'
)
_S1_PARAGRAPH_2 = (
    'Use ONLY these labels where applicable -- weather: Sunny|Cloudy|Rainy|Fog|Fair|Unknown; '
    'daytime: Day|Night; visibility: Clear|Reduced|Limited|Poor; road: '
    'Mid-block|Intersection|Diverge|Merge|Roundabout; traffic_events: Roadside '
    'parking|Construction|Lane guiding|Lane closure|Partially occupied lane|Road '
    'closure|Accident|Traffic signal malfunction|Traffic jam; object type: '
    'Car|Truck|Bus|Motorcyclist|Construction vehicle|Pedestrian|Cyclist|Scooter '
    'rider|Emergency vehicle|School bus|Traffic light|Sign|Stop line|Cross '
    'walk|Bump|Temporary control|Object|Animal|Other; traffic-control content: '
    'red|green|yellow|stop|lane guiding. Describe location, intention and state in the '
    "dataset's short phrases."
)
#: Stage-1 system prompt (also used for Attr-QA samples). Contains ``{POINT_COORD_DESC}``.
SYSTEM_S1: str = _S1_PARAGRAPH_1 + "\n" + _S1_PARAGRAPH_2

# The Stage-2 prompt appends the reasoning chain paragraph to SYSTEM_S1 (note the leading space).
_S2_APPEND = " " + (
    "Before the scene description, output <ego_state>: the ego's CURRENT behavior inferred "
    'from its past trajectory -- lon= stopped Ts | accelerating v1->v2 m/s | decelerating '
    'v1->v2 m/s | cruising v m/s, and lat= the current lateral maneuver (LANE_KEEPING, '
    'LEFT_TURN, RIGHT_TURN, LEFT_LANE_CHANGE, RIGHT_LANE_CHANGE, LEFT_NUDGE, RIGHT_NUDGE, '
    'PULL_OUT, PULL_OVER, LANE_BORROWING). Each listed object also gets <implication>: one '
    "sentence on how it constrains or releases the ego's path. After the objects, continue "
    "the chain: <reason> (1-2 sentences: the DOMINANT factor that shapes the ego's motion "
    'plus at most one enabling factor, citing only objects you listed), <final_plan> (one '
    "action or two sequential actions, e.g. 'keep lane and cruise', 'stop and wait', "
    "'accelerate and turn left'), <motion> (the motion trend the plan implies and the "
    'trajectory must follow: lon= accelerate | decelerate | keep | stop; lat= straight | '
    'left_turn | right_turn), and finally <traj>: the future 5 s trajectory as 5 waypoints '
    '[x, y] in meters at t = 1, 2, 3, 4, 5 s in the ego frame.'
)
#: Stage-2 (full chain) system prompt = SYSTEM_S1 + chain paragraph (strict prefix extension).
SYSTEM_S2: str = SYSTEM_S1 + _S2_APPEND

#: Final line of the user turn, per task. ``attrqa`` is a format string with ``{T}``, ``{x}``, ``{y}``.
INSTRUCTIONS: Dict[str, str] = {
    "s1": "Describe the scene per the required format.",
    "s2": "Describe the scene, then reason and plan, then output the trajectory per the required format.",
    "attrqa": "Describe the object of type {T} located at <point>{x},{y}</point> per the required format.",
}

TASKS: Sequence[str] = ("s1", "s2", "attrqa")
INTENTS: Sequence[str] = ("GO_STRAIGHT", "GO_LEFT", "GO_RIGHT")
DEFAULT_INTENT = "GO_STRAIGHT"

HISTORY_LABEL = "History (past 4 s, 16 waypoints at 4 Hz, ego frame, meters): "
#: The user turn. ``{history}`` = 16 ``[x, y]`` points, ``{intent}`` = GO_STRAIGHT|GO_LEFT|GO_RIGHT.
USER_TEMPLATE: str = HISTORY_LABEL + "{history}\nIntent: {intent}\n{instruction}"

#: Qwen2-family chat scaffold up to (and including) the assistant header. ``{image_pads}`` is
#: ``<|image_pad|>`` repeated ``n_img_tokens`` times (what the processor expands one pad into).
CHAT_TEMPLATE: str = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|>{image_pads}<|vision_end|>{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_PAD = "<|image_pad|>"

_SYSTEM_BY_TASK = {"s1": SYSTEM_S1, "attrqa": SYSTEM_S1, "s2": SYSTEM_S2}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _check_task(task: str) -> str:
    if task not in _SYSTEM_BY_TASK:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    return task


def _fmt1(v: float) -> str:
    """Format a coordinate with 1 decimal; ``-0.0`` is normalised to ``0.0``."""
    r = round(float(v), 1)
    if r == 0.0:          # also catches -0.0
        r = 0.0
    return f"{r:.1f}"


def system_for(task: str, point_desc: str) -> str:
    """Return the system prompt for ``task`` with ``{POINT_COORD_DESC}`` substituted.

    ``task`` in ``s1 | s2 | attrqa`` (``attrqa`` uses ``SYSTEM_S1``). ``point_desc`` is the
    adapter's ``PointCodec.desc`` (e.g. ``"pixel coords of the panorama (processed image)"``).
    """
    sys_text = _SYSTEM_BY_TASK[_check_task(task)]
    if POINT_COORD_PLACEHOLDER in sys_text:
        if not point_desc:
            raise ValueError(f"point_desc is required for task {task!r}")
        sys_text = sys_text.replace(POINT_COORD_PLACEHOLDER, str(point_desc))
    return sys_text


def instruction_for(task: str, attrqa: Optional[Mapping[str, Any]] = None) -> str:
    """Return the instruction line for ``task``; ``attrqa`` = ``{'type': T, 'x': int, 'y': int}``.

    For ``attrqa`` the point is the GT object's (already adapter-encoded) integer coordinates;
    they are rendered as plain integers without spaces (``<point>1420,600</point>``), matching
    the ``<point>`` syntax the model emits in its own answers.
    """
    _check_task(task)
    if task != "attrqa":
        return INSTRUCTIONS[task]
    if not attrqa or "type" not in attrqa or "x" not in attrqa or "y" not in attrqa:
        raise ValueError("attrqa task needs attrqa={'type': T, 'x': int, 'y': int}")
    return INSTRUCTIONS["attrqa"].format(
        T=str(attrqa["type"]), x=int(round(float(attrqa["x"]))), y=int(round(float(attrqa["y"]))))


def history_text_from_frame(frame: Mapping[str, Any]) -> str:
    """``"[x, y], [x, y], ..., [0.0, 0.0]"`` -- the 16 ``past_states`` positions, 1 decimal.

    Same rule as ``core.labels.history_text``; kept local so the prompt module has no
    dependency on the label module. Uses the last ``HIST_STEPS`` (16) steps if more are
    present; raises ``ValueError`` when the frame carries no past positions.
    """
    ps = frame.get("past_states") or {}
    xs, ys = list(ps.get("pos_x") or []), list(ps.get("pos_y") or [])
    if not xs or len(xs) != len(ys):
        raise ValueError("frame.past_states.pos_x/pos_y missing or of unequal length")
    xs, ys = xs[-HIST_STEPS:], ys[-HIST_STEPS:]
    return ", ".join(f"[{_fmt1(x)}, {_fmt1(y)}]" for x, y in zip(xs, ys))


def intent_of(frame: Mapping[str, Any]) -> str:
    """``intent_corrected`` with fallback to ``intent`` and finally ``GO_STRAIGHT`` (as text)."""
    return frame.get("intent_corrected") or frame.get("intent") or DEFAULT_INTENT


def user_text(frame: Mapping[str, Any], task: str,
              attrqa: Optional[Mapping[str, Any]] = None) -> str:
    """Render the user turn: History line, Intent line, task instruction.

    ``frame`` is the parsed ``frame.json``; ``attrqa`` is only used for ``task='attrqa'``.
    """
    return USER_TEMPLATE.format(history=history_text_from_frame(frame), intent=intent_of(frame),
                                instruction=instruction_for(task, attrqa))


def chat_messages(task: str, frame: Mapping[str, Any], point_desc: str,
                  attrqa: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The conversation as ``apply_chat_template`` expects it (system text + user[image, text]).

    Mirrors the conversation the adapters' ``chat_template_check`` feeds to
    ``apply_chat_template``; :func:`render_prompt` is the manual, byte-exact rendering of that
    conversation with ``add_generation_prompt=True``.
    """
    return [
        {"role": "system", "content": system_for(task, point_desc)},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(frame, task, attrqa)}]},
    ]


def render_prompt(task: str, frame: Mapping[str, Any], n_img_tokens: int, point_desc: str,
                  attrqa: Optional[Mapping[str, Any]] = None) -> str:
    """Full chat string up to ``"<|im_start|>assistant\\n"`` with the image pads expanded.

    ``n_img_tokens`` = ``image_grid_thw.prod() // merge_size**2`` from the adapter's
    ``image_inputs`` (Qwen2.5-VL patch 14 / Qwen3-VL patch 16, merge 2). Tokenising the result with
    ``tokenizer(text, add_special_tokens=False)`` gives exactly the ``input_ids`` the processor
    produces for ``chat_messages`` + the image. ``n_img_tokens=1`` reproduces the *unexpanded*
    ``apply_chat_template`` string.
    """
    n = int(n_img_tokens)
    if n < 1:
        raise ValueError("n_img_tokens must be >= 1")
    return CHAT_TEMPLATE.format(system=system_for(task, point_desc), image_pads=IMAGE_PAD * n,
                                user=user_text(frame, task, attrqa))


def prompt_hash(point_desc: str) -> str:
    """``sha256(SYSTEM_S1 | SYSTEM_S2 | user template | INSTRUCTIONS)[:16]``.

    The system prompts are hashed *after* ``{POINT_COORD_DESC}`` substitution, so the hash is
    per backbone family; the chat scaffold is included as part of the user template. Written to
    ``prompt_hash.json`` next to checkpoints and verified by the evaluation scripts.
    """
    parts = [
        system_for("s1", point_desc),
        system_for("s2", point_desc),
        USER_TEMPLATE + "\x1f" + CHAT_TEMPLATE,
        json.dumps(INSTRUCTIONS, sort_keys=True, ensure_ascii=True),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
