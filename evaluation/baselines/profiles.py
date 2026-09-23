"""Capability profiles of the released models.

``True``  = native capability, scored with the standard metric
``False`` = the model cannot produce it -> the column is '—' (None), never 0
``'text'`` = only recoverable from free text -> scored through ``text_extract`` (types without
            geometry go through ``typeonly``); flagged in the metrics as ``provenance='text'``

Entries are read off the published model cards and the models' own inference scripts.
``unverified`` marks an entry inferred from documentation rather than from an actual run; the run
itself supersedes it (``rescore.py`` records what it really found in the output).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Union

Cap = Union[bool, str]                   # True | False | 'text'


@dataclass(frozen=True)
class Profile:
    name: str                            # adapter / model key
    display: str
    source: str                          # 'adapter' (Stage-2 prompt through an adapter) | 'native' (own script)
    points: Cap
    types: Cap
    behaviour: Cap                       # state / intention / content attributes
    implication: Cap
    reason: Cap
    plan: Cap
    trajectory: Cap
    point_mode: str = "abs_pixel"        # coordinate system of any point it emits
    note: str = ""
    unverified: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def has(self, cap: str) -> bool:
        return bool(getattr(self, cap))


PROFILES: Dict[str, Profile] = {p.name: p for p in (
    Profile("qwen25_7b", "Qwen2.5-VL-7B", "adapter",
            points=True, types=True, behaviour="text", implication="text", reason="text", plan="text",
            trajectory="text", point_mode="abs_pixel",
            note="general VLM with native grounding (bbox/points, JSON); no action head -> trajectory only "
                 "as prompted numbers; follows SYSTEM_S2 tags imperfectly -> tolerant extraction"),
    Profile("qwen3vl_8b", "Qwen3-VL-8B", "adapter",
            points=True, types=True, behaviour="text", implication="text", reason="text", plan="text",
            trajectory="text", point_mode="norm1000",
            note="as Qwen2.5-VL; points normalised 0-1000; may emit an empty <think></think> preamble"),
    Profile("cosmos_2b", "Cosmos-Reason2-2B", "adapter",
            points=True, types=True, behaviour="text", implication="text", reason=True, plan="text",
            trajectory="text", point_mode="norm1000",
            note="native <think>/<answer> CoT; 2D point/bbox localisation per model card; the numbers it "
                 "writes for a trajectory are unreliable and often unparseable"),
    Profile("cosmos_8b", "Cosmos-Reason2-8B", "adapter",
            points=True, types=True, behaviour="text", implication="text", reason=True, plan="text",
            trajectory="text", point_mode="norm1000", note="as Cosmos-2B"),
    Profile("alpamayo_r1_10b", "Alpamayo-R1-10B", "native",
            points=False, types="text", behaviour="text", implication=False, reason=True, plan=True,
            trajectory=True, point_mode="norm1000",
            note="native fields: cot (one causal sentence, e.g. 'Keep lane ... since the road ahead is clear') = "
                 "reason; the released weights emit no meta_action, so a plan comes from the cot text only if the "
                 "extractor finds one, else '—'; diffusion trajectory (64x3 @10 Hz) -> 20 @4 Hz. "
                 "No per-object points; object types only inside the cot text"),
    Profile("alpamayo_15_10b", "Alpamayo-1.5-10B", "native",
            points=False, types="text", behaviour="text", implication=False, reason=True, plan=True,
            trajectory=True, point_mode="norm1000", note="as Alpamayo-R1"),
    Profile("autovla_3b", "AutoVLA (3B)", "native",
            points=False, types=False, behaviour=False, implication=False, reason=False, plan=False,
            trajectory=True, point_mode="abs_pixel",
            note="released checkpoint emits the same constant <think> sentence on every frame plus action "
                 "tokens -> only the trajectory is scorable"),
    Profile("impromptu_7b", "Impromptu-VLA-7B", "native",
            points=False, types=False, behaviour=False, implication=False, reason=False, plan=False,
            trajectory=True, point_mode="abs_pixel",
            note="released driving checkpoint answers the nuScenes trajectory prompt with <PLANNING> [x,y] "
                 "waypoints over a horizon shorter than the 5 s asked for -> resampled linearly onto the 0.25 s "
                 "grid and HELD at the last point out to 5 s (a short horizon is penalised, not extrapolated); "
                 "front view = middle third of the panorama; no text fields"),
)}

ALIASES = {"qwen2.5-vl-7b": "qwen25_7b", "qwen3-vl-8b": "qwen3vl_8b", "cosmos-r2-2b": "cosmos_2b",
           "cosmos-r2-8b": "cosmos_8b", "alpamayo-r1-10b": "alpamayo_r1_10b", "alpamayo-1.5-10b": "alpamayo_15_10b",
           "autovla": "autovla_3b", "impromptu-vla-7b": "impromptu_7b"}


def get_profile(name: str) -> Profile:
    key = ALIASES.get(name, name)
    if key not in PROFILES:
        raise KeyError(f"no baseline profile for {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[key]
