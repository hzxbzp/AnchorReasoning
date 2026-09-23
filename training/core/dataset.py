#!/usr/bin/env python3
"""WaymoDataset -- training / evaluation samples for the two-stage SFT.

Per sample (``__getitem__``)::

    input_ids, labels, loss_weights, field_ids, traj_mask (bool), pixel_values,
    image_grid_thw, task

* prompt region: ``labels=-100``, ``loss_weights=0``, ``field_ids=-1``, ``traj_mask=False``;
* assistant region: segment tokens from ``target_builder.build_target`` with
  ``weight = W[field] * weight_mult`` (curriculum-locked fields -> 0, ``struct`` always on);
* ``traj_mask`` marks the trajectory block: every supervised token from the ``<traj>`` struct
  tag through ``</traj>`` and the closing ``<|im_end|>``. The trainer normalises the text
  and the trajectory sums separately. S1 / Attr-QA samples have an all-``False`` mask.

Task switching
    * S2 always builds the full sample: understanding chain + trajectory (``'s2'``).
    * S1: with probability ``attrqa_ratio`` -> ``'attrqa'`` (one object, minority-attribute
      objects x ``attrqa_minority_boost``), but only while the attribute fields are unlocked
      (curriculum stage b, or no curriculum at all) and the frame has objects.

Pools
    * S1: the full training index (dev scenes were removed at index build time).
    * S2: rows with ``partition in s2_partitions`` (p7..p21) and ``chain_complete == 1``
      (from the motion-meta cache). All caches are filtered with the index so they stay
      row-aligned.

Points
    ``_attach_points`` writes ``ann['_point'] = adapter.point_codec.encode(px, py, scale)``
    for every annotation: a SAM2-mask interior point (random pixel in training when
    ``point_mode_train == 'random_interior'``, the pixel nearest the centroid otherwise /
    for evaluation), falling back to the bbox centre.

Bad frames (unreadable json, short past/future states) are skipped by moving on to the next
index row, at most 8 times.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from training.core import target_builder as tb
from training.core.curriculum import CURRICULUM_STAGES
from training.core.labels import MAJORITY_ATTR
from training.core.normalize import norm_context, norm_type
from training.core.paths import FUT_STEPS, HIST_STEPS, S2_PARTITIONS, partition_of
from training.core.prompts import prompt_hash, render_prompt
from training.core.sampler import normalize_mean, stage2_sample_weights, weight_marginals

__all__ = ["WaymoDataset", "stage2_sample_weights", "rle_to_mask", "interior_point",
           "FIELD_ID", "ATTRQA_MAJORITY"]

log = logging.getLogger("dataset")

FIELD_ID: Dict[str, int] = {f: i for i, f in enumerate(tb.FIELDS)}

# Attribute values that count as the MAJORITY classes; the single source of truth is
# ``labels.MAJORITY_ATTR`` (also used by ``labels.minority_attr_frame`` for the sampling cache).
# An object whose intention or state is any other non-empty value is a "minority" object and
# gets the Attr-QA selection boost. ``ATTRQA_MAJORITY`` is kept as an alias.
ATTRQA_MAJORITY = MAJORITY_ATTR

DEFAULT_SAMPLING = {
    "intent_power": 1.0, "context_power": 0.3, "context_cap": 5.0, "event_boost": 2.0,
    "object_cap": 4, "object_boost_ge3": 2.0, "combined_cap": 8.0,
    "motion_mult": {"start": 2.5, "stop": 2.0, "stay": 1.2, "decel": 1.2}, "attr_mult": 1.5,
}
_BAD_FRAME_ERRORS = (json.JSONDecodeError, OSError, ValueError, KeyError)
MAX_SKIP = 8


# ---------------------------------------------------------------------------------------
# mask / point helpers
# ---------------------------------------------------------------------------------------
def rle_to_mask(size, counts):
    """Decode an UNCOMPRESSED COCO RLE (column-major) into a (h, w) uint8 mask.
    Compressed (string) counts are not supported -> ``None`` (caller falls back to bbox)."""
    if not size or counts is None or isinstance(counts, (str, bytes)):
        return None
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for c in counts:
        c = int(c)
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((h, w), order="F")  # COCO RLE is column-major


def interior_point(mask, sample, rng):
    """Return (x, y) inside ``mask``. ``sample=True`` -> random foreground pixel
    (augmentation); else the foreground pixel nearest the centroid (stable)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    if sample:
        i = rng.randrange(len(xs))
        return float(xs[i]), float(ys[i])
    cx, cy = xs.mean(), ys.mean()
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return float(xs[i]), float(ys[i])


def _bbox_match(pano_bbox, coco_anns):
    """Match a panorama_geo bbox {x,y,w,h} to the closest coco annotation (L1 on xywh)."""
    if not coco_anns:
        return None
    px, py = pano_bbox.get("x", 0), pano_bbox.get("y", 0)
    pw, ph = pano_bbox.get("w", 0), pano_bbox.get("h", 0)
    best, bd = None, 1e18
    for ca in coco_anns:
        b = ca.get("bbox") or [0, 0, 0, 0]
        d = abs(b[0] - px) + abs(b[1] - py) + abs(b[2] - pw) + abs(b[3] - ph)
        if d < bd:
            bd, best = d, ca
    return best


def _attr_val(at: dict, field: str):
    v = at.get(field)
    if isinstance(v, list):
        v = v[0] if v else None
    return v


def is_minority_obj(ann: dict) -> bool:
    """True if the object's intention or state is a non-majority value (Attr-QA boost)."""
    at = ann.get("attributes") or {}
    for f in ("intention", "state"):
        v = _attr_val(at, f)
        if v is None:
            continue
        if str(v).strip().lower() not in ATTRQA_MAJORITY:
            return True
    return False


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _cfg_get(d, key, default=None):
    """dict / OmegaConf-DictConfig tolerant ``get``."""
    if d is None:
        return default
    try:
        v = d.get(key, default)
    except Exception:
        v = getattr(d, key, default)
    return default if v is None else v


def _as_plain(d) -> dict:
    """Shallow-convert a dict-like (incl. OmegaConf DictConfig) into a plain dict."""
    if d is None:
        return {}
    try:
        from omegaconf import OmegaConf  # optional
        if OmegaConf.is_config(d):
            return OmegaConf.to_container(d, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    return dict(d)


# ---------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------
class WaymoDataset(Dataset):
    """See module docstring. ``cfg`` is the full yaml config dict; ``adapter`` a
    ``BackboneAdapter`` (providing ``image_inputs`` / ``point_codec`` / ``load_processor``);
    ``stage`` in ``{'s1', 's2'}``. Optional: ``processor`` (reuse an already-loaded one),
    ``training`` (False -> stable centroid points, no Attr-QA draw)."""

    def __init__(self, cfg: dict, adapter, stage: str, processor=None, training: bool = True):
        if stage not in ("s1", "s2"):
            raise ValueError(f"stage must be 's1'|'s2', got {stage!r}")
        self.cfg = cfg
        self.stage = stage
        self.adapter = adapter
        self.training = bool(training)
        self.codec = adapter.point_codec
        self.point_desc = getattr(self.codec, "desc", "")

        data = _as_plain(_cfg_get(cfg, "data", {}))
        self.data_cfg = data
        self.long_edge = int(data.get("long_edge", 2916))
        self.attrqa_ratio = float(data.get("attrqa_ratio", 0.15)) if stage == "s1" else 0.0
        self.attrqa_minority_boost = float(data.get("attrqa_minority_boost", 3.0))
        self.point_mode_train = str(data.get("point_mode_train", "random_interior"))
        self.s2_partitions = set(data.get("s2_partitions") or S2_PARTITIONS)

        self.field_names: List[str] = list(tb.FIELDS)
        self.field_id: Dict[str, int] = dict(FIELD_ID)

        self.sampling = dict(DEFAULT_SAMPLING)
        self.sampling["motion_mult"] = dict(DEFAULT_SAMPLING["motion_mult"])
        for k, v in _as_plain(_cfg_get(cfg, "sampling", {})).items():
            if k == "motion_mult" and v:
                self.sampling["motion_mult"].update(_as_plain(v))
            elif v is not None:
                self.sampling[k] = v

        # field weights (defaults from ``target_builder.W``) with an optional yaml override
        self.W: Dict[str, float] = dict(tb.W)
        fw = _as_plain(_cfg_get(_cfg_get(cfg, "loss", {}), "field_weights", {}))
        for k, v in fw.items():
            self.W[k] = float(v)

        # curriculum (S1 only; S2 has none -> all fields supervised)
        cur = _cfg_get(cfg, "curriculum", None)
        sched = list(_cfg_get(cur, "schedule", []) or []) if (cur and stage == "s1") else []
        for s in sched:
            if s not in CURRICULUM_STAGES:
                raise ValueError(f"unknown curriculum stage {s!r}; known: {sorted(CURRICULUM_STAGES)}")
        self.schedule: Optional[List[str]] = sched or None
        self._stage = mp.Value("i", 0) if self.schedule else None

        # tokenizer / processor
        self.processor = processor if processor is not None else adapter.load_processor()
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.imend = self.tok.convert_tokens_to_ids("<|im_end|>")
        if self.imend is None or self.imend < 0:
            raise ValueError("tokenizer has no <|im_end|> token")

        # index + caches (row aligned) + S2 pool filter
        self._load_index_and_caches(data)
        max_samples = data.get("max_samples")
        if max_samples:
            self._truncate(int(max_samples))
        self.n_bad = 0

    # ---- index / caches ---------------------------------------------------------------
    def _load_index_and_caches(self, data: dict):
        index_path = data.get("index")
        if not index_path or not os.path.isfile(index_path):
            raise FileNotFoundError(f"data.index not found: {index_path}")
        index: List[str] = _load_json(index_path)
        n = len(index)

        def _opt(key):
            p = data.get(key)
            if p and os.path.isfile(p):
                rows = _load_json(p)
                if len(rows) != n:
                    raise ValueError(f"cache {key}={p} has {len(rows)} rows, index has {n}")
                return rows
            if p:
                log.warning("cache %s not found at %s (will be derived on demand)", key, p)
            return None

        meta = _opt("meta")
        ctx = _opt("ctx_meta")
        motion = _opt("motion_meta")
        self.attr_freq: Optional[dict] = None
        p = data.get("attr_freq")
        if p and os.path.isfile(p):
            self.attr_freq = _load_json(p)
        self.plan_repair: Dict[str, dict] = {}
        p = data.get("plan_repair")
        if self.stage == "s2" and p and os.path.isfile(p):
            self.plan_repair = _load_json(p) or {}

        meta = [self._norm_meta_row(r) for r in meta] if meta is not None else None
        ctx = [self._norm_ctx_row(r) for r in ctx] if ctx is not None else None
        motion = [self._norm_motion_row(r, index[i]) for i, r in enumerate(motion)] \
            if motion is not None else None

        if self.stage == "s2":
            if motion is None:
                motion = self._derive_motion_meta(index)
            rows = [i for i in range(n)
                    if motion[i]["partition"] in self.s2_partitions
                    and int(motion[i]["chain_complete"]) == 1]
            log.info("S2 pool: %d / %d rows (partitions=%s, chain_complete=1)",
                     len(rows), n, sorted(self.s2_partitions))
            if not rows:
                raise ValueError("S2 pool is empty after filtering (check motion meta / partitions)")
        else:
            rows = list(range(n))

        self.rows = rows                       # original index row ids (for cache alignment)
        self.index = [index[i] for i in rows]
        self.meta = [meta[i] for i in rows] if meta is not None else None
        self.ctx_meta = [ctx[i] for i in rows] if ctx is not None else None
        self.motion_meta = [motion[i] for i in rows] if motion is not None else None

    def _truncate(self, k: int):
        self.rows = self.rows[:k]
        self.index = self.index[:k]
        for name in ("meta", "ctx_meta", "motion_meta"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v[:k])

    @staticmethod
    def _norm_meta_row(r) -> Tuple[str, int]:
        """index_meta row -> (intent, n_objects); accepts ``[intent, n]`` or a dict."""
        if isinstance(r, dict):
            it = r.get("intent_corrected") or r.get("intent") or "GO_STRAIGHT"
            k = r.get("n_objects", r.get("n_obj", 0)) or 0
        else:
            it, k = (r[0] or "GO_STRAIGHT"), (r[1] or 0)
        return str(it), int(k)

    @staticmethod
    def _norm_ctx_row(r) -> Tuple[str, str, int]:
        """index_ctx_meta row -> (weather, visibility, has_interesting_event)."""
        if isinstance(r, dict):
            return (str(r.get("weather") or "Unknown"), str(r.get("visibility") or "Unknown"),
                    int(r.get("has_interesting_event", r.get("interesting", 0)) or 0))
        return str(r[0] or "Unknown"), str(r[1] or "Unknown"), int(r[2] or 0)

    @staticmethod
    def _norm_motion_row(r, fdir: str) -> dict:
        """index_motion_meta row (dict) -> dict with the keys the dataset relies on."""
        if not isinstance(r, dict):
            raise ValueError(f"index_motion_meta rows must be dicts, got {type(r).__name__}")
        out = dict(r)
        out.setdefault("sample_class", "keep")
        out["has_minority_attr"] = int(out.get("has_minority_attr", 0) or 0)
        out["chain_complete"] = int(out.get("chain_complete", 0) or 0)
        part = out.get("partition")
        out["partition"] = str(part) if part else partition_of(fdir)
        return out

    # ---- on-demand meta derivation (only when a cache file is missing) ------------------
    def _derive_meta(self) -> List[Tuple[str, int]]:
        meta = []
        for fdir in self.index:
            it, k = "GO_STRAIGHT", 0
            try:
                fr = _load_json(os.path.join(fdir, "frame.json"))
                it = fr.get("intent_corrected", fr.get("intent")) or "GO_STRAIGHT"
                pg = _load_json(os.path.join(fdir, "panorama_geo.json"))
                k = len(pg.get("annotations") or [])
            except Exception:
                pass
            meta.append((str(it), int(k)))
        return meta

    def _derive_ctx_meta(self) -> List[Tuple[str, str, int]]:
        meta = []
        for fdir in self.index:
            w, v, inter = "Unknown", "Unknown", 0
            try:
                pg = _load_json(os.path.join(fdir, "panorama_geo.json"))
                c = norm_context(pg.get("context"))
                w, v = c.get("weather") or "Unknown", c.get("visibility") or "Unknown"
                types = [t for e in (pg.get("traffic_events") or []) for t in (e.get("types") or [])]
                inter = 1 if any(t != "Roadside parking" for t in types) else 0
            except Exception:
                pass
            meta.append((str(w), str(v), int(inter)))
        return meta

    @staticmethod
    def _derive_motion_meta(index: Sequence[str]) -> List[dict]:
        """Fallback when the motion-meta cache is absent: partition / chain_complete from
        the files, sample_class / has_minority_attr via ``labels`` when importable."""
        try:
            from training.core.labels import minority_attr_frame, sample_class
        except Exception:  # labels module not available -> neutral factors
            minority_attr_frame = sample_class = None  # type: ignore[assignment]
        rows = []
        for fdir in index:
            row = {"partition": partition_of(fdir), "chain_complete": 0, "sample_class": "keep",
                   "has_minority_attr": 0}
            try:
                fr = _load_json(os.path.join(fdir, "frame.json"))
                pg = _load_json(os.path.join(fdir, "panorama_geo.json"))
                anns = pg.get("annotations") or []
                row["chain_complete"] = int(bool(pg.get("reason")) and bool(pg.get("final_plan"))
                                            and all(a.get("driving_implication") for a in anns))
                if sample_class is not None:
                    row["sample_class"] = sample_class(fr)
                if minority_attr_frame is not None:
                    row["has_minority_attr"] = int(bool(minority_attr_frame(pg)))
            except Exception:
                pass
            rows.append(row)
        return rows

    # ---- curriculum API (used by CurriculumCallback / sampler / trainer) ----------------
    def current_unlock(self) -> Optional[set]:
        if self._stage is not None:
            return CURRICULUM_STAGES[self.schedule[self._stage.value]]
        return None

    def current_stage_name(self) -> str:
        return self.schedule[self._stage.value] if self._stage is not None else "static"

    def num_stages(self) -> int:
        return len(self.schedule) if self.schedule else 1

    def set_stage(self, idx: int):
        if self._stage is not None:
            self._stage.value = max(0, min(int(idx), len(self.schedule) - 1))

    def _attrqa_allowed(self) -> bool:
        """Attr-QA needs the attribute fields unlocked (stage b / no curriculum)."""
        unlock = self.current_unlock()
        return unlock is None or "type" in unlock

    # ---- sampling weights --------------------------------------------------------------
    def sample_weight_components(self) -> Dict[str, np.ndarray]:
        """Per-sample weight vectors, index-aligned with ``self.index``, each mean-normalised:
        ``intent`` (inverse freq ^ intent_power), ``ctx`` (max of weather / visibility
        inverse-freq ^ context_power x event_boost, clipped to [1, context_cap]), ``obj``
        (min(max(n,1), object_cap) x boost if n>=3), ``motion`` (motion_mult[sample_class]),
        ``attr`` (attr_mult if the frame has a minority-attribute object)."""
        s = self.sampling
        if self.meta is None:
            self.meta = self._derive_meta()
        if self.ctx_meta is None:
            self.ctx_meta = self._derive_ctx_meta()
        n = len(self.index)

        cnt = Counter(m[0] for m in self.meta)
        iw = {k: (1.0 / c) ** float(s["intent_power"]) for k, c in cnt.items()}
        w_intent = [iw[it] for it, _ in self.meta]
        cap_obj, boost3 = int(s["object_cap"]), float(s["object_boost_ge3"])
        w_obj = [min(max(k, 1), cap_obj) * (boost3 if k >= 3 else 1.0) for _, k in self.meta]

        wc = Counter(m[0] for m in self.ctx_meta)
        vc = Counter(m[1] for m in self.ctx_meta)
        wmax = max(wc.values()) if wc else 1
        vmax = max(vc.values()) if vc else 1
        cpow, ccap, eb = float(s["context_power"]), float(s["context_cap"]), float(s["event_boost"])
        w_ctx = []
        for wv, vv, interesting in self.ctx_meta:
            # weather & visibility are correlated -> take max (lift by the rarest single
            # attribute), not the product.
            f = max((wmax / wc[wv]) ** cpow, (vmax / vc[vv]) ** cpow)
            if interesting:
                f *= eb
            w_ctx.append(min(max(f, 1.0), ccap))

        mm = s["motion_mult"]
        am = float(s["attr_mult"])
        if self.motion_meta is not None:
            w_motion = [float(mm.get(r.get("sample_class", "keep"), 1.0)) for r in self.motion_meta]
            w_attr = [am if int(r.get("has_minority_attr", 0)) else 1.0 for r in self.motion_meta]
        else:
            w_motion = [1.0] * n
            w_attr = [1.0] * n
        return {
            "intent": normalize_mean(w_intent), "ctx": normalize_mean(w_ctx),
            "obj": normalize_mean(w_obj), "motion": normalize_mean(w_motion),
            "attr": normalize_mean(w_attr),
        }

    def stage2_weights(self, cap: Optional[float] = None) -> np.ndarray:
        """``stage2_sample_weights(self.sample_weight_components(), cap)`` (cap defaults to
        ``sampling.combined_cap``)."""
        if cap is None:
            cap = float(self.sampling.get("combined_cap", 8.0))
        return stage2_sample_weights(self.sample_weight_components(), cap=cap)

    def weight_marginals(self, weights: Sequence[float]) -> Dict[str, Dict[str, float]]:
        """Effective marginals (intent / sample_class / minority-attr / n_objects bucket)
        under per-sample ``weights`` -- for the start-up sanity print."""
        if self.meta is None:
            self.meta = self._derive_meta()
        keys = {"intent": [m[0] for m in self.meta],
                "n_objects": [("0" if k == 0 else "1-2" if k <= 2 else "3+") for _, k in self.meta]}
        if self.motion_meta is not None:
            keys["sample_class"] = [r.get("sample_class", "?") for r in self.motion_meta]
            keys["minority_attr"] = [int(r.get("has_minority_attr", 0)) for r in self.motion_meta]
        return weight_marginals(weights, keys)

    # ---- frame IO ----------------------------------------------------------------------
    @staticmethod
    def _load_frame(fdir: str) -> Tuple[dict, dict]:
        frame = _load_json(os.path.join(fdir, "frame.json"))
        pano = _load_json(os.path.join(fdir, "panorama_geo.json"))
        if not isinstance(frame, dict) or not isinstance(pano, dict):
            raise ValueError(f"unexpected json structure in {fdir}")
        return frame, pano

    @staticmethod
    def _validate_frame(frame: dict, need_future: bool):
        ps = frame.get("past_states") or {}
        n_past = min(len(ps.get(k) or []) for k in ("pos_x", "pos_y", "vel_x", "vel_y"))
        if n_past < HIST_STEPS:
            raise ValueError(f"past_states too short: {n_past} < {HIST_STEPS}")
        if need_future:
            fs = frame.get("future_states") or {}
            n_fut = min(len(fs.get(k) or []) for k in ("pos_x", "pos_y"))
            if n_fut < FUT_STEPS:
                raise ValueError(f"future_states too short: {n_fut} < {FUT_STEPS}")

    def _image(self, fdir: str) -> dict:
        img = self.adapter.image_inputs(os.path.join(fdir, "panorama_geo.png"), self.long_edge)
        for k in ("pixel_values", "image_grid_thw", "n_img_tokens", "scale"):
            if k not in img:
                raise ValueError(f"adapter.image_inputs missing key {k!r}")
        return img

    def _rng(self, fdir: str) -> random.Random:
        """Training: fresh RNG per call (per-worker seeded by the DataLoader -> true
        augmentation across epochs); evaluation: deterministic per frame."""
        if self.training:
            return random.Random(random.getrandbits(64))
        return random.Random(int(hashlib.md5(fdir.encode("utf-8")).hexdigest()[:8], 16))

    def _attach_points(self, pano: dict, fdir: str, scale: Tuple[float, float],
                       rng: random.Random, sample: Optional[bool] = None):
        """Write ``ann['_point']`` (adapter-encoded ints) for each annotation. ``sample``
        (random interior pixel) defaults to ``training and point_mode_train=='random_interior'``."""
        if sample is None:
            sample = self.training and self.point_mode_train == "random_interior"
        coco_anns: list = []
        scp = os.path.join(fdir, "panorama_geo_sam2_coco.json")
        if os.path.isfile(scp):
            try:
                coco_anns = (_load_json(scp) or {}).get("annotations", []) or []
            except Exception:
                coco_anns = []
        for a in (pano.get("annotations") or []):
            pt = None
            ca = _bbox_match(a.get("bbox") or {}, coco_anns)
            if ca is not None:
                seg = ca.get("segmentation") or {}
                m = rle_to_mask(seg.get("size"), seg.get("counts")) if isinstance(seg, dict) else None
                if m is not None:
                    pt = interior_point(m, sample=sample, rng=rng)
            if pt is None:  # fallback: bbox centre
                b = a.get("bbox") or {}
                if b:
                    pt = (float(b.get("x", 0)) + float(b.get("w", 0)) / 2,
                          float(b.get("y", 0)) + float(b.get("h", 0)) / 2)
            if pt is not None:
                x, y = self.codec.encode(pt[0], pt[1], scale)
                a["_point"] = (int(x), int(y))

    # ---- task selection ----------------------------------------------------------------
    def task_for(self, fdir: str, pano: Optional[dict] = None, rng: Optional[random.Random] = None) -> str:
        """Decide the task for ``fdir`` (see module docstring)."""
        if self.stage == "s2":
            return "s2"
        if (self.training and self.attrqa_ratio > 0 and self._attrqa_allowed()
                and pano is not None and (pano.get("annotations") or [])):
            r = rng.random() if rng is not None else random.random()
            if r < self.attrqa_ratio:
                return "attrqa"
        return "s1"

    def pick_attrqa_obj(self, anns: Sequence[dict], rng: random.Random) -> Optional[dict]:
        """Choose one annotation (must carry ``_point``), minority-attribute objects
        weighted x ``attrqa_minority_boost``."""
        cands = [a for a in anns if a.get("_point") is not None]
        if not cands:
            return None
        ws = [self.attrqa_minority_boost if is_minority_obj(a) else 1.0 for a in cands]
        return rng.choices(cands, weights=ws, k=1)[0]

    @staticmethod
    def attrqa_spec(ann: dict) -> dict:
        """``{'type': T, 'x': x, 'y': y}`` for ``prompts.user_text(..., attrqa=...)``."""
        at = ann.get("attributes") or {}
        x, y = ann["_point"]
        return {"type": norm_type(at.get("type", "Other")), "x": int(x), "y": int(y)}

    # ---- assembling one sample ---------------------------------------------------------
    def _assemble(self, prompt_ids: List[int], segments: Sequence[tuple]):
        """Tokenise target segments, apply field weights / curriculum, build traj_mask.

        ``traj_mask`` turns on at the ``<traj>`` struct tag and stays on through the end of the
        assistant turn."""
        unlock = self.current_unlock()
        fid_of = self.field_id
        a_ids: List[int] = []
        a_w: List[float] = []
        a_fid: List[int] = []
        a_tm: List[bool] = []
        in_traj = False
        for seg in segments:
            text, field = seg[0], seg[1]
            mult = float(seg[2]) if len(seg) > 2 and seg[2] is not None else 1.0
            if field not in fid_of:
                raise ValueError(f"unknown target field {field!r}")
            if not in_traj and (field == "traj" or (field == "struct" and "<traj>" in text)):
                in_traj = True
            w = self.W.get(field, 1.0) * mult
            if unlock is not None and field != "struct" and field not in unlock:
                w = 0.0   # curriculum: locked field (format tokens always supervised)
            ids = self.tok(text, add_special_tokens=False).input_ids
            a_ids += ids
            a_w += [w] * len(ids)
            a_fid += [fid_of[field]] * len(ids)
            a_tm += [in_traj] * len(ids)
        np_ = len(prompt_ids)
        sid = fid_of["struct"]
        input_ids = list(prompt_ids) + a_ids + [self.imend]
        labels = [-100] * np_ + a_ids + [self.imend]
        weights = [0.0] * np_ + a_w + [self.W["struct"]]
        field_ids = [-1] * np_ + a_fid + [sid]
        traj_mask = [False] * np_ + a_tm + [in_traj]
        return input_ids, labels, weights, field_ids, traj_mask

    def _build(self, fdir: str) -> Dict[str, Any]:
        frame, pano = self._load_frame(fdir)
        self._validate_frame(frame, need_future=(self.stage == "s2"))
        img = self._image(fdir)
        scale = tuple(float(v) for v in img["scale"])
        rng = self._rng(fdir)
        task = self.task_for(fdir, pano, rng)

        attrqa = None
        attrqa_obj = None
        overlay = None
        self._attach_points(pano, fdir, scale, rng)
        if task == "attrqa":
            attrqa_obj = self.pick_attrqa_obj(pano.get("annotations") or [], rng)
            if attrqa_obj is None:
                task = "s1"
            else:
                attrqa = self.attrqa_spec(attrqa_obj)
        if task == "s2":
            overlay = self.plan_repair.get(fdir) or self.plan_repair.get(os.path.basename(fdir.rstrip("/")))

        segments = tb.build_target(frame, pano, task, point_codec=self.codec, freq_table=self.attr_freq,
                                   overlay=overlay, attrqa_obj=attrqa_obj)
        prompt = render_prompt(task, frame, int(img["n_img_tokens"]), self.point_desc, attrqa=attrqa)
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        input_ids, labels, weights, field_ids, traj_mask = self._assemble(prompt_ids, segments)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float32),
            "field_ids": torch.tensor(field_ids, dtype=torch.long),
            "traj_mask": torch.tensor(traj_mask, dtype=torch.bool),
            "pixel_values": img["pixel_values"],
            "image_grid_thw": img["image_grid_thw"],
            "task": task,
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        n = len(self.index)
        if n == 0:
            raise IndexError("empty dataset")
        last_err: Optional[BaseException] = None
        for attempt in range(MAX_SKIP):           # skip rare corrupt / short frames
            fdir = self.index[(i + attempt) % n]
            try:
                return self._build(fdir)
            except _BAD_FRAME_ERRORS as e:
                last_err = e
                self.n_bad += 1
                if self.n_bad <= 5:
                    log.warning("skipping bad frame %s: %s: %s", fdir, type(e).__name__, e)
                continue
        raise RuntimeError(f"{MAX_SKIP} consecutive unreadable frames starting at index {i}: {last_err}")

    # ---- evaluation helpers ------------------------------------------------------------
    def prompt_inputs(self, fdir: str, task: str, attrqa: Optional[dict] = None) -> Dict[str, Any]:
        """Generation-mode inputs for ONE frame: prompt-only ids (1, L) + vision tensors +
        ``scale`` (for ``point_codec.decode``) + ``fdir``. ``task`` in s1|s2|attrqa;
        ``attrqa`` = ``{'type', 'x', 'y'}`` with ALREADY-encoded coordinates."""
        frame = _load_json(os.path.join(fdir, "frame.json"))
        img = self._image(fdir)
        prompt = self.render_prompt(task, frame, int(img["n_img_tokens"]), attrqa=attrqa)
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "pixel_values": img["pixel_values"],
            "image_grid_thw": img["image_grid_thw"],
            "scale": tuple(float(v) for v in img["scale"]),
            "fdir": fdir,
        }

    def render_prompt(self, task: str, frame: dict, n_img_tokens: int, attrqa: Optional[dict] = None) -> str:
        """Chat string up to ``"<|im_start|>assistant\\n"`` for this dataset's prompt format."""
        return render_prompt(task, frame, int(n_img_tokens), self.point_desc, attrqa=attrqa)

    def prompt_hash(self) -> str:
        """``prompts.prompt_hash(point_desc)`` -- what the ``prompt_hash.json`` written into the
        run / save directories should carry."""
        return prompt_hash(self.point_desc)

    def gt_pano(self, fdir: str, scale: Tuple[float, float]) -> dict:
        """``panorama_geo.json`` with stable (centroid) ``_point`` attached to every
        annotation -- the ground truth side for Attr-QA evaluation."""
        pano = _load_json(os.path.join(fdir, "panorama_geo.json"))
        self._attach_points(pano, fdir, tuple(scale), self._rng(fdir), sample=False)
        return pano

    def attrqa_candidates(self, fdir: str, scale: Tuple[float, float]) -> List[Tuple[dict, dict]]:
        """``[(attrqa_spec, annotation), ...]`` for every object of the frame (stable points)."""
        pano = self.gt_pano(fdir, scale)
        return [(self.attrqa_spec(a), a) for a in (pano.get("annotations") or []) if a.get("_point") is not None]

    def stats(self) -> dict:
        """Cheap summary for logging at start-up."""
        out = {"stage": self.stage, "n": len(self.index), "schedule": self.schedule,
               "attrqa_ratio": self.attrqa_ratio, "point_mode_train": self.point_mode_train,
               "long_edge": self.long_edge, "prompt_hash": self.prompt_hash()}
        if self.stage == "s2":
            out["n_plan_repair_hits"] = sum(1 for f in self.index if f in self.plan_repair)
        if self.motion_meta is not None:
            out["sample_class"] = dict(Counter(r.get("sample_class", "?") for r in self.motion_meta))
            out["minority_attr_frac"] = round(float(np.mean([int(r.get("has_minority_attr", 0))
                                                             for r in self.motion_meta])), 4)
        if self.meta is not None:
            out["intent"] = dict(Counter(m[0] for m in self.meta))
        return out
