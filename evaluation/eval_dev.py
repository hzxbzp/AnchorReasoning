#!/usr/bin/env python3
"""Generation-based evaluation core.

This module is the *shared library* of the ``evaluation/`` package plus the dev-set CLI:

* :class:`PromptBuilder` -- one frame -> prompt tensors through the backbone adapter
  (``adapter.image_inputs`` + ``core.prompts.render_prompt``); no dataset/index needed.
* :func:`generate_text` -- the frozen decoding protocol: greedy, ``max_new_tokens=1024``, the
  adapter's banned special tokens masked with ``bad_words_ids`` and the anti-repetition guard
  described below; optional *forced assistant prefix* (used by the causal-intervention
  experiments in ``ci_eval.py``).
* :func:`make_record` / :func:`score_records` -- parse the chain, PCHIP-upsample the 5 text
  waypoints to the official 20-point grid, derive GT labels, and score with
  ``core.metrics`` (understanding / trajectory / chain / composite).
* :func:`run_dev_eval` -- the function the training callback calls
  (``run_dev_eval(model, processor, adapter, frames, task, max_new_tokens, bad_ids) -> (metrics, records)``);
  the optional ``prefix_fn`` hook teacher-forces an assistant prefix per frame.
* :func:`paired_bootstrap_ci` -- per-frame paired bootstrap 95% CI for a comparison of two runs.
* CLI: ``python -m evaluation.eval_dev --ckpt <dir> --adapter <name> --stage s1|s2 --frames <json> --out <json>``.

``no_repeat_ngram_size`` is deliberately NOT passed to ``model.generate``: the stock processor
also scans the PROMPT, whose History line for a stopped ego repeats "[0.0, 0.0], " sixteen times,
so a stationary ``<traj>`` could never be emitted; five identical waypoints likewise repeat on a
short period, and legitimately repeated attribute structure
("</intention><state>cruising</state><implication>") occurs once per additional cruising object.
:class:`RepetitionGuard` therefore implements the same idea (break decoding loops) as a
generated-only n-gram ban (n = ``no_repeat_ngram``, an n-gram may occur at most ``ngram_max_occ``
= 6 times) plus a periodic-run guard (a verbatim block repeated 2x consecutively -- 3x for
period 2, 4x for period 1 -- may not be continued), both switched OFF once ``<traj>`` has been
emitted.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from training.core.paths import API_KEY_FILE, CLUSTER_JSON, DATA, FUT_STEPS, PANO_W, RFS_DIR
from training.core import prompts as P
from training.core import traj_codec as TC
from training.core import labels as L
from training.core.parse_output import parse_output
from training.core.metrics.understanding import evaluate_understanding
from training.core.metrics.objects import evaluate_objects, match_all
from training.core.metrics.behaviour import evaluate_behaviour
from training.core.metrics.judges import evaluate_reason
from training.core.metrics.traj import traj_metrics
from training.core.metrics.chain import chain_metrics
from training.core.metrics import composite as COMPOSITE

GEN_DEFAULTS = dict(max_new_tokens=1024, no_repeat_ngram=12, ngram_max_occ=6, period_max=256)
TERMINATORS = ("<|im_end|>", "<|endoftext|>")
V0_BINS: Sequence[Tuple[str, float, float]] = (
    ("v0<0.5", 0.0, 0.5), ("0.5-3", 0.5, 3.0), ("3-8", 3.0, 8.0), ("8-15", 8.0, 15.0), ("15+", 15.0, float("inf")))
#: ``prefix_fn`` return value that makes :func:`run_dev_eval` skip the frame.
SKIP_FRAME = object()


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Paired per-frame bootstrap of ``mean(a) - mean(b)``; every comparison gets a 95% CI.

    ``a``/``b`` are aligned per-frame values (same frames, same order); pairs with a ``None``/nan on
    either side are dropped. Returns ``dict(n, mean_a, mean_b, diff, ci_lo, ci_hi, p_diff_gt0)``
    (``p_diff_gt0`` = share of bootstrap replicates with a positive difference).
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None and not (math.isnan(float(x)) or math.isnan(float(y)))]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mean_a": None, "mean_b": None, "diff": None, "ci_lo": None, "ci_hi": None, "p_diff_gt0": None}
    arr = np.asarray(pairs, dtype=float)
    d = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boots = d[idx].mean(axis=1)
    return {"n": n, "mean_a": float(arr[:, 0].mean()), "mean_b": float(arr[:, 1].mean()), "diff": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 100 * alpha / 2)), "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
            "p_diff_gt0": float((boots > 0).mean())}


# ======================================================================================
# Frame discovery / loading
# ======================================================================================
def rfs_frame_dirs(root: str = RFS_DIR) -> List[str]:
    """All rater-feedback frame dirs ``<RFS_DIR>/<scene_id>-<fid>`` with frame.json + panorama_geo.json."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        b = os.path.basename(d)
        if not os.path.isdir(d) or b.startswith("._") or b.startswith("DONE_"):
            continue
        if os.path.isfile(os.path.join(d, "frame.json")) and os.path.isfile(os.path.join(d, "panorama_geo.json")):
            out.append(d)
    return out


def load_frames(spec: str) -> List[str]:
    """Resolve a ``--frames`` argument to a list of frame dirs.

    ``spec`` may be ``"rfs"`` (the rater-feedback frames), a directory (all frame dirs below it), or
    a JSON file holding either a list of frame dirs or a dict with a ``frames``/``fdirs`` list (the
    ``dev_eval_frames.json`` / ``dev_online_128.json`` written by
    ``data_preparation/make_dev_split.py``).
    """
    if spec in ("rfs", "val456", "rfs456"):
        return rfs_frame_dirs()
    if os.path.isdir(spec):
        out = []
        for dp, dns, fns in os.walk(spec):
            dns[:] = [d for d in dns if not d.startswith("._")]
            if "frame.json" in fns and "panorama_geo.json" in fns:
                out.append(dp)
        return sorted(out)
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"--frames {spec!r}: not a file, dir or 'rfs'")
    obj = json.load(open(spec))
    if isinstance(obj, dict):
        for k in ("frames", "fdirs", "dev_eval_frames", "online"):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
        else:
            raise ValueError(f"{spec}: dict without a frames/fdirs list")
    fr = []
    for it in obj:
        fr.append(it["fdir"] if isinstance(it, dict) else str(it))
    return fr


def load_cluster_map(path: str = CLUSTER_JSON) -> Dict[str, str]:
    """``{scene_id: scenario_cluster}`` from the official val cluster json (empty if missing)."""
    if not os.path.isfile(path):
        return {}
    raw = json.load(open(path))
    return {k: (v.get("scenario_cluster") if isinstance(v, dict) else v) for k, v in raw.items()}


def load_frame_bundle(fdir: str) -> Tuple[dict, dict, list]:
    """``(frame.json, panorama_geo.json, sam2 coco annotations)`` of one frame dir."""
    frame = json.load(open(os.path.join(fdir, "frame.json")))
    pano = json.load(open(os.path.join(fdir, "panorama_geo.json")))
    scp = os.path.join(fdir, "panorama_geo_sam2_coco.json")
    coco: list = []
    if os.path.isfile(scp):
        try:
            coco = (json.load(open(scp)) or {}).get("annotations", []) or []
        except Exception:
            coco = []
    return frame, pano, coco


def scene_frame_ids(frame: dict, fdir: str) -> Tuple[str, str]:
    """``(scene_id, frame_id)`` from frame.json, falling back to the ``<scene>-<fid>`` dir name."""
    sid, fid = frame.get("scene_id"), frame.get("frame_id")
    if sid is None or fid is None:
        base = os.path.basename(fdir.rstrip("/"))
        if base.startswith("DONE_"):
            base = base[5:]
        s, _, f = base.rpartition("-")
        sid, fid = sid or s, fid or f
    return str(sid), str(fid)


def gt_xy20(frame: dict) -> Optional[List[Tuple[float, float]]]:
    """GT future as 20 ``(x, y)`` points (``None`` if the frame has fewer than 20)."""
    fs = frame.get("future_states") or {}
    px, py = fs.get("pos_x") or [], fs.get("pos_y") or []
    if min(len(px), len(py)) < FUT_STEPS:
        return None
    return [(float(px[i]), float(py[i])) for i in range(FUT_STEPS)]


def ego_speed(frame: dict) -> float:
    """|v0| = current ego speed from ``past_states.vel_*[-1]`` (m/s)."""
    vx, vy = TC.ego_velocity(frame)
    return float(math.hypot(vx, vy))


def v0_bin(v0: float) -> str:
    for name, lo, hi in V0_BINS:
        if lo <= v0 < hi:
            return name
    return V0_BINS[-1][0]


# ======================================================================================
# SAM2 mask helpers (object points used by the causal-intervention prefixes)
# ======================================================================================
def rle_to_mask(size, counts) -> Optional[np.ndarray]:
    """Uncompressed COCO RLE -> HxW uint8 mask (column-major). Compressed RLE -> ``None``."""
    if not size or counts is None or isinstance(counts, (str, bytes)):
        return None
    h, w = size
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for c in counts:
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def bbox_match(pano_bbox: dict, coco_anns: list) -> Optional[dict]:
    """The coco annotation whose bbox is closest (L1) to a panorama_geo bbox ``{x,y,w,h}``."""
    if not coco_anns or not pano_bbox:
        return None
    px, py, pw, ph = (pano_bbox.get(k, 0) for k in ("x", "y", "w", "h"))
    best, bd = None, 1e18
    for ca in coco_anns:
        b = ca.get("bbox") or [0, 0, 0, 0]
        d = abs(b[0] - px) + abs(b[1] - py) + abs(b[2] - pw) + abs(b[3] - ph)
        if d < bd:
            bd, best = d, ca
    return best


def stable_interior_point(ann: dict, coco_anns: list) -> Optional[Tuple[float, float]]:
    """Panorama-pixel point on the object: mask pixel nearest the mask centroid, else bbox center."""
    ca = bbox_match(ann.get("bbox") or {}, coco_anns)
    if ca is not None:
        seg = ca.get("segmentation") or {}
        m = rle_to_mask(seg.get("size"), seg.get("counts")) if seg else None
        if m is not None:
            ys, xs = np.nonzero(m)
            if len(xs):
                cx, cy = xs.mean(), ys.mean()
                i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
                return float(xs[i]), float(ys[i])
    b = ann.get("bbox") or {}
    if b:
        return float(b.get("x", 0) + b.get("w", 0) / 2), float(b.get("y", 0) + b.get("h", 0) / 2)
    return None


def attach_points(pano: dict, coco_anns: list, point_codec, scale) -> dict:
    """Write ``_point`` (adapter-encoded ints) and ``_scale`` into ``pano`` for ``build_target``."""
    pano["_scale"] = tuple(scale)
    for a in pano.get("annotations") or []:
        pt = stable_interior_point(a, coco_anns)
        if pt is not None:
            x, y = point_codec.encode(pt[0], pt[1], scale)
            a["_point"] = (int(x), int(y))
    return pano


# ======================================================================================
# Checkpoint / adapter loading, prompt-hash gate
# ======================================================================================
def read_adapter_name(ckpt: str) -> Optional[str]:
    """``adapter_name.json`` written by ``train_stage.py`` next to the checkpoint (str or dict)."""
    p = os.path.join(ckpt, "adapter_name.json")
    if not os.path.isfile(p):
        return None
    obj = json.load(open(p))
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("adapter", "adapter_name", "name"):
            if isinstance(obj.get(k), str):
                return obj[k]
        for v in obj.values():
            if isinstance(v, str):
                return v
    return None


def read_prompt_hash(ckpt: str) -> Optional[str]:
    """The 16-hex prompt hash stored in ``<ckpt>/prompt_hash.json`` (str or dict), else ``None``."""
    p = os.path.join(ckpt, "prompt_hash.json")
    if not os.path.isfile(p):
        return None
    obj = json.load(open(p))
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("prompt_hash", "hash"):
            if isinstance(obj.get(k), str):
                return obj[k]
        for v in obj.values():
            if isinstance(v, str) and len(v) == 16:
                return v
    return None


def check_prompt_hash(ckpt: str, adapter, strict: bool = True) -> Optional[bool]:
    """Compare ``<ckpt>/prompt_hash.json`` with ``prompts.prompt_hash(adapter.point_codec.desc)``.

    Returns ``True`` on match, ``None`` when the checkpoint has no hash file (base model -> warning
    only). On mismatch the evaluation must not silently score a different prompt, so it exits with
    ``sys.exit(2)`` if ``strict``, else returns ``False`` with a warning.
    """
    want = P.prompt_hash(adapter.point_codec.desc)
    have = read_prompt_hash(ckpt)
    if have is None:
        print(f"[prompt_hash] WARNING: no prompt_hash.json in {ckpt}; current hash = {want}", flush=True)
        return None
    if have == want:
        print(f"[prompt_hash] OK {want}", flush=True)
        return True
    msg = f"[prompt_hash] MISMATCH: checkpoint={have} current={want} ({adapter.name}); prompts changed since training"
    if strict:
        print(msg + " -> exit", flush=True)
        sys.exit(2)
    print(msg + " (continuing: --no-hash-check)", flush=True)
    return False


def load_model_and_processor(ckpt: Optional[str], adapter_name: Optional[str] = None,
                             attn: str = "flash_attention_2", device: str = "cuda:0",
                             dtype=torch.bfloat16):
    """Load ``(model, processor, adapter)`` for evaluation.

    ``adapter_name`` falls back to ``<ckpt>/adapter_name.json``; ``ckpt=None`` loads the adapter's
    base weights (zero-shot baselines). ``extra_vocab_hook`` is applied (it is idempotent) and the
    model is put in eval mode with the KV cache enabled.

    A failing ``extra_vocab_hook`` is **not** swallowed: it propagates to the caller. For the base
    weights (``ckpt=None``) and for the Alpamayo adapters (whose hook verifies the ``<iN>`` /
    special-token ids against the checkpoint) a failure means tokenizer and model vocabularies
    disagree, and every generated id would be misread -- silently continuing would produce an
    evaluation of garbage. For a saved checkpoint the hook is a no-op, so it cannot fail there
    unless the checkpoint is inconsistent, which is equally worth an error.
    """
    from training.adapters import get_adapter
    name = adapter_name or (read_adapter_name(ckpt) if ckpt else None)
    if not name:
        raise ValueError("adapter name unknown: pass --adapter or provide <ckpt>/adapter_name.json")
    adapter = get_adapter(name)
    processor = adapter.load_processor()
    model = adapter.load_model(ckpt, dtype=dtype, attn=attn, device_map=device)
    adapter.extra_vocab_hook(processor.tokenizer, model)      # raises on a vocab/embedding mismatch
    model.eval()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = True
        except Exception:
            pass
    return model, processor, adapter


# ======================================================================================
# Prompt construction
# ======================================================================================
class PromptBuilder:
    """Frame -> generation inputs through the adapter (image) and ``core.prompts`` (text).

    ``build`` returns ``dict(input_ids[1,L], attention_mask, pixel_values, image_grid_thw, scale,
    n_img_tokens, prompt_text)``. ``image()`` is exposed so callers that run several prompts on the
    same frame pre-process the image only once.
    """

    def __init__(self, adapter, processor, long_edge: int = PANO_W):
        self.adapter = adapter
        self.processor = processor
        self.tok = processor.tokenizer
        self.long_edge = int(long_edge)

    def image(self, img_path: str) -> dict:
        return self.adapter.image_inputs(img_path, self.long_edge)

    def build(self, frame: dict, task: str, img: Optional[dict] = None, img_path: Optional[str] = None,
              attrqa: Optional[dict] = None) -> dict:
        if img is None:
            if img_path is None:
                raise ValueError("PromptBuilder.build: need img or img_path")
            img = self.image(img_path)
        text = P.render_prompt(task, frame, int(img["n_img_tokens"]), self.adapter.point_codec.desc, attrqa)
        ids = self.tok(text, add_special_tokens=False).input_ids
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                "pixel_values": img["pixel_values"], "image_grid_thw": img["image_grid_thw"],
                "scale": tuple(img["scale"]), "n_img_tokens": int(img["n_img_tokens"]), "prompt_text": text}

    def build_for_dir(self, fdir: str, task: str, attrqa: Optional[dict] = None) -> Tuple[dict, dict, dict, list]:
        """``(inputs, frame, pano, coco)`` for one frame dir."""
        frame, pano, coco = load_frame_bundle(fdir)
        img_path = os.path.join(fdir, "panorama_geo.png")
        return self.build(frame, task, img_path=img_path, attrqa=attrqa), frame, pano, coco


# ======================================================================================
# Decoding
# ======================================================================================
class RepetitionGuard(LogitsProcessor):
    """Anti-degeneration guard for structured chain decoding (see module docstring).

    Mechanisms (all on GENERATED tokens only, batch size 1, off after ``<traj>``):
      1. n-gram occurrence cap: a token completing an ``ngram_size``-gram already seen
         ``ngram_max_occ`` times is banned (``ngram_size<=1`` or ``ngram_max_occ<=0`` disables).
      2. periodic-run guard: if the tail of the generation is ``k`` verbatim copies of a block of
         period ``p`` (``k`` = 4 for p=1, 3 for p=2, 2 for p>=3; p <= ``period_max``), the token that
         would continue the run is banned.
    ``stats`` records how often each mechanism fired (surfaced as ``gen_len.guard_bans``).
    """

    def __init__(self, prompt_len: int, tokenizer, ngram_size: int = 12, ngram_max_occ: int = 6,
                 period_max: int = 256, traj_tag: str = "<traj>"):
        self.prompt_len = int(prompt_len)
        self.tok = tokenizer
        self.n = int(ngram_size or 0)
        self.max_occ = int(ngram_max_occ or 0)
        self.pmax = int(period_max or 0)
        self.traj_tag = traj_tag
        self._counts: Dict[tuple, Dict[int, int]] = {}
        self._seen = 0
        self.in_traj = False
        self.stats = {"ngram_bans": 0, "loop_bans": 0}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[0] != 1 or self.in_traj:
            return scores
        gen = input_ids[0, self.prompt_len:].tolist()
        n_gen = len(gen)
        if self.n > 1 and self.max_occ > 0:
            for i in range(max(self._seen, self.n - 1), n_gen):
                key = tuple(gen[i - self.n + 1:i])
                d = self._counts.setdefault(key, {})
                d[gen[i]] = d.get(gen[i], 0) + 1
        self._seen = n_gen
        if n_gen and self.traj_tag in self.tok.decode(gen[-8:], skip_special_tokens=False):
            self.in_traj = True
            return scores
        banned: set = set()
        if self.n > 1 and self.max_occ > 0 and n_gen >= self.n - 1:
            key = tuple(gen[n_gen - self.n + 1:n_gen])
            for t, c in self._counts.get(key, {}).items():
                if c >= self.max_occ:
                    banned.add(t)
            self.stats["ngram_bans"] += len(banned)
        if self.pmax > 0 and n_gen >= 2:
            g = np.asarray(gen, dtype=np.int64)
            before = len(banned)
            for p in range(1, min(self.pmax, n_gen // 2) + 1):
                k = 4 if p == 1 else (3 if p == 2 else 2)
                if n_gen < p * k:
                    continue
                tail = g[n_gen - p:]
                if all(np.array_equal(g[n_gen - j * p:n_gen - (j - 1) * p], tail) for j in range(2, k + 1)):
                    banned.add(int(g[n_gen - p]))
            self.stats["loop_bans"] += len(banned) - before
        if banned:
            scores[0, list(banned)] = -float("inf")
        return scores


def eos_ids(tokenizer) -> List[int]:
    """``<|im_end|>`` plus the tokenizer's eos id (deduplicated)."""
    ids = []
    for t in ("<|im_end|>",):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.append(i)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def strip_terminators(text: str) -> str:
    """Remove trailing chat terminators (and anything after the first one)."""
    cut = len(text)
    for t in TERMINATORS:
        i = text.find(t)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


@torch.no_grad()
def generate_text(model, tokenizer, inputs: dict, max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"],
                  bad_ids: Optional[Sequence[int]] = None, no_repeat_ngram: int = GEN_DEFAULTS["no_repeat_ngram"],
                  ngram_max_occ: int = GEN_DEFAULTS["ngram_max_occ"], period_max: int = GEN_DEFAULTS["period_max"],
                  forced_prefix: Optional[str] = None) -> dict:
    """Greedy-decode one prompt.

    Returns ``dict(text, generated, n_new_tokens, hit_cap, guard, prompt_len)``, where ``text`` =
    ``forced_prefix`` (if any) + generated text with terminators stripped. A forced prefix is
    appended to the prompt as assistant tokens (teacher-forced), so the returned ``text`` is a
    complete assistant answer that ``parse_output`` can read. ``hit_cap`` is ``True`` when no
    terminator was produced within ``max_new_tokens`` (possible truncation).
    """
    device = next(model.parameters()).device
    ids = inputs["input_ids"]
    if forced_prefix:
        pre = tokenizer(forced_prefix, add_special_tokens=False).input_ids
        ids = torch.cat([ids, torch.tensor([pre], dtype=torch.long)], dim=1)
    prompt_len = int(ids.shape[1])
    kw = {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}
    pv = inputs.get("pixel_values")
    if pv is not None:
        kw["pixel_values"] = pv.to(device=device, dtype=next(model.parameters()).dtype)
        kw["image_grid_thw"] = inputs["image_grid_thw"].to(device)
    guard = RepetitionGuard(prompt_len, tokenizer, ngram_size=no_repeat_ngram, ngram_max_occ=ngram_max_occ,
                            period_max=period_max)
    if forced_prefix and "<traj>" in forced_prefix and "</traj>" not in forced_prefix:
        guard.in_traj = True      # continuation IS the trajectory: a stationary "[0.0, 0.0], ..." must be allowed
    eos = eos_ids(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    bw = [[int(i)] for i in (bad_ids or []) if int(i) >= 0] or None
    out = model.generate(**kw, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1,
                         bad_words_ids=bw, logits_processor=LogitsProcessorList([guard]),
                         eos_token_id=eos, pad_token_id=pad, use_cache=True)
    gen = out[0, prompt_len:].tolist()
    ended = bool(gen) and gen[-1] in eos
    raw = tokenizer.decode(gen, skip_special_tokens=False)
    text = strip_terminators(raw)
    return {"text": (forced_prefix or "") + text, "generated": text, "n_new_tokens": len(gen),
            "hit_cap": not ended, "guard": dict(guard.stats), "prompt_len": prompt_len}


# ======================================================================================
# Records
# ======================================================================================
def _safe(fn: Callable, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def synthetic_frame(frame: dict, xy20) -> dict:
    """A frame whose ``future_states`` is a (20,2) trajectory (for deriving labels of a prediction)."""
    xy = np.asarray(xy20, dtype=float).reshape(-1, 2)
    return {"past_states": frame.get("past_states"), "intent": frame.get("intent"),
            "intent_corrected": frame.get("intent_corrected"), "ego_behavior": frame.get("ego_behavior"),
            "future_states": {"pos_x": xy[:, 0].tolist(), "pos_y": xy[:, 1].tolist(), "pos_z": [0.0] * len(xy)}}


def motion_of_xy20(frame: dict, xy20, intent: Optional[str] = None) -> Optional[dict]:
    """Trend-class ``{'lon','lat','text'}`` derived from a 20-point trajectory."""
    if xy20 is None:
        return None
    return _safe(L.motion_label, synthetic_frame(frame, xy20), intent or P.intent_of(frame))


def sample_class_of_xy20(frame: dict, xy20) -> Optional[str]:
    """Fine sample class (stay/start/stop/decel/accel/keep) of a predicted trajectory."""
    if xy20 is None:
        return None
    return _safe(L.sample_class, synthetic_frame(frame, xy20))


def traj_from_text(text: Optional[str], frame: dict) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """``(traj5, xy20)`` -- parsed 5 waypoints and their PCHIP upsampling (``None, None`` if invalid)."""
    # A prediction exists only when the answer carries a <traj> segment (this is what pred_rate
    # counts); the codec itself pairs bare numbers only inside that segment, so <point>/<rank>
    # digits of a tag-less answer can never be read as a trajectory.
    pts = TC.parse_traj_text(text) if text and "<traj>" in text else None
    if not pts:
        return None, None
    try:
        return pts, TC.upsample_pchip(pts, TC.hist_xy_from_frame(frame))
    except Exception:
        return pts, None


def make_record(fdir: str, frame: dict, pano: dict, coco: list, scale, point_codec, text: str, task: str = "s2",
                n_new_tokens: int = 0, hit_cap: bool = False, cluster: Optional[str] = None,
                pred_xy_override=None, max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"],
                prefix: Optional[str] = None) -> dict:
    """Assemble one evaluation record (the unit consumed by every ``core.metrics`` function).

    Keys: ``pred`` (parsed chain), ``pano``/``coco_anns``/``scale``/``point_codec`` (understanding
    metrics), ``frame``, ``gt20``, ``v0``, ``intent``, ``sample_class``/``motion_gt``/``ego_state_gt``
    (rule labels), ``traj5``, ``pred_xy`` (20x2 list or None), ``pred_motion``/``pred_sample_class``
    (derived from ``pred_xy``), ``ade1/ade3/ade5/fde5`` (nan if no prediction), ``text``,
    ``n_new_tokens``/``hit_cap``, ``task``, ``cluster``, ``scene_id``/``frame_id``/``fdir``.
    ``pred_xy_override`` (20x2) bypasses text parsing, for predictions that already come as
    coordinates instead of text. ``prefix`` is the teacher-forced assistant prefix, kept for the rows.
    The record also carries the key aliases read by ``core.metrics.chain`` (``pred20``,
    ``gt_ego_state``/``gt_motion``/``gt_sample_class``, ``n_tokens``/``max_new_tokens``/``finish_reason``).
    """
    pred = parse_output(text or "")
    traj5, xy20 = traj_from_text(text, frame) if pred_xy_override is None else (pred.get("traj"), None)
    if pred_xy_override is not None:
        xy20 = np.asarray(pred_xy_override, dtype=float).reshape(-1, 2)
        if xy20.shape[0] == 0:
            xy20 = None
    intent = P.intent_of(frame)
    gt20 = gt_xy20(frame)
    v0 = ego_speed(frame)
    sid, fid = scene_frame_ids(frame, fdir)
    rec = {
        "fdir": fdir, "scene_id": sid, "frame_id": fid, "task": task, "text": text, "pred": pred,
        "pano": pano, "coco_anns": coco, "scale": tuple(scale), "point_codec": point_codec, "frame": frame,
        "gt20": gt20, "v0": v0, "v0_bin": v0_bin(v0), "intent": intent, "cluster": cluster,
        "sample_class": _safe(L.sample_class, frame), "motion_gt": _safe(L.motion_label, frame, intent),
        "ego_state_gt": _safe(L.ego_state_label, frame),
        "traj5": traj5, "pred_xy": xy20.tolist() if xy20 is not None else None,
        "pred_motion": motion_of_xy20(frame, xy20, intent), "pred_sample_class": sample_class_of_xy20(frame, xy20),
        "n_new_tokens": int(n_new_tokens), "hit_cap": bool(hit_cap), "max_new_tokens": int(max_new_tokens),
        "prefix": prefix,
    }
    # aliases consumed by core.metrics.chain (record format of that module)
    rec["pred20"] = rec["pred_xy"]
    rec["gt_ego_state"], rec["gt_motion"], rec["gt_sample_class"] = rec["ego_state_gt"], rec["motion_gt"], rec["sample_class"]
    rec["n_tokens"] = rec["n_new_tokens"]
    rec["finish_reason"] = "length" if hit_cap else "stop"
    if xy20 is not None and gt20 is not None:
        a1, a3, a5, f5 = TC.traj_ade_fde(xy20, gt20)
    else:
        a1 = a3 = a5 = f5 = float("nan")
    rec.update({"ade1": a1, "ade3": a3, "ade5": a5, "fde5": f5})
    return rec


def records_to_rows(records: Sequence[dict]) -> List[dict]:
    """Compact, JSON-serialisable per-frame rows for ``chains.json``."""
    rows = []
    for r in records:
        p = r["pred"]
        rows.append({
            "scene_id": r["scene_id"], "frame_id": r["frame_id"], "fdir": r["fdir"], "task": r.get("task"),
            "n_objects": p.get("n_objects"), "emitted": len(p.get("objects") or []),
            "ego_state": p.get("ego_state"), "reason": p.get("reason"), "final_plan": p.get("final_plan"),
            "motion": p.get("motion"), "traj": r.get("traj5"), "has_traj": r.get("pred_xy") is not None,
            "pred_xy": r.get("pred_xy"), "pred_motion": r.get("pred_motion"),
            "motion_gt": r.get("motion_gt"), "sample_class": r.get("sample_class"), "v0": r.get("v0"),
            "ade5": None if math.isnan(r.get("ade5", float("nan"))) else r["ade5"],
            "n_new_tokens": r.get("n_new_tokens"), "hit_cap": r.get("hit_cap"), "prefix": r.get("prefix"),
            "prompt_len": r.get("prompt_len"),
            "text": r["text"],
        })
    return rows


def dump_rfs_preds(records: Sequence[dict], path: str, missing: str = "zeros") -> str:
    """Write ``[{scene_id, frame_id, pred_xy(20x[x,y])}]`` -- the input format of ``run_rfs.py``.

    Frames without a valid trajectory get ``[[0,0]]*20`` (``missing='zeros'``, the stationary
    fallback the official prep also applies) or ``null`` (``missing='null'``).
    """
    rows = []
    for r in records:
        xy = r.get("pred_xy")
        if xy is None:
            xy = [[0.0, 0.0]] * FUT_STEPS if missing == "zeros" else None
        rows.append({"scene_id": r["scene_id"], "frame_id": r["frame_id"], "pred_xy": xy})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(rows, open(path, "w"))
    return path


# ======================================================================================
# Scoring
# ======================================================================================
def _nanmean(xs: Iterable[float]) -> Optional[float]:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else None


def _rate(flags: Sequence[bool]) -> Optional[float]:
    return float(np.mean([bool(f) for f in flags])) if len(flags) else None


def traj_extra(records: Sequence[dict]) -> dict:
    """Trajectory summaries computed here (independent of ``metrics/traj.py`` key names).

    ``traj_score`` = mean(exp(-ADE5/2)) with missing predictions scored 0;
    ``start_frame_acc`` = share of GT start frames whose predicted trajectory is a start
    (derived sample class); ``start_pred_static`` = share of them predicted stationary instead;
    ``static_frame_acc`` for GT stay frames; ``pred_rate``; per-sample-class / v0-bin ADE5.
    """
    n = len(records)
    ade5 = [r["ade5"] for r in records]
    score = [math.exp(-r["ade5"] / 2.0) if not math.isnan(r["ade5"]) else 0.0 for r in records]
    start = [r for r in records if r.get("sample_class") == "start"]
    stay = [r for r in records if r.get("sample_class") == "stay"]
    by_class: Dict[str, dict] = {}
    for r in records:
        c = r.get("sample_class") or "unknown"
        by_class.setdefault(c, {"n": 0, "ade5": [], "pred_rate": []})
        by_class[c]["n"] += 1
        by_class[c]["ade5"].append(r["ade5"])
        by_class[c]["pred_rate"].append(r.get("pred_xy") is not None)
    by_v0: Dict[str, dict] = {}
    for r in records:
        b = r.get("v0_bin") or "?"
        by_v0.setdefault(b, {"n": 0, "ade5": []})
        by_v0[b]["n"] += 1
        by_v0[b]["ade5"].append(r["ade5"])
    return {
        "n": n, "pred_rate": _rate([r.get("pred_xy") is not None for r in records]),
        "ade1": _nanmean(r["ade1"] for r in records), "ade3": _nanmean(r["ade3"] for r in records),
        "ade5": _nanmean(ade5), "fde5": _nanmean(r["fde5"] for r in records),
        "traj_score": float(np.mean(score)) if n else None,
        "n_start": len(start),
        "start_frame_acc": _rate([r.get("pred_sample_class") == "start" for r in start]) if start else None,
        "start_pred_static": _rate([r.get("pred_sample_class") in ("stay", None) for r in start]) if start else None,
        "n_stay": len(stay),
        "static_frame_acc": _rate([r.get("pred_sample_class") == "stay" for r in stay]) if stay else None,
        "by_sample_class": {c: {"n": d["n"], "ade5": _nanmean(d["ade5"]), "pred_rate": _rate(d["pred_rate"])}
                            for c, d in sorted(by_class.items())},
        "by_v0_bin": {b: {"n": d["n"], "ade5": _nanmean(d["ade5"])} for b, d in by_v0.items()},
    }


def chain_extra(records: Sequence[dict]) -> dict:
    """Chain summaries computed here (independent of ``metrics/chain.py`` key names).

    Missing rates per chain field, ego_state lon-class accuracy vs the rule label, motion
    accuracy vs the GT trend label, motion-vs-trajectory consistency (predicted ``<motion>`` lon
    vs the class derived from the predicted trajectory) and plan-vs-motion consistency via
    ``labels.plan_consistent`` (None = unmatched/unspecified, not counted).
    """
    n = len(records)
    miss = {k: [] for k in ("ego_state", "reason", "final_plan", "motion", "traj", "implication")}
    ego_ok, mot_ok, mt_ok, pm_ok = [], [], [], []
    for r in records:
        p = r["pred"]
        miss["ego_state"].append(p.get("ego_state") is None)
        miss["reason"].append(not p.get("reason"))
        miss["final_plan"].append(not p.get("final_plan"))
        miss["motion"].append(p.get("motion") is None)
        miss["traj"].append(r.get("pred_xy") is None)
        objs = p.get("objects") or []
        miss["implication"].append(bool(objs) and any(not o.get("implication") for o in objs))
        es, gt_es = p.get("ego_state"), r.get("ego_state_gt")
        if es and gt_es:
            lon = str(es.get("lon") or "").split()[0] if es.get("lon") else ""
            ego_ok.append(lon == gt_es.get("lon_class"))
        pm, gm = p.get("motion"), r.get("motion_gt")
        if pm and gm:
            mot_ok.append(pm.get("lon") == gm.get("lon"))
        dm = r.get("pred_motion")
        if pm and dm:
            mt_ok.append(pm.get("lon") == dm.get("lon"))
        if p.get("final_plan") and dm:
            # the tolerant matrix needs v_end / start_t of the *predicted* trajectory, as in metrics/chain.py
            pxy = r.get("pred_xy")
            if pxy is not None and len(pxy) >= 2:
                _pts = [(0.0, 0.0)] + [tuple(q) for q in pxy]
                _sp = [((_pts[i + 1][0] - _pts[i][0]) ** 2 + (_pts[i + 1][1] - _pts[i][1]) ** 2) ** 0.5 / 0.25
                       for i in range(len(_pts) - 1)]
                _ve, _st = L.v_end_from_speeds(_sp), L.start_time_from_speeds(_sp)
            else:
                _ve, _st = None, None
            c = _safe(L.plan_consistent, p["final_plan"], dm, r.get("v0", 0.0), v_end=_ve, start_t=_st)
            if c is not None:
                pm_ok.append(bool(c))
    return {
        "n": n, "missing": {k: _rate(v) for k, v in miss.items()},
        "ego_state_lon_acc": _rate(ego_ok) if ego_ok else None, "n_ego_state_scored": len(ego_ok),
        "motion_acc": _rate(mot_ok) if mot_ok else None,
        "motion_traj_consistency": _rate(mt_ok) if mt_ok else None,
        "plan_traj_consistency": _rate(pm_ok) if pm_ok else None, "n_plan_scored": len(pm_ok),
        "hit_cap_rate": _rate([r.get("hit_cap") for r in records]),
    }


def score_records(records: Sequence[dict], task: str = "s2", impl_judge=None,
                  max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"], by_cluster: bool = True,
                  reason_judge=None) -> dict:
    """Score a list of records -> metrics dict.

    Keys: ``n_frames``, ``understanding`` (s1/s2), ``objects`` + ``behaviour`` (s1/s2),
    ``reason`` (s2), ``trajectory`` (s2: ``metrics.traj``), ``traj_extra``,
    ``chain`` (s2: ``metrics.chain``, incl. the ``plan_tax_*`` closed-set plan metric),
    ``chain_extra``, ``gen_len``, ``composite`` (the proxy score computed by
    ``core.metrics.composite``), ``by_cluster`` (optional).

    ``impl_judge`` / ``reason_judge``: the LLM judges in ``core.metrics.judges``; without them
    implication falls back to the token-overlap proxy and reason reports only its missing rate.
    """
    m: Dict[str, Any] = {"n_frames": len(records), "task": task}
    if task in ("s1", "s2", "attrqa") and records:
        m["understanding"] = evaluate_understanding(list(records), impl_judge=impl_judge)
        # object and behaviour metrics share ONE assignment pass (masks decoded once per frame)
        try:
            pre = match_all(list(records))
            m["objects"] = evaluate_objects(list(records), pre=pre)
            m["behaviour"] = evaluate_behaviour(list(records), pre=pre)
        except Exception as e:
            m["objects"] = m["behaviour"] = {"error": repr(e)}
    if task == "s2" and records:
        m["reason"] = evaluate_reason(list(records), judge=reason_judge)
        preds = [r.get("pred_xy") for r in records]
        gts = [r.get("gt20") for r in records]
        v0s = [r.get("v0") for r in records]
        try:
            m["trajectory"] = traj_metrics(preds, gts, v0s)
        except Exception as e:
            m["trajectory"] = {"error": repr(e)}
        m["traj"] = m["trajectory"]          # alias: metrics.composite looks the traj terms up under 'traj'
        m["traj_extra"] = traj_extra(records)
    if task == "s2" and records:
        try:
            m["chain"] = chain_metrics(list(records))
        except Exception as e:
            m["chain"] = {"error": repr(e)}
        m["chain_extra"] = chain_extra(records)
    glens = [r.get("n_new_tokens", 0) for r in records]
    guard = {"ngram_bans": 0, "loop_bans": 0}
    for r in records:
        for k in guard:
            guard[k] += int((r.get("guard") or {}).get(k, 0))
    m["gen_len"] = {"cap": int(max_new_tokens), "max": int(max(glens)) if glens else 0,
                    "p99": int(np.percentile(glens, 99)) if glens else 0,
                    "mean": float(np.mean(glens)) if glens else 0.0,
                    "hit_cap": int(sum(bool(r.get("hit_cap")) for r in records)), "guard_bans": guard}
    m["composite"] = compute_composite(m, task)
    if by_cluster and records and any(r.get("cluster") for r in records):
        buckets: Dict[str, list] = {}
        for r in records:
            buckets.setdefault(r.get("cluster") or "Unknown", []).append(r)
        m["by_cluster"] = {}
        for c, rs in sorted(buckets.items()):
            d = {"n": len(rs)}
            if task in ("s1", "s2"):
                try:
                    d["understanding_composite"] = evaluate_understanding(rs).get("composite")
                except Exception:
                    d["understanding_composite"] = None
            if task == "s2":
                d["ade5"] = _nanmean(r["ade5"] for r in rs)
            m["by_cluster"][c] = d
    return m


def compute_composite(m: dict, task: str) -> Optional[float]:
    """The proxy composite of a ``score_records`` dict -- one implementation only.

    s1 / attrqa -> ``core.metrics.composite.s1_composite(m['understanding'])``;
    s2 -> ``s2_composite(m)`` (reads the terms under ``understanding`` / ``traj`` / ``chain``).
    ``None`` when no term is available (empty subset).
    """
    if task in ("s1", "attrqa"):
        v = COMPOSITE.s1_composite(m.get("understanding") or {})
    elif task == "s2":
        v = COMPOSITE.s2_composite(m)
    else:
        v = None
    return None if v is None else float(v)


# ======================================================================================
# Main loop
# ======================================================================================
@torch.no_grad()
def run_dev_eval(model, processor, adapter, frames: Sequence[str], task: str = "s2",
                 max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"], bad_ids: Optional[Sequence[int]] = None,
                 long_edge: int = PANO_W, no_repeat_ngram: int = GEN_DEFAULTS["no_repeat_ngram"],
                 impl_judge=None, reason_judge=None, cluster_map: Optional[dict] = None,
                 by_cluster: bool = False, verbose: bool = True, log_every: int = 25,
                 prefix_fn: Optional[Callable[[str, dict, dict, list], Optional[str]]] = None,
                 prompt_builder: Optional["PromptBuilder"] = None) -> Tuple[dict, List[dict]]:
    """Generate on ``frames`` with the ``task`` prompt (s1 | s2) and score everything.

    Returns ``(metrics, records)`` -- see :func:`score_records` / :func:`make_record`. Frames whose
    files cannot be read are skipped (counted in ``metrics['n_skipped']``). ``bad_ids`` defaults to
    ``adapter.bad_token_ids(tokenizer)``. ``prefix_fn(fdir, frame, pano, coco)`` (optional) returns
    an assistant prefix that is teacher-forced before generation (``""``/``None`` = none;
    :data:`SKIP_FRAME` = skip the frame) -- the causal-intervention experiments use it.
    Training-time callers must restore ``model.train()`` themselves.
    """
    tok = processor.tokenizer
    if bad_ids is None:
        bad_ids = adapter.bad_token_ids(tok)
    pb = prompt_builder or PromptBuilder(adapter, processor, long_edge)
    cmap = cluster_map if cluster_map is not None else (load_cluster_map() if by_cluster else {})
    was_training = getattr(model, "training", False)
    model.eval()
    records: List[dict] = []
    n_skip = 0
    t0 = time.time()
    for i, fdir in enumerate(frames):
        try:
            inp, frame, pano, coco = pb.build_for_dir(fdir, task)
        except Exception as e:
            n_skip += 1
            if verbose:
                print(f"[eval] skip {fdir}: {e!r}", flush=True)
            continue
        prefix = None
        if prefix_fn is not None:
            prefix = prefix_fn(fdir, frame, pano, coco)
            if prefix is SKIP_FRAME:
                n_skip += 1
                continue
        g = generate_text(model, tok, inp, max_new_tokens=max_new_tokens, bad_ids=bad_ids,
                          no_repeat_ngram=no_repeat_ngram, forced_prefix=prefix or None)
        rec = make_record(fdir, frame, pano, coco, inp["scale"], adapter.point_codec, g["text"], task=task,
                          n_new_tokens=g["n_new_tokens"], hit_cap=g["hit_cap"],
                          cluster=cmap.get(scene_frame_ids(frame, fdir)[0]) if cmap else None,
                          max_new_tokens=max_new_tokens, prefix=prefix or None)
        rec["guard"] = g["guard"]
        rec["prompt_len"] = g.get("prompt_len")
        records.append(rec)
        if verbose and ((i + 1) % log_every == 0 or i + 1 == len(frames)):
            print(f"[eval:{task}] {i + 1}/{len(frames)} frames", flush=True)
    metrics = score_records(records, task=task, impl_judge=impl_judge, max_new_tokens=max_new_tokens,
                            by_cluster=by_cluster, reason_judge=reason_judge)
    metrics["n_skipped"] = n_skip
    metrics["seconds"] = time.time() - t0
    if was_training:
        model.train()
    return metrics, records


def make_impl_judge(model_name: Optional[str], key_file: Optional[str]):
    """LLM judge for the implication field (relaxed rubric; samples are dropped on an API error)."""
    from training.core.metrics.judges import make_implication_judge, JUDGE_MODEL
    print(f"[judge] implication scored by {model_name or JUDGE_MODEL}", flush=True)
    return make_implication_judge(model=model_name or JUDGE_MODEL, key_file=key_file)


def make_reason_judge_(model_name: Optional[str], key_file: Optional[str]):
    """LLM judge for the reason field (dominant cause and effect, no faithfulness axis)."""
    from training.core.metrics.judges import make_reason_judge, JUDGE_MODEL
    print(f"[judge] reason scored by {model_name or JUDGE_MODEL}", flush=True)
    return make_reason_judge(model=model_name or JUDGE_MODEL, key_file=key_file)


def _f(v: Any, nd: int = 3) -> str:
    """Format an optional float for the one-screen summary."""
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"


def print_summary(m: dict) -> None:
    """Human-readable one-screen summary of a metrics dict."""
    u, te, ce = m.get("understanding") or {}, m.get("traj_extra") or {}, m.get("chain_extra") or {}
    print(f"n_frames={m.get('n_frames')} composite={m.get('composite')}")
    if u:
        keys = ["ctx_macro_acc", "events_f1", "presence_f1", "obj_recall", "obj_recall_rank1", "loc_hit_relaxed",
                "attr_intention_acc", "attr_state_acc", "attr_location_acc", "rank_acc", "count_vs_gt", "composite"]
        print("  understanding: " + " ".join(f"{k}={u[k]:.3f}" for k in keys if isinstance(u.get(k), (int, float))))
    if te:
        print(f"  trajectory: ADE1={te.get('ade1')} ADE3={te.get('ade3')} ADE5={te.get('ade5')} FDE5={te.get('fde5')} "
              f"pred_rate={te.get('pred_rate')} traj_score={te.get('traj_score')} start_acc={te.get('start_frame_acc')} "
              f"(n_start={te.get('n_start')}) static_acc={te.get('static_frame_acc')}")
    if ce:
        print(f"  chain: missing={ce.get('missing')} ego_state_lon_acc={ce.get('ego_state_lon_acc')} "
              f"motion_acc={ce.get('motion_acc')} motion_traj={ce.get('motion_traj_consistency')} "
              f"plan_traj={ce.get('plan_traj_consistency')} hit_cap={ce.get('hit_cap_rate')}")
    o, b, rs, ch = m.get("objects") or {}, m.get("behaviour") or {}, m.get("reason") or {}, m.get("chain") or {}
    if o and "error" not in o:
        print(f"  objects: presence={_f(o.get('presence_acc'))} (base {_f(o.get('presence_baseline_always_yes'))}) "
              f"det_recall={_f(o.get('det_recall'))} r1={_f(o.get('det_recall_rank1'))} "
              f"type_acc={_f(o.get('type_acc_found_taxon'))} in_mask={_f(o.get('in_mask_rate'))} "
              f"fp={o.get('n_false_positive')}")
    if b and "error" not in b:
        sc, ct = b.get("state_coarse") or {}, b.get("content_main") or {}
        print(f"  behaviour: state={_f(sc.get('acc'))}/macro {_f(sc.get('macro_recall'))} "
              f"(base {_f(sc.get('majority_baseline'))}) content={_f(ct.get('acc'))}/macro {_f(ct.get('macro_recall'))} "
              f"red<->green={_f((b.get('red_green') or {}).get('rate'))}")
    if rs:
        print(f"  reason: cause={_f(rs.get('reason_cause'))} effect={_f(rs.get('reason_effect'))} "
              f"missing={_f(rs.get('reason_missing_rate'))} scored={rs.get('reason_n_scored')} "
              f"dropped={rs.get('reason_n_dropped')}")
    if ch and "error" not in ch:
        print(f"  plan: lat={_f(ch.get('plan_tax_lat_acc'))}/macro {_f(ch.get('plan_tax_lat_macro_recall'))} "
              f"lon={_f(ch.get('plan_tax_lon_acc'))}/macro {_f(ch.get('plan_tax_lon_macro_recall'))} "
              f"joint={_f(ch.get('plan_tax_joint_acc'))} (lon_excluded={ch.get('plan_tax_lon_excluded')})")
    if m.get("gen_len"):
        print(f"  gen_len: {m['gen_len']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="dev-set generation evaluation (S1 understanding / S2 chain)")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (save_pretrained); 'base' = adapter base weights")
    ap.add_argument("--adapter", default=None, help="adapter name (default: <ckpt>/adapter_name.json)")
    ap.add_argument("--stage", default="s2", choices=["s1", "s2"], help="prompt/task to evaluate")
    ap.add_argument("--frames", default=os.path.join(DATA, "dev_eval_frames.json"),
                    help="json list of frame dirs | directory | 'rfs'")
    ap.add_argument("--out", default=None, help="metrics json (chains written next to it as *_chains.json)")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--long-edge", type=int, default=PANO_W)
    ap.add_argument("--max-new-tokens", type=int, default=GEN_DEFAULTS["max_new_tokens"])
    ap.add_argument("--no-repeat-ngram", type=int, default=GEN_DEFAULTS["no_repeat_ngram"])
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--by-cluster", action="store_true", help="bucket by scenario cluster (val frames only)")
    ap.add_argument("--judge", action="store_true", help="score implication with the LLM judge")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--key-file", default=API_KEY_FILE, help="OpenAI key file for --judge")
    ap.add_argument("--no-hash-check", action="store_true", help="warn instead of exit on prompt_hash mismatch")
    args = ap.parse_args(argv)

    frames = load_frames(args.frames)
    if args.subset:
        frames = frames[:args.subset]
    ckpt = None if args.ckpt == "base" else args.ckpt
    model, processor, adapter = load_model_and_processor(ckpt, args.adapter, attn=args.attn, device=args.device)
    if ckpt:
        check_prompt_hash(ckpt, adapter, strict=not args.no_hash_check)
    judge = make_impl_judge(args.judge_model, args.key_file) if args.judge else None
    print(f"[eval_dev] adapter={adapter.name} task={args.stage} frames={len(frames)} ckpt={ckpt}", flush=True)
    metrics, records = run_dev_eval(model, processor, adapter, frames, task=args.stage,
                                    max_new_tokens=args.max_new_tokens, long_edge=args.long_edge,
                                    no_repeat_ngram=args.no_repeat_ngram, impl_judge=judge,
                                    by_cluster=args.by_cluster)
    metrics["ckpt"], metrics["adapter"] = ckpt, adapter.name
    print_summary(metrics)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(metrics, open(args.out, "w"), indent=2)
        cp = args.out[:-5] + "_chains.json" if args.out.endswith(".json") else args.out + "_chains.json"
        json.dump(records_to_rows(records), open(cp, "w"), indent=1, ensure_ascii=False)
        print(f"[out] metrics -> {args.out}\n[out] chains -> {cp}", flush=True)


if __name__ == "__main__":
    main()
