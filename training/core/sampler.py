#!/usr/bin/env python3
"""Sampling weights and samplers for the supervised fine-tuning runs.

* ``StageAwareWeightedSampler`` -- Stage-1 sampling that follows the curriculum stage
  (a: intent x context, b: intent x object).
* ``stage2_sample_weights`` -- Stage-2 static weights
  ``min(intent*ctx*obj, cap) * motion * attr`` (motion/attr factors applied AFTER the cap),
  mean-normalised, ready for ``torch.utils.data.WeightedRandomSampler``.
* small helpers (``normalize_mean``, ``weight_marginals``, ``build_sampling``) used by the
  dataset and the training script.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


def normalize_mean(x: Iterable[float]) -> np.ndarray:
    """Return ``x`` as a float64 array scaled to mean 1 (all-zero / empty input -> ones)."""
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64)
    if a.size == 0:
        return a
    m = float(a.mean())
    if not np.isfinite(m) or m <= 0:
        return np.ones_like(a)
    return a / m


def stage2_sample_weights(comp: Dict[str, Sequence[float]], cap: float = 8.0) -> np.ndarray:
    """Stage-2 per-sample weights: ``min(intent*ctx*obj, cap) * motion * attr``.

    ``comp`` is the dict returned by ``WaymoDataset.sample_weight_components`` (each vector
    index-aligned and mean~1). The S1 product is first normalised to mean 1 so that ``cap``
    means "at most ``cap`` x the average frame"; the motion / minority-attribute factors are
    multiplied AFTER the cap (they must not be eaten by it), then the result is
    mean-normalised again. Missing ``motion``/``attr`` -> 1.
    """
    wi = np.asarray(comp["intent"], dtype=np.float64)
    wc = np.asarray(comp.get("ctx", np.ones_like(wi)), dtype=np.float64)
    wo = np.asarray(comp.get("obj", np.ones_like(wi)), dtype=np.float64)
    n = wi.shape[0]
    if wc.shape[0] != n or wo.shape[0] != n:
        raise ValueError(f"component length mismatch: intent={n} ctx={wc.shape[0]} obj={wo.shape[0]}")
    w = normalize_mean(wi * wc * wo)
    if cap is not None and cap > 0:
        w = np.minimum(w, float(cap))
    wm = comp.get("motion")
    wa = comp.get("attr")
    if wm is not None:
        wm = np.asarray(wm, dtype=np.float64)
        if wm.shape[0] != n:
            raise ValueError(f"motion component length {wm.shape[0]} != {n}")
        w = w * wm
    if wa is not None:
        wa = np.asarray(wa, dtype=np.float64)
        if wa.shape[0] != n:
            raise ValueError(f"attr component length {wa.shape[0]} != {n}")
        w = w * wa
    return normalize_mean(w)


def weight_marginals(weights: Sequence[float], keys: Dict[str, Sequence]) -> Dict[str, Dict[str, float]]:
    """Effective (weighted) marginal distribution of categorical per-sample ``keys``.

    ``keys`` maps a name (e.g. ``'intent'``, ``'sample_class'``) to an index-aligned sequence
    of category values; the result gives, per name, the fraction of the sampling mass that
    lands on each category -- what the training script prints at start-up. Fractions
    are exact (each name's values sum to 1); round them when printing, not here.
    """
    w = np.asarray(weights, dtype=np.float64)
    tot = float(w.sum()) or 1.0
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in keys.items():
        if len(vals) != w.shape[0]:
            raise ValueError(f"keys[{name!r}] has {len(vals)} entries, weights has {w.shape[0]}")
        acc: Dict[str, float] = defaultdict(float)
        for v, wi in zip(vals, w):
            acc[str(v)] += float(wi)
        out[name] = {k: v / tot for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    return out


class StageAwareWeightedSampler(Sampler):
    """WeightedRandomSampler whose distribution depends on the CURRENT curriculum stage:
      stage 'a'      -> intent x context weights (turn balance + mild weather/visibility/event balance);
      stage 'b'      -> intent x object weights (multi-object balance + >=3 boost).
    Re-reads the shared curriculum stage every `chunk` draws so a mid-epoch a->b switch
    changes the sampling distribution in near-real-time. Mirrors WeightedRandomSampler
    (with replacement); Accelerate shards it across ranks."""

    def __init__(self, w_intent, w_ctx, w_obj, stage_value, schedule, num_samples, chunk=512):
        wi = np.asarray(w_intent, dtype=np.float64)
        pa = wi * np.asarray(w_ctx, dtype=np.float64)        # stage a: intent x context
        self.p_a = pa / pa.sum()
        pbc = wi * np.asarray(w_obj, dtype=np.float64)       # stage b: intent x object
        self.p_bc = pbc / pbc.sum()
        self.stage_value = stage_value          # shared mp.Value('i'); index into schedule
        self.schedule = list(schedule) if schedule else None
        self.num_samples = int(num_samples)
        self.chunk = int(chunk)
        self._n = self.p_a.shape[0]

    def __len__(self):
        return self.num_samples

    def _stage_a(self):
        if not self.schedule or self.stage_value is None:
            return False
        return self.schedule[self.stage_value.value] == "a"

    def current_probs(self) -> np.ndarray:
        """Probability vector in force right now (stage a -> p_a, else p_bc)."""
        return self.p_a if self._stage_a() else self.p_bc

    def __iter__(self):
        rng = np.random.default_rng()
        done = 0
        while done < self.num_samples:
            k = min(self.chunk, self.num_samples - done)
            p = self.p_a if self._stage_a() else self.p_bc
            for i in rng.choice(self._n, size=k, p=p):
                yield int(i)
            done += k


def build_sampling(ds, cap: Optional[float] = None) -> dict:
    """Convenience for the training script: pick the sampling strategy for ``ds``
    (a ``WaymoDataset``) from its stage / curriculum.

    Returns ``{'sampler': StageAwareWeightedSampler|None, 'sample_weights': list|None,
    'components': dict, 'marginals': dict}``. S1 with a curriculum schedule -> stage-aware
    sampler; otherwise -> static Stage-2 weights (``stage2_sample_weights``) for the
    trainer's ``WeightedRandomSampler``. ``cap`` defaults to ``ds.sampling['combined_cap']``.
    """
    comp = ds.sample_weight_components()
    n = len(ds)
    if ds.stage == "s1" and ds.schedule:
        sampler = StageAwareWeightedSampler(comp["intent"], comp["ctx"], comp["obj"],
                                            ds._stage, ds.schedule, n)
        marg = {"stage_a": ds.weight_marginals(sampler.p_a * n),
                "stage_b": ds.weight_marginals(sampler.p_bc * n)}
        return {"sampler": sampler, "sample_weights": None, "components": comp, "marginals": marg}
    if cap is None:
        cap = float(ds.sampling.get("combined_cap", 8.0))
    w = stage2_sample_weights(comp, cap=cap)
    return {"sampler": None, "sample_weights": w.tolist(), "components": comp,
            "marginals": {"static": ds.weight_marginals(w)}}
