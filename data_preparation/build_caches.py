#!/usr/bin/env python3
"""Build the per-frame caches, aligned line-by-line with ``index_train.json``.

Outputs (all in ``--out-dir``, default ``ANCHOR_DATA``):

* ``index_meta.json``         list of ``[intent_corrected, n_objects]``
* ``index_ctx_meta.json``     list of ``[weather, visibility, has_interesting_event(0/1)]``
  ("interesting" = a traffic event other than 'Roadside parking')
* ``index_motion_meta.json``  list of ``{motion_lon, motion_lat, sample_class, v0, lon_now, has_minority_attr,
  chain_complete, partition}`` (Stage-2 sampling, Stage-2 pool filter, statistics)
* ``attr_class_freq.json``    ``{'intention': {value: count}, 'state': {value: count}}`` counted over the
  objects of the index frames (only for object types whose field set contains the attribute)
* ``caches_stats.json``       marginal distributions + Stage-2 pool statistics

Label rules come from ``training.core.labels``, the single implementation shared with training and evaluation;
``LABELS.source`` records it in the stats.  The ``LABELS`` namespace is also imported by ``make_dev_split.py``.

Usage::

    PYTHONPATH=<repo root> python -m data_preparation.build_caches \
        [--index $ANCHOR_DATA/index_train.json] [--workers 32] [--limit 20 --out-dir /tmp/x]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core import labels as L  # noqa: E402
from training.core import normalize as NORM  # noqa: E402
from training.core.paths import DATA, S2_PARTITIONS, partition_of  # noqa: E402

MOTION_LON = L.MOTION_LON
MOTION_LAT = L.MOTION_LAT
SAMPLE_CLASSES = L.SAMPLE_CLASSES
LON_NOW_CLASSES = L.EGO_LON_CLASSES

MOTION_META_FIELDS = ("motion_lon", "motion_lat", "sample_class", "v0", "lon_now", "has_minority_attr",
                      "chain_complete", "partition")


# =============================================================================================================
# Label rules: ``training.core.labels`` is the single implementation shared with training and evaluation.
# ``LABELS`` is the namespace used by the worker below and by ``make_dev_split.py``.
# =============================================================================================================
def _first(v):
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def attr_str(v) -> str:
    """Attribute value as a stripped string ('' for None/empty); lists take their first element."""
    v = _first(v)
    return str(v).strip() if v is not None else ""


def ego_lon_class(frame: dict) -> str:
    """ego_state longitudinal class (stopped | accelerating | decelerating | cruising) = ``labels.ego_state_label``."""
    return L.ego_state_label(frame)["lon_class"]


LABELS = SimpleNamespace(
    sample_class=L.sample_class,
    motion_label=L.motion_label,
    lat_class=L.lat_class,
    minority_attr_frame=L.minority_attr_frame,
    future_speeds=L.future_speeds,
    ego_lon_class=ego_lon_class,
    v0=L.v0_of,
    plan_to_motion=L.plan_to_motion,
    plan_consistent=L.plan_consistent,
    source="training.core.labels",
)

# =============================================================================================================
# Per-frame worker
# =============================================================================================================
def _load(path: str):
    with open(path) as fh:
        return json.load(fh)


def default_record(fdir: str) -> dict:
    return {
        "ok": False,
        "meta": ["GO_STRAIGHT", 0],
        "ctx": ["Unknown", "Unknown", 0],
        "motion": {"motion_lon": "keep", "motion_lat": "straight", "sample_class": "keep", "v0": 0.0,
                   "lon_now": "cruising", "has_minority_attr": 0, "chain_complete": 0, "partition": partition_of(fdir)},
        "attr": {"intention": {}, "state": {}},
    }


def frame_record(fdir: str) -> dict:
    """Worker: all cache rows for one frame dir (defaults + ok=False when the frame is unreadable)."""
    rec = default_record(fdir)
    try:
        fr = _load(os.path.join(fdir, "frame.json"))
        pg = _load(os.path.join(fdir, "panorama_geo.json"))
    except Exception as e:  # unreadable frame: keep defaults, stay aligned
        rec["error"] = f"load: {e}"
        return rec
    intent = fr.get("intent_corrected") or fr.get("intent") or "GO_STRAIGHT"
    anns = pg.get("annotations") or []
    rec["meta"] = [str(intent), len(anns)]

    ctx = NORM.norm_context(pg.get("context"))
    types = [t for e in (pg.get("traffic_events") or []) for t in (e.get("types") or [])]
    interesting = 1 if any(t != "Roadside parking" for t in types) else 0
    rec["ctx"] = [ctx.get("weather") or "Unknown", ctx.get("visibility") or "Unknown", interesting]

    m = rec["motion"]
    try:
        ml = LABELS.motion_label(fr, intent)
        m["motion_lon"], m["motion_lat"] = ml["lon"], ml["lat"]
        m["sample_class"] = LABELS.sample_class(fr)
        m["lon_now"] = LABELS.ego_lon_class(fr)
        m["v0"] = round(L.v0_of(fr), 3)
    except Exception as e:
        rec["error"] = f"labels: {e}"
    chain = bool(pg.get("reason")) and bool(pg.get("final_plan")) and all(bool(a.get("driving_implication")) for a in anns)
    m["chain_complete"] = int(chain)
    try:
        m["has_minority_attr"] = int(bool(LABELS.minority_attr_frame(pg)))
    except Exception:      # malformed annotations: keep the default (0), stay aligned
        m["has_minority_attr"] = 0

    cnt = {"intention": Counter(), "state": Counter()}
    for a in anns:
        at = a.get("attributes") or {}
        t = NORM.norm_type(str(at.get("type") or ""))
        fields = NORM.fields_for_type(t)
        for k in ("intention", "state"):
            if k in fields:
                v = attr_str(at.get(k))
                if v:
                    cnt[k][v] += 1
    rec["attr"] = {k: dict(c) for k, c in cnt.items()}
    rec["ok"] = "error" not in rec
    return rec


# =============================================================================================================
# Driver
# =============================================================================================================
def build_caches(index: Sequence[str], workers: int = 32, chunksize: int = 64, log=print) -> Tuple[List, List, List, Dict, Dict]:
    """Return ``(meta, ctx_meta, motion_meta, attr_freq, stats)`` aligned with ``index``."""
    t0 = time.time()
    meta, ctx, motion = [], [], []
    freq = {"intention": Counter(), "state": Counter()}
    n_bad, n_label_err = 0, 0
    workers = max(1, min(workers, max(len(index), 1)))
    if workers == 1:
        it = map(frame_record, index)
        pool = None
    else:
        pool = Pool(workers)
        it = pool.imap(frame_record, index, chunksize=chunksize)      # ordered -> aligned
    try:
        for i, rec in enumerate(it):
            meta.append(rec["meta"])
            ctx.append(rec["ctx"])
            motion.append(rec["motion"])
            for k in freq:
                freq[k].update(rec["attr"].get(k, {}))
            if not rec["ok"]:
                if str(rec.get("error", "")).startswith("labels"):
                    n_label_err += 1
                else:
                    n_bad += 1
            if (i + 1) % 20000 == 0:
                log(f"  [caches] {i + 1}/{len(index)}  {time.time() - t0:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    attr_freq = {k: dict(sorted(c.items(), key=lambda t: (-t[1], t[0]))) for k, c in freq.items()}
    stats = summarize(index, meta, ctx, motion, attr_freq)
    stats.update({"n_unreadable_frames": n_bad, "n_label_errors": n_label_err, "elapsed_s": round(time.time() - t0, 1),
                  "label_source": LABELS.source, "created": _dt.datetime.now().isoformat(timespec="seconds")})
    return meta, ctx, motion, attr_freq, stats


def summarize(index: Sequence[str], meta: List, ctx: List, motion: List, attr_freq: Dict) -> Dict:
    """Marginal distributions of the caches + Stage-2 pool statistics (p7..p21 & chain_complete)."""
    def dist(key):
        return dict(sorted(Counter(m[key] for m in motion).items()))

    by_part: Dict[str, Counter] = {}
    for mm, mo in zip(meta, motion):
        c = by_part.setdefault(mo["partition"], Counter())
        c["frames"] += 1
        c["chain_complete"] += mo["chain_complete"]
        c["has_minority_attr"] += mo["has_minority_attr"]
        c["n_objects"] += mm[1]
        c[f"sc_{mo['sample_class']}"] += 1
    s2 = [(mm, mo) for mm, mo in zip(meta, motion) if mo["partition"] in S2_PARTITIONS and mo["chain_complete"] == 1]
    n = len(index)
    return {
        "index_rows": n,
        "intent": dict(sorted(Counter(m[0] for m in meta).items())),
        "n_objects_hist": dict(sorted(Counter(min(m[1], 8) for m in meta).items())),
        "weather": dict(sorted(Counter(c[0] for c in ctx).items())),
        "visibility": dict(sorted(Counter(c[1] for c in ctx).items())),
        "interesting_event_rate": round(sum(c[2] for c in ctx) / max(n, 1), 4),
        "motion_lon": dist("motion_lon"), "motion_lat": dist("motion_lat"),
        "sample_class": dist("sample_class"), "lon_now": dist("lon_now"),
        "has_minority_attr_rate": round(sum(m["has_minority_attr"] for m in motion) / max(n, 1), 4),
        "chain_complete_rate": round(sum(m["chain_complete"] for m in motion) / max(n, 1), 4),
        "by_partition": {p: dict(sorted(c.items())) for p, c in sorted(by_part.items(), key=lambda t: int(t[0][1:]) if t[0][1:].isdigit() else 999)},
        "s2_pool": {
            "n_frames": len(s2),
            "partitions": S2_PARTITIONS,
            "motion_lon": dict(sorted(Counter(mo["motion_lon"] for _m, mo in s2).items())),
            "motion_lat": dict(sorted(Counter(mo["motion_lat"] for _m, mo in s2).items())),
            "sample_class": dict(sorted(Counter(mo["sample_class"] for _m, mo in s2).items())),
            "lon_now": dict(sorted(Counter(mo["lon_now"] for _m, mo in s2).items())),
            "intent": dict(sorted(Counter(mm[0] for mm, _mo in s2).items())),
            "has_minority_attr_rate": round(sum(mo["has_minority_attr"] for _m, mo in s2) / max(len(s2), 1), 4),
        },
        "attr_freq_top": {k: dict(list(v.items())[:12]) for k, v in attr_freq.items()},
        "attr_n_objects": {k: sum(v.values()) for k, v in attr_freq.items()},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(DATA, "index_train.json"))
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N index rows only (smoke run)")
    args = ap.parse_args(argv)

    with open(args.index) as fh:
        index = json.load(fh)
    if args.limit:
        index = index[: args.limit]
    print(f"[build_caches] rows={len(index)} workers={args.workers} labels={LABELS.source}", flush=True)
    meta, ctx, motion, attr_freq, stats = build_caches(index, workers=args.workers, chunksize=args.chunksize,
                                                       log=lambda m: print(m, flush=True))
    stats["index_file"] = os.path.abspath(args.index)
    stats["limit"] = args.limit
    os.makedirs(args.out_dir, exist_ok=True)
    outs = {
        "index_meta.json": meta,
        "index_ctx_meta.json": ctx,
        "index_motion_meta.json": motion,
        "attr_class_freq.json": attr_freq,
        "caches_stats.json": stats,
    }
    for name, obj in outs.items():
        with open(os.path.join(args.out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1 if name.endswith("stats.json") or name.startswith("attr") else None)
    print(f"rows={stats['index_rows']} unreadable={stats['n_unreadable_frames']} label_errors={stats['n_label_errors']}")
    print("motion_lon:", stats["motion_lon"], "\nmotion_lat:", stats["motion_lat"], "\nsample_class:", stats["sample_class"],
          "\nlon_now:", stats["lon_now"])
    print(f"chain_complete_rate={stats['chain_complete_rate']}  minority_attr_rate={stats['has_minority_attr_rate']}")
    print("Stage-2 pool:", json.dumps(stats["s2_pool"]))
    print(f"caches -> {args.out_dir}  ({stats['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
