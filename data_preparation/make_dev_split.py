#!/usr/bin/env python3
"""Internal dev split: 30 scenes from p19-p21, stratified by intent x future-motion class.

Stratification: every candidate scene gets a stratum key ``<majority intent>|<majority sample_class>`` computed
over its evaluation frames (fid in [140, 160], i.e. the frame positions used by the rated validation set).
Scenes are drawn with proportional (largest-remainder) quotas per stratum, then greedily repaired (swap in/out
single scenes) until the hard constraints on the *evaluation frames* hold: start frames >= 40, braking-to-stop
frames >= 40, left turns >= 15, right turns >= 15 (the ``sample_class`` / ``lat_class`` rules of
``training.core.labels``). Several seeded restarts are tried; the most balanced feasible draw wins.

Outputs (``--out-dir``, default ``ANCHOR_DATA``):

* ``dev_scenes.json``    dict: ``{"scenes": [scene ids], "scene_info": {...}, "stats": {...}}``
* ``dev_eval_frames.json``  list of all frame dirs of the dev scenes with fid in [140, 160]
* ``dev_online_128.json``   fixed random subset of the evaluation frames (online evaluation during training)

Usage::

    PYTHONPATH=<repo root> python -m data_preparation.make_dev_split \
        [--seed 0] [--workers 16] [--limit 5 --n-scenes 2 --relax --out-dir /tmp/x]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import DATA, DATA_ROOT, partition_of  # noqa: E402
from data_preparation.build_caches import LABELS  # noqa: E402
from data_preparation.build_index import frame_id, iter_scene_dirs, list_frame_dirs, load_json, scene_id_of  # noqa: E402

DEV_PARTITIONS = ["p19", "p20", "p21"]
FID_LO, FID_HI = 140, 160
N_SCENES, N_ONLINE = 30, 128
DEFAULT_MINS = {"n_start": 40, "n_stop": 40, "n_left": 15, "n_right": 15}
VERSION = "v2"


# =============================================================================================================
# Per-scene profile (worker)
# =============================================================================================================
def scene_profile(args: Tuple[str, int, int]) -> dict:
    """Worker: per-frame labels of the evaluation frames of one scene + aggregate counts and stratum key."""
    sdir, lo, hi = args
    frames = [f for f in list_frame_dirs(sdir) if lo <= frame_id(f) <= hi]
    per = []
    for fdir in frames:
        try:
            fr = load_json(os.path.join(fdir, "frame.json"))
        except Exception:
            continue
        intent = fr.get("intent_corrected") or fr.get("intent") or "GO_STRAIGHT"
        try:
            sc = LABELS.sample_class(fr)
            lat = LABELS.lat_class(fr, intent)
        except Exception:
            sc, lat = "keep", "straight"
        per.append({"fdir": fdir, "intent": str(intent), "sample_class": sc, "lat": lat})
    scs = Counter(p["sample_class"] for p in per)
    lats = Counter(p["lat"] for p in per)
    intents = Counter(p["intent"] for p in per)
    intent_major = intents.most_common(1)[0][0] if intents else "GO_STRAIGHT"
    motion_major = scs.most_common(1)[0][0] if scs else "keep"
    return {
        "scene_id": scene_id_of(sdir), "scene_dir": sdir, "partition": partition_of(sdir),
        "n_eval_frames": len(per),
        "n_start": scs["start"], "n_stop": scs["stop"], "n_left": lats["left_turn"], "n_right": lats["right_turn"],
        "sample_class_counts": dict(sorted(scs.items())), "intent_counts": dict(sorted(intents.items())),
        "lat_counts": dict(sorted(lats.items())),
        "stratum": f"{intent_major}|{motion_major}",
        "frames": per,
    }


def profile_scenes(scene_dirs: Sequence[str], lo: int = FID_LO, hi: int = FID_HI, workers: int = 16, log=print) -> List[dict]:
    t0 = time.time()
    jobs = [(s, lo, hi) for s in scene_dirs]
    workers = max(1, min(workers, max(len(jobs), 1)))
    if workers == 1:
        out = [scene_profile(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            out = pool.map(scene_profile, jobs, chunksize=2)
    log(f"  [profile] {len(out)} scenes in {time.time() - t0:.0f}s")
    return out


# =============================================================================================================
# Stratified selection with constraint repair
# =============================================================================================================
def totals(sel: Sequence[dict], keys: Sequence[str]) -> Dict[str, int]:
    return {k: sum(s[k] for s in sel) for k in keys}


def shortfall(sel: Sequence[dict], mins: Dict[str, int]) -> int:
    t = totals(sel, list(mins))
    return sum(max(0, m - t[k]) for k, m in mins.items())


def proportional_quota(sizes: Dict[str, int], n: int) -> Dict[str, int]:
    """Largest-remainder allocation of ``n`` slots over strata (never exceeding a stratum's size)."""
    total = sum(sizes.values())
    if total == 0:
        return {k: 0 for k in sizes}
    n = min(n, total)
    raw = {k: n * sz / total for k, sz in sizes.items()}
    base = {k: min(int(math.floor(raw[k])), sizes[k]) for k in sizes}
    rem = n - sum(base.values())
    order = sorted(sizes, key=lambda k: (-(raw[k] - math.floor(raw[k])), -sizes[k], k))
    while rem > 0:
        progressed = False
        for k in order:
            if rem <= 0:
                break
            if base[k] < sizes[k]:
                base[k] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break
    return base


def stratum_deviation(sel: Sequence[dict], quota: Dict[str, int]) -> int:
    c = Counter(s["stratum"] for s in sel)
    return sum(abs(c.get(k, 0) - q) for k, q in quota.items()) + sum(v for k, v in c.items() if k not in quota)


def stratified_draw(scenes: Sequence[dict], n: int, rng: random.Random) -> Tuple[List[dict], Dict[str, int]]:
    groups: Dict[str, List[dict]] = {}
    for s in scenes:
        groups.setdefault(s["stratum"], []).append(s)
    quota = proportional_quota({k: len(v) for k, v in groups.items()}, n)
    sel = []
    for k, q in quota.items():
        if q > 0:
            sel.extend(rng.sample(groups[k], q))
    return sel, quota


def repair(sel: List[dict], pool: Sequence[dict], mins: Dict[str, int], quota: Dict[str, int], rng: random.Random,
           max_iter: int = 400, cand_width: int = 40) -> List[dict]:
    """Greedy single-swap repair: replace one selected scene by an unselected one while the constraint shortfall
    decreases (ties broken by stratum balance). Stops at shortfall 0 or when no improving swap exists."""
    sel = list(sel)
    chosen = {s["scene_id"] for s in sel}
    for _ in range(max_iter):
        sf = shortfall(sel, mins)
        if sf == 0:
            break
        t = totals(sel, list(mins))
        worst = max(mins, key=lambda k: mins[k] - t[k])
        cands = [s for s in pool if s["scene_id"] not in chosen and s[worst] > 0]
        rng.shuffle(cands)
        cands.sort(key=lambda s: -s[worst])
        best = None
        for cand in cands[:cand_width]:
            for i in range(len(sel)):
                new = sel[:i] + sel[i + 1:] + [cand]
                score = (shortfall(new, mins), stratum_deviation(new, quota))
                if best is None or score < best[0]:
                    best = (score, i, cand)
        if best is None or best[0][0] >= sf:
            break
        _, i, cand = best
        chosen.discard(sel[i]["scene_id"])
        chosen.add(cand["scene_id"])
        sel[i] = cand
    return sel


def select_dev_scenes(scenes: Sequence[dict], n: int, mins: Dict[str, int], seed: int = 0, restarts: int = 20,
                      log=print) -> Tuple[List[dict], Dict]:
    """Best of ``restarts`` seeded stratified draws + repair; returns (selected, selection_info)."""
    if n > len(scenes):
        log(f"  [select] only {len(scenes)} candidate scenes for n={n}; taking all")
    best = None
    for r in range(max(1, restarts)):
        rng = random.Random(seed + r)
        sel, quota = stratified_draw(scenes, min(n, len(scenes)), rng)
        sel = repair(sel, scenes, mins, quota, rng)
        score = (shortfall(sel, mins), stratum_deviation(sel, quota))
        if best is None or score < best[0]:
            best = (score, sel, quota, seed + r)
        if score == (0, 0):
            break
    score, sel, quota, used_seed = best
    sel = sorted(sel, key=lambda s: s["scene_id"])
    info = {"shortfall": score[0], "stratum_deviation": score[1], "seed_used": used_seed, "quota": quota,
            "selected_strata": dict(sorted(Counter(s["stratum"] for s in sel).items())),
            "population_strata": dict(sorted(Counter(s["stratum"] for s in scenes).items()))}
    return sel, info


# =============================================================================================================
# Driver
# =============================================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[DATA_ROOT])
    ap.add_argument("--partitions", nargs="+", default=DEV_PARTITIONS)
    ap.add_argument("--fid-lo", type=int, default=FID_LO)
    ap.add_argument("--fid-hi", type=int, default=FID_HI)
    ap.add_argument("--n-scenes", type=int, default=N_SCENES)
    ap.add_argument("--n-online", type=int, default=N_ONLINE)
    ap.add_argument("--min-start", type=int, default=DEFAULT_MINS["n_start"])
    ap.add_argument("--min-stop", type=int, default=DEFAULT_MINS["n_stop"])
    ap.add_argument("--min-left", type=int, default=DEFAULT_MINS["n_left"])
    ap.add_argument("--min-right", type=int, default=DEFAULT_MINS["n_right"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N candidate scenes only (smoke run; implies --relax)")
    ap.add_argument("--relax", action="store_true", help="do not fail when the constraints cannot be met")
    ap.add_argument("--out-dir", default=DATA)
    args = ap.parse_args(argv)
    mins = {"n_start": args.min_start, "n_stop": args.min_stop, "n_left": args.min_left, "n_right": args.min_right}

    scene_dirs = []
    for root in args.roots:
        scene_dirs += [s for _p, s in iter_scene_dirs(root, args.partitions)]
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
        args.relax = True
    print(f"[make_dev_split] candidates={len(scene_dirs)} partitions={args.partitions} labels={LABELS.source}", flush=True)
    profiles = profile_scenes(scene_dirs, args.fid_lo, args.fid_hi, args.workers, log=lambda m: print(m, flush=True))
    profiles = [p for p in profiles if p["n_eval_frames"] > 0]

    sel, info = select_dev_scenes(profiles, args.n_scenes, mins, seed=args.seed, restarts=args.restarts,
                                  log=lambda m: print(m, flush=True))
    tot = totals(sel, list(mins))
    ok = info["shortfall"] == 0

    eval_frames = []
    for s in sel:
        eval_frames.extend(p["fdir"] for p in s["frames"])
    eval_frames.sort(key=lambda f: (scene_id_of(f), frame_id(f)))
    rng = random.Random(args.seed)
    online = sorted(rng.sample(eval_frames, min(args.n_online, len(eval_frames))),
                    key=lambda f: (scene_id_of(f), frame_id(f)))

    eval_sc = Counter(p["sample_class"] for s in sel for p in s["frames"])
    eval_lat = Counter(p["lat"] for s in sel for p in s["frames"])
    eval_intent = Counter(p["intent"] for s in sel for p in s["frames"])
    dev = {
        "version": VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "seed_used": info["seed_used"],
        "partitions": args.partitions, "roots": args.roots,
        "fid_range": [args.fid_lo, args.fid_hi],
        "constraints": mins, "constraints_met": ok, "limit": args.limit,
        "n_scenes": len(sel), "n_candidate_scenes": len(profiles),
        "scenes": [s["scene_id"] for s in sel],
        "scene_info": {s["scene_id"]: {k: s[k] for k in ("partition", "stratum", "n_eval_frames", "n_start", "n_stop",
                                                          "n_left", "n_right", "sample_class_counts", "intent_counts")}
                       for s in sel},
        "stats": {"eval_frames": len(eval_frames), "online_frames": len(online), "totals": tot,
                  "eval_sample_class": dict(sorted(eval_sc.items())), "eval_lat": dict(sorted(eval_lat.items())),
                  "eval_intent": dict(sorted(eval_intent.items())),
                  "by_partition": dict(sorted(Counter(s["partition"] for s in sel).items())),
                  **{k: v for k, v in info.items() if k != "seed_used"}},
        "label_source": LABELS.source,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dev_scenes.json"), "w") as fh:
        json.dump(dev, fh, indent=1)
    with open(os.path.join(args.out_dir, "dev_eval_frames.json"), "w") as fh:
        json.dump(eval_frames, fh, indent=0)
    with open(os.path.join(args.out_dir, "dev_online_128.json"), "w") as fh:
        json.dump(online, fh, indent=0)

    print(f"selected {len(sel)} scenes (of {len(profiles)}); eval frames={len(eval_frames)} online={len(online)}")
    print(f"totals={tot} constraints={mins} met={ok} shortfall={info['shortfall']} stratum_dev={info['stratum_deviation']}")
    print("strata selected:", info["selected_strata"])
    print("strata population:", info["population_strata"])
    print("eval sample_class:", dict(eval_sc), " lat:", dict(eval_lat), " intent:", dict(eval_intent))
    print(f"-> {args.out_dir}/dev_scenes.json, dev_eval_frames.json, dev_online_128.json")
    if not ok and not args.relax:
        print("ERROR: constraints not met (use --relax to accept)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
