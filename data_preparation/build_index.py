#!/usr/bin/env python3
"""Build the training index (``index_train.json``) and its statistics (``index_stats.json``).

Selection rules (per-scene, content-aware strides so rare frames are not dropped):

* turn frames (intent GO_LEFT / GO_RIGHT)      -> keep with TURN_STRIDE  (=1, all)
* straight frames with key objects             -> keep with OBJ_STRIDE   (=2)
* straight frames without objects (negatives)  -> keep with EMPTY_STRIDE (=12)
* negatives are then capped to NEG_RATIO_CAP (=35%) of the kept set; ``random.seed(0)`` for the shuffles.

Scenes are scanned by a process pool (one job per scene). ``--exclude-scenes`` drops every frame of the
internal dev scenes from the training index, the statistics file records per-partition / per-scene /
positive-negative counts, and ``--limit N`` scans only the first N scenes (smoke runs).

This module also hosts the directory-discovery helpers shared by the other ``data_preparation`` scripts
(``list_partitions / list_scenes / list_frame_dirs / frame_id / load_scene_list``).

Usage::

    PYTHONPATH=<repo root> python -m data_preparation.build_index \
        --exclude-scenes $ANCHOR_DATA/dev_scenes.json [--workers 32] [--limit 20 --out /tmp/x.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import DATA, DATA_ROOT, partition_of  # noqa: E402

# ---- selection constants -----------------------------------------------------------------------------------
TURN_STRIDE = 1        # keep all turn frames (only ~8% globally)
OBJ_STRIDE = 2         # keep half of object-bearing straight frames
EMPTY_STRIDE = 12      # sparse negatives
NEG_RATIO_CAP = 0.35   # negatives <= 35% of kept set
SEED = 0

FRAME_FILES = ("frame.json", "panorama_geo.json")
TURN_INTENTS = ("GO_LEFT", "GO_RIGHT")


# =============================================================================================================
# Directory discovery helpers (shared)
# =============================================================================================================
def is_hidden(name: str) -> bool:
    """True for AppleDouble/.DS_Store/hidden entries and stale ``DONE_`` markers."""
    return name.startswith(".") or name.startswith("DONE")


def partition_number(pname: str) -> int:
    """'p19--xx' -> 19 (used to sort partitions numerically)."""
    head = pname.split("--")[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    return int(digits) if digits else 10 ** 6


def list_partitions(root: str = DATA_ROOT, names: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Return ``[(partition_short_name, partition_dir), ...]`` sorted by partition number.

    ``names`` optionally restricts to short names such as ``['p19', 'p20', 'p21']``. Only directories whose
    name looks like ``p<N>--<suffix>`` are considered; stray files (e.g. ``label.yml``) are ignored.
    """
    wanted = set(names) if names is not None else None
    out = []
    if not os.path.isdir(root):
        return out
    for entry in os.listdir(root):
        pdir = os.path.join(root, entry)
        if is_hidden(entry) or not os.path.isdir(pdir) or "--" not in entry or not entry.startswith("p"):
            continue
        short = entry.split("--")[0]
        if not short[1:].isdigit():
            continue
        if wanted is not None and short not in wanted:
            continue
        out.append((short, pdir))
    out.sort(key=lambda t: partition_number(t[0]))
    return out


def list_scenes(pdir: str) -> List[str]:
    """Sorted scene directories inside a partition directory (directories only; stray files skipped)."""
    out = []
    for entry in os.listdir(pdir):
        sdir = os.path.join(pdir, entry)
        if is_hidden(entry) or not os.path.isdir(sdir):
            continue
        out.append(sdir)
    out.sort()
    return out


def frame_id(fdir: str) -> int:
    """'<scene>-<fid>' -> int(fid); falls back to all digits of the basename."""
    base = os.path.basename(fdir.rstrip("/"))
    tail = base.rsplit("-", 1)[-1] if "-" in base else base
    digits = "".join(ch for ch in tail if ch.isdigit())
    if digits:
        return int(digits)
    return int("".join(ch for ch in base if ch.isdigit()) or 0)


def list_frame_dirs(scene_dir: str, require: Sequence[str] = FRAME_FILES) -> List[str]:
    """Frame directories of one scene that contain all files in ``require``, sorted by frame id."""
    out = []
    for entry in os.listdir(scene_dir):
        fdir = os.path.join(scene_dir, entry)
        if is_hidden(entry) or not os.path.isdir(fdir):
            continue
        if all(os.path.isfile(os.path.join(fdir, f)) for f in require):
            out.append(fdir)
    out.sort(key=frame_id)
    return out


def scene_id_of(path: str) -> str:
    """Scene id of a scene dir or a frame dir ('<scene>/<scene>-<fid>')."""
    base = os.path.basename(path.rstrip("/"))
    parent = os.path.basename(os.path.dirname(path.rstrip("/")))
    if "-" in base and base.rsplit("-", 1)[0] == parent:
        return parent
    return base


def load_scene_list(path: str) -> Set[str]:
    """Read a scene-id list; accepts a bare JSON list or a dict with a ``scenes`` key."""
    with open(path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        obj = obj.get("scenes", [])
    return {str(s) for s in obj}


def iter_scene_dirs(root: str = DATA_ROOT, partitions: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """``[(partition_short, scene_dir), ...]`` over all partitions (numerically sorted) and scenes (sorted)."""
    out = []
    for short, pdir in list_partitions(root, partitions):
        for sdir in list_scenes(pdir):
            out.append((short, sdir))
    return out


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# =============================================================================================================
# Scanning and frame selection
# =============================================================================================================
def scan_scene(scene_dir: str) -> Tuple[str, List[Tuple[str, str, bool]]]:
    """Worker: ``(scene_dir, [(fdir, intent, has_objects), ...])`` in frame order; unreadable frames skipped."""
    recs = []
    for fdir in list_frame_dirs(scene_dir):
        try:
            fr = load_json(os.path.join(fdir, "frame.json"))
            pg = load_json(os.path.join(fdir, "panorama_geo.json"))
        except Exception:
            continue
        intent = fr.get("intent_corrected") or fr.get("intent") or "GO_STRAIGHT"
        has_obj = bool(pg.get("annotations"))
        recs.append((fdir, str(intent), bool(has_obj)))
    return scene_dir, recs


def select_scene_frames(recs: Sequence[Tuple[str, str, bool]]) -> List[Tuple[str, str, bool]]:
    """Apply the per-scene strides; returns the kept ``(fdir, intent, has_obj)`` records in order."""
    kept = []
    turn_i = obj_i = empty_i = 0
    for fdir, intent, has_obj in recs:
        is_turn = intent in TURN_INTENTS
        if is_turn:
            keep = (turn_i % TURN_STRIDE == 0)
            turn_i += 1
        elif has_obj:
            keep = (obj_i % OBJ_STRIDE == 0)
            obj_i += 1
        else:
            keep = (empty_i % EMPTY_STRIDE == 0)
            empty_i += 1
        if keep:
            kept.append((fdir, intent, has_obj))
    return kept


def build_index(scene_dirs: Sequence[str], workers: int = 32, exclude_scenes: Optional[Set[str]] = None,
                seed: int = SEED, log=print) -> Tuple[List[str], Dict]:
    """Scan ``scene_dirs`` in parallel, apply the selection rules and return ``(index, stats)``.

    ``exclude_scenes`` (scene ids) are removed *before* selection so the negative cap is computed on the
    training population only.
    """
    exclude_scenes = exclude_scenes or set()
    t0 = time.time()
    todo, excluded = [], []
    for sdir in scene_dirs:
        (excluded if scene_id_of(sdir) in exclude_scenes else todo).append(sdir)

    results: Dict[str, List[Tuple[str, str, bool]]] = {}
    if todo:
        workers = max(1, min(workers, len(todo)))
        if workers == 1:
            for sdir in todo:
                results[sdir] = scan_scene(sdir)[1]
        else:
            with Pool(workers) as pool:
                for i, (sdir, recs) in enumerate(pool.imap_unordered(scan_scene, todo, chunksize=2)):
                    results[sdir] = recs
                    if (i + 1) % 200 == 0:
                        log(f"  [scan] {i + 1}/{len(todo)} scenes  {time.time() - t0:.0f}s")

    pos, neg = [], []
    cnt = Counter()
    per_part = {}
    n_scanned = 0
    for sdir in todo:                                  # sorted order -> deterministic shuffles
        recs = results.get(sdir, [])
        part = partition_of(sdir)
        pp = per_part.setdefault(part, Counter())
        pp["scenes"] += 1
        pp["frames_scanned"] += len(recs)
        n_scanned += len(recs)
        for fdir, intent, has_obj in select_scene_frames(recs):
            (pos if has_obj else neg).append((fdir, intent, has_obj, part))
            cnt[(intent, "obj" if has_obj else "empty")] += 1

    rng = random.Random(seed)
    rng.shuffle(neg)
    max_neg = int(NEG_RATIO_CAP / (1 - NEG_RATIO_CAP) * len(pos))
    neg = neg[:max_neg]
    kept = pos + neg
    rng.shuffle(kept)
    index = [k[0] for k in kept]

    intents_pre = Counter()
    for (it, _oe), c in cnt.items():
        intents_pre[it] += c
    intents_kept = Counter(k[1] for k in kept)
    for fdir, intent, has_obj, part in kept:
        pp = per_part[part]
        pp["kept"] += 1
        pp["pos" if has_obj else "neg"] += 1
        pp[f"kept_{intent}"] += 1
    excluded_parts = Counter(partition_of(s) for s in excluded)

    stats = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "params": {"TURN_STRIDE": TURN_STRIDE, "OBJ_STRIDE": OBJ_STRIDE, "EMPTY_STRIDE": EMPTY_STRIDE,
                   "NEG_RATIO_CAP": NEG_RATIO_CAP, "seed": seed},
        "n_scenes_scanned": len(todo),
        "n_scenes_excluded": len(excluded),
        "n_scenes_excluded_by_partition": dict(sorted(excluded_parts.items())),
        "n_frames_scanned": n_scanned,
        "kept": len(index), "pos": len(pos), "neg": len(neg),
        "neg_ratio": round(len(neg) / max(len(index), 1), 4),
        "by_partition": {p: dict(sorted(c.items())) for p, c in sorted(per_part.items(), key=lambda t: partition_number(t[0]))},
        "by_intent_objempty": {f"{it}|{oe}": c for (it, oe), c in sorted(cnt.items())},
        "intent_totals_pre_neg_cap": dict(sorted(intents_pre.items())),
        "intent_totals_kept": dict(sorted(intents_kept.items())),
        "excluded_scenes": sorted(scene_id_of(s) for s in excluded),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return index, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[DATA_ROOT], help="dataset roots holding p<N>--<suffix> dirs")
    ap.add_argument("--partitions", nargs="*", default=None, help="restrict to partitions, e.g. p19 p20 p21")
    ap.add_argument("--exclude-scenes", default=None, help="dev_scenes.json (all frames of these scenes are dropped)")
    ap.add_argument("--out", default=os.path.join(DATA, "index_train.json"))
    ap.add_argument("--stats-out", default=None, help="default: <out dir>/index_stats.json")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N scenes (smoke run)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    scene_dirs = []
    for root in args.roots:
        scene_dirs += [s for _p, s in iter_scene_dirs(root, args.partitions)]
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    exclude = load_scene_list(args.exclude_scenes) if args.exclude_scenes else set()
    print(f"[build_index] scenes={len(scene_dirs)} exclude={len(exclude)} workers={args.workers}", flush=True)

    index, stats = build_index(scene_dirs, workers=args.workers, exclude_scenes=exclude, seed=args.seed,
                               log=lambda m: print(m, flush=True))
    stats["roots"] = list(args.roots)
    stats["exclude_scenes_file"] = args.exclude_scenes
    stats["limit"] = args.limit
    stats["index_file"] = os.path.abspath(args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(index, fh)
    stats_out = args.stats_out or os.path.join(os.path.dirname(os.path.abspath(args.out)), "index_stats.json")
    with open(stats_out, "w") as fh:
        json.dump(stats, fh, indent=1)

    print(f"scenes={stats['n_scenes_scanned']} (excluded dev {stats['n_scenes_excluded']})  frames_scanned={stats['n_frames_scanned']}  "
          f"kept={stats['kept']} (pos={stats['pos']}, neg={stats['neg']}, neg_ratio={stats['neg_ratio']:.2f})")
    print("by partition (kept):")
    for p, c in stats["by_partition"].items():
        print(f"  {p:>4}: scenes={c.get('scenes', 0):4d} scanned={c.get('frames_scanned', 0):6d} kept={c.get('kept', 0):6d} "
              f"pos={c.get('pos', 0):6d} neg={c.get('neg', 0):5d}")
    print("by (intent, obj/empty):", stats["by_intent_objempty"])
    print("intent totals (pre-neg-cap):", stats["intent_totals_pre_neg_cap"], " kept:", stats["intent_totals_kept"])
    print(f"index -> {args.out}\nstats -> {stats_out}  ({stats['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
