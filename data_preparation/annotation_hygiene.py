#!/usr/bin/env python3
"""Annotation hygiene for the decoded Waymo End-to-End Driving frames. DRY-RUN by default.

Two fixes are applied to the partitions selected with ``--partitions`` (every partition under ``--roots`` when
the option is omitted):

1. Context enum normalisation of ``panorama_geo.json``: scenario spelling variants are unified, an empty
   weather field becomes "Fair", and traffic-event types are mapped onto the canonical set (typos fixed,
   unmappable entries dropped).
2. Intent de-flicker: the ego ``intent`` of near-stationary frames flickers between GO_STRAIGHT / GO_LEFT /
   GO_RIGHT because the future horizon is degenerate, so every low-speed frame inherits the intent of the next
   stable moving segment (the manoeuvre it is about to execute), falling back to the previous stable segment.
   ``--apply`` writes the result as ``intent_corrected`` into every ``frame.json`` (the original ``intent`` is
   preserved; ``--changed-only`` touches only the frames whose label actually changed).

``--apply`` appends one run record to ``--changelog`` (per-file context changes + per-frame intent corrections,
so the run is auditable and reversible) and runs the acceptance check (a random sample of frames:
``intent_corrected`` present, context and traffic-event values inside the enum tables). A dry run writes the
same statistics to ``--report`` and modifies nothing.

Usage::

    PYTHONPATH=<repo root> python -m data_preparation.annotation_hygiene \
        [--dry-run] [--limit 200 --limit-scenes 2] [--apply] [--verify-only]
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import DATA, DATA_ROOT  # noqa: E402
from data_preparation.build_index import (  # noqa: E402
    is_hidden, iter_scene_dirs, list_frame_dirs, list_partitions, scene_id_of,
)

CHANGELOG_DEFAULT = os.path.join(DATA, "annotation_hygiene_changelog.json")
REPORT_DEFAULT = os.path.join(DATA, "annotation_hygiene_dryrun.json")

# Enum tables used by the acceptance check
ENUMS = {
    "weather": {"Sunny", "Cloudy", "Rainy", "Fog", "Fair", "Unknown"},
    "daytime": {"Day", "Night"},
    "visibility": {"Clear", "Reduced", "Limited", "Poor"},
    "road": {"Mid-block", "Intersection", "Diverge", "Merge", "Roundabout"},
}
EVENT_ENUM = {"Roadside parking", "Construction", "Lane guiding", "Lane closure", "Partially occupied lane",
              "Road closure", "Accident", "Traffic signal malfunction", "Traffic jam"}


def partition_roots(roots: Sequence[str], partitions: Optional[Iterable[str]]) -> List[str]:
    """Partition directories of ``roots``, optionally restricted to the given short partition names."""
    out = []
    for root in roots:
        out += [pdir for _short, pdir in list_partitions(root, partitions)]
    return out


# =============================================================================================================
# Step 1: context enum normalisation of panorama_geo.json
# =============================================================================================================
SCENARIO_MAP = {            # comma variants -> space form; empty -> Unknown
    "Suburban, Residential": "Suburban Residential",
    "Urban, Residential": "Urban Residential",
    "Urban, Highway": "Urban Highway",
    "Rural, Highway": "Rural Highway",
    "Suburban, Highway": "Suburban Highway",
    "Parking, Lot": "Parking Lot",
    "Rural, Residential": "Rural Residential",
    "": "Unknown",
}
WEATHER_NULL = "Fair"       # None / "" -> Fair (a missing weather label means fair weather, usually at night)
EVENT_MAP = {               # raw value -> canonical value
    "left roadside parking": "Roadside parking",
    "Vehicle performing a parallel parking maneuver": "Roadside parking",
    "cone on the road": "Partially occupied lane",
    "cone on the road side": "Partially occupied lane",
    "traffic jam": "Traffic jam",          # keep, fix capitalisation
    # every value already in EVENT_ENUM is kept unchanged
}
EVENT_DROP = {              # values that carry no usable event information -> delete
    "pedestrains crossing in groups",
    "none",
    "Airport", "Factory",
    "unpaved parking lot with no marking",
    "No Lane Lines",
}


def iter_pano(roots: Sequence[str]):
    """Yield every ``panorama_geo.json`` below ``roots`` (hidden / marker directories skipped)."""
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not is_hidden(d)]
            if "panorama_geo.json" in filenames:
                yield os.path.join(dirpath, "panorama_geo.json")


def clean_context(doc: dict) -> Dict:
    """Normalise one ``panorama_geo.json`` document in memory; returns the changes ({} when nothing changed)."""
    changes: Dict[str, object] = {}
    ctx = doc.get("context") or {}
    scenario = ctx.get("scenario")
    if scenario in SCENARIO_MAP and SCENARIO_MAP[scenario] != scenario:
        changes["scenario"] = [scenario, SCENARIO_MAP[scenario]]
        ctx["scenario"] = SCENARIO_MAP[scenario]
    weather = ctx.get("weather")
    if weather is None or weather == "":
        changes["weather"] = [weather, WEATHER_NULL]
        ctx["weather"] = WEATHER_NULL
    doc["context"] = ctx

    events = doc.get("traffic_events") or []
    if events:
        orig_events = copy.deepcopy(events)          # full original, so the change can be reverted
        new_events, touched = [], False
        for ev in events:
            new_types = []
            for t in ev.get("types") or []:
                if t in EVENT_DROP:
                    touched = True
                    continue
                mapped = EVENT_MAP.get(t, t)
                if mapped != t:
                    touched = True
                new_types.append(mapped)
            if new_types:
                new_events.append({**ev, "types": new_types})
            else:
                touched = True                       # whole event dropped
        if touched:
            changes["events_orig"] = orig_events
            changes["events_new"] = new_events
            doc["traffic_events"] = new_events
    return changes


def run_clean_labels(roots: Sequence[str], apply: bool, limit: Optional[int] = None) -> Dict:
    """Normalise the context enums of every panorama file below ``roots``; ``apply=False`` only counts."""
    t0 = time.time()
    log: Dict[str, Dict] = {}
    n_scanned = 0
    for path in iter_pano(roots):
        if limit is not None and n_scanned >= limit:
            break
        n_scanned += 1
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        changes = clean_context(doc)
        if not changes:
            continue
        log[path] = changes
        if apply:
            with open(path, "w") as fh:
                json.dump(doc, fh)

    kinds, sc_tr, we_tr, ev_drop = Counter(), Counter(), Counter(), Counter()
    for _p, ch in log.items():
        for k in ch:
            kinds[k] += 1
        if "scenario" in ch:
            sc_tr[f"{ch['scenario'][0]!r}->{ch['scenario'][1]!r}"] += 1
        if "weather" in ch:
            we_tr[f"{ch['weather'][0]!r}->{ch['weather'][1]!r}"] += 1
        if "events_orig" in ch:
            before = Counter(t for e in ch["events_orig"] for t in (e.get("types") or []))
            after = Counter(t for e in ch["events_new"] for t in (e.get("types") or []))
            for t, n in (before - after).items():
                ev_drop[t] += n
    return {
        "mode": "apply" if apply else "dry-run",
        "n_files_scanned": n_scanned,
        "n_files_changed": len(log),
        "by_kind": dict(sorted(kinds.items())),
        "scenario_transitions": dict(sc_tr.most_common()),
        "weather_transitions": dict(we_tr.most_common()),
        "event_types_removed_or_mapped": dict(ev_drop.most_common()),
        "elapsed_s": round(time.time() - t0, 1),
        "changes": log,
    }


# =============================================================================================================
# Step 2: intent de-flicker, one worker per scene
# =============================================================================================================
INTENTS = {"GO_STRAIGHT", "GO_LEFT", "GO_RIGHT"}
ABBR = {"GO_STRAIGHT": "S", "GO_LEFT": "L", "GO_RIGHT": "R", None: "?"}

V_MOVE = 1.0      # m/s: >= V_MOVE counts as "moving" and may anchor a stable segment
V_STAT = 1.0      # m/s: < V_STAT counts as stationary / just starting; only these frames are corrected
L_STABLE = 5      # minimum number of consecutive moving same-intent frames that form a stable anchor


def frame_speed(frame: dict) -> float:
    """Current ego speed = magnitude of the last past velocity sample."""
    ps = frame.get("past_states") or {}
    vx = ps.get("vel_x") or []
    vy = ps.get("vel_y") or []
    if not vx:
        return 0.0
    return math.hypot(vx[-1], vy[-1])


def load_scene(scene_dir: str) -> List[Tuple[str, str, dict]]:
    """``[(frame_id, frame_dir, frame.json), ...]`` of one scene, sorted by numeric frame id."""
    out = []
    for entry in os.listdir(scene_dir):
        fdir = os.path.join(scene_dir, entry)
        path = os.path.join(fdir, "frame.json")
        if is_hidden(entry) or not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        out.append((str(doc.get("frame_id", entry.rsplit("-", 1)[-1])), fdir, doc))
    out.sort(key=lambda t: int("".join(ch for ch in t[0] if ch.isdigit()) or 0))
    return out


def build_anchors(intents: Sequence[Optional[str]], moving: Sequence[bool]) -> List[Tuple[int, int, str]]:
    """Maximal runs of consecutive moving frames with the same intent and length >= L_STABLE.

    Returns ``[(start_idx, end_idx_inclusive, intent), ...]``.
    """
    anchors: List[Tuple[int, int, str]] = []
    i, n = 0, len(intents)
    while i < n:
        if moving[i] and intents[i] in INTENTS:
            j = i
            while j + 1 < n and moving[j + 1] and intents[j + 1] == intents[i]:
                j += 1
            if j - i + 1 >= L_STABLE:
                anchors.append((i, j, str(intents[i])))
            i = j + 1
        else:
            i += 1
    return anchors


def correct_scene(frames: Sequence[Tuple[str, str, dict]]):
    """De-flicker one scene; returns ``(original_intents, speeds, corrected_intents, anchors)``."""
    intents = [d.get("intent") for _fid, _fdir, d in frames]
    speeds = [frame_speed(d) for _fid, _fdir, d in frames]
    moving = [s >= V_MOVE for s in speeds]
    anchors = build_anchors(intents, moving)

    corrected = list(intents)
    for i in range(len(frames)):
        if speeds[i] >= V_STAT:          # moving and transitional frames are left untouched
            continue
        nxt = next((a[2] for a in anchors if a[0] > i), None)        # upcoming manoeuvre
        if nxt is None:
            nxt = next((a[2] for a in reversed(anchors) if a[1] < i), None)   # else the previous one
        if nxt is not None:
            corrected[i] = nxt
    return intents, speeds, corrected, anchors


def intent_scene_worker(args) -> Dict:
    """Worker: intent corrections of one scene; writes ``intent_corrected`` into frame.json when ``apply``."""
    sdir, apply, changed_only = args
    res = {"scene": scene_id_of(sdir), "n_frames": 0, "changes": {}, "n_written": 0, "n_already": 0,
           "n_missing_before": 0, "n_anchors": 0, "transitions": {}}
    frames = load_scene(sdir)
    if not frames:
        return res
    orig, _speeds, corr, anchors = correct_scene(frames)
    res["n_frames"], res["n_anchors"] = len(frames), len(anchors)
    tr = Counter()
    for (fid, fdir, d), o, c in zip(frames, orig, corr):
        if "intent_corrected" not in d:
            res["n_missing_before"] += 1
        if c != o:
            res["changes"][fid] = {"orig": o, "corrected": c}
            tr[f"{ABBR.get(o, o)}->{ABBR.get(c, c)}"] += 1
        if not apply or c is None:
            continue
        if changed_only and c == o:
            continue
        if d.get("intent_corrected") == c:
            res["n_already"] += 1
            continue
        d["intent_corrected"] = c
        with open(os.path.join(fdir, "frame.json"), "w") as fh:
            json.dump(d, fh)
        res["n_written"] += 1
    res["transitions"] = dict(tr)
    return res


def run_intent_fix(scene_dirs: Sequence[str], apply: bool, changed_only: bool = False, workers: int = 16,
                   log=print) -> Dict:
    t0 = time.time()
    jobs = [(s, apply, changed_only) for s in scene_dirs]
    workers = max(1, min(workers, max(len(jobs), 1)))
    if workers == 1:
        results = [intent_scene_worker(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(intent_scene_worker, jobs, chunksize=1)
    st = Counter()
    tr = Counter()
    corrections = {}
    for r in results:
        st["scenes"] += 1
        st["frames"] += r["n_frames"]
        st["changed"] += len(r["changes"])
        st["scenes_changed"] += 1 if r["changes"] else 0
        st["written"] += r["n_written"]
        st["already_correct"] += r["n_already"]
        st["missing_before"] += r["n_missing_before"]
        tr.update(r["transitions"])
        if r["changes"]:
            corrections[r["scene"]] = r["changes"]
    out = {
        "mode": "apply" if apply else "dry-run",
        "params": {"V_MOVE": V_MOVE, "V_STAT": V_STAT, "L_STABLE": L_STABLE, "changed_only": changed_only},
        "stats": dict(sorted(st.items())),
        "changed_rate": round(st["changed"] / max(st["frames"], 1), 4),
        "transitions": dict(tr.most_common()),
        "elapsed_s": round(time.time() - t0, 1),
        "corrections": corrections,
    }
    log(f"  [intent] scenes={st['scenes']} frames={st['frames']} changed={st['changed']} ({out['changed_rate']:.1%}) "
        f"written={st['written']} missing_before={st['missing_before']}  {out['elapsed_s']}s")
    return out


# =============================================================================================================
# Acceptance check
# =============================================================================================================
def verify(scene_dirs: Sequence[str], n: int = 200, seed: int = 0, log=print) -> Dict:
    """Sample ``n`` frames: intent_corrected presence and context/event values inside the enum tables."""
    rng = random.Random(seed)
    frames = []
    for sdir in scene_dirs:
        frames += list_frame_dirs(sdir)
    sample = rng.sample(frames, min(n, len(frames))) if frames else []
    st = Counter()
    bad_ctx, bad_ev, scenarios = Counter(), Counter(), Counter()
    for fdir in sample:
        try:
            with open(os.path.join(fdir, "frame.json")) as fh:
                fr = json.load(fh)
            with open(os.path.join(fdir, "panorama_geo.json")) as fh:
                pg = json.load(fh)
        except Exception:
            st["unreadable"] += 1
            continue
        st["checked"] += 1
        if fr.get("intent_corrected") in INTENTS:
            st["intent_corrected_present"] += 1
        ctx = pg.get("context") or {}
        ok = True
        for k, allowed in ENUMS.items():
            v = ctx.get(k)
            if v not in allowed:
                ok = False
                bad_ctx[f"{k}={v!r}"] += 1
        scenarios[str(ctx.get("scenario"))] += 1
        for e in pg.get("traffic_events") or []:
            for t in e.get("types") or []:
                if t not in EVENT_ENUM:
                    ok = False
                    bad_ev[str(t)] += 1
        if ok:
            st["context_ok"] += 1
    n_ok = max(st["checked"], 1)
    out = {"n_sampled": len(sample), "n_frames_total": len(frames), "seed": seed, "stats": dict(sorted(st.items())),
           "intent_corrected_rate": round(st["intent_corrected_present"] / n_ok, 4),
           "context_enum_ok_rate": round(st["context_ok"] / n_ok, 4),
           "bad_context_values": dict(bad_ctx.most_common()), "bad_event_types": dict(bad_ev.most_common()),
           "scenario_values": dict(scenarios.most_common())}
    log(f"  [verify] n={len(sample)} intent_corrected={out['intent_corrected_rate']:.1%} context_enum_ok={out['context_enum_ok_rate']:.1%} "
        f"bad_ctx={dict(bad_ctx)} bad_events={dict(bad_ev)}")
    return out


# =============================================================================================================
# Driver
# =============================================================================================================
def append_changelog(path: str, run: Dict) -> None:
    log = {"runs": []}
    if os.path.isfile(path):
        with open(path) as fh:
            try:
                log = json.load(fh)
            except Exception:
                log = {"runs": []}
        if not isinstance(log, dict) or "runs" not in log:
            log = {"runs": [log]}
    log["runs"].append(run)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(log, fh, indent=0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[DATA_ROOT])
    ap.add_argument("--partitions", nargs="*", default=None,
                    help="restrict to these partitions, e.g. p19 p20 p21 (default: every partition under --roots)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="modify panorama_geo.json / frame.json in place")
    g.add_argument("--dry-run", action="store_true", help="(default) only count what would change")
    ap.add_argument("--limit", type=int, default=None, help="clean step: first N panorama files only (smoke run)")
    ap.add_argument("--limit-scenes", type=int, default=None, help="intent step: first N scenes only (default 1 when --limit is set)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--skip-intent", action="store_true")
    ap.add_argument("--changed-only", action="store_true",
                    help="write intent_corrected only into the frames whose label changed")
    ap.add_argument("--changelog", default=CHANGELOG_DEFAULT)
    ap.add_argument("--report", default=REPORT_DEFAULT, help="dry-run statistics file")
    ap.add_argument("--verify-only", action="store_true", help="only run the acceptance check")
    ap.add_argument("--verify-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    apply = bool(args.apply)

    roots = partition_roots(args.roots, args.partitions)
    scene_dirs = []
    for root in args.roots:
        scene_dirs += [s for _p, s in iter_scene_dirs(root, args.partitions)]
    if not roots:
        print(f"no partitions {args.partitions or 'p*'} under {args.roots}", file=sys.stderr)
        return 2
    limit_scenes = args.limit_scenes if args.limit_scenes is not None else (1 if args.limit else None)
    intent_scenes = scene_dirs[:limit_scenes] if limit_scenes else scene_dirs
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[annotation_hygiene] {mode} partitions={args.partitions or 'all'} roots={roots} "
          f"scenes={len(scene_dirs)} limit={args.limit} limit_scenes={limit_scenes}", flush=True)

    run = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), "mode": mode.lower(),
           "args": {k: v for k, v in vars(args).items()}, "partition_dirs": roots}
    if args.verify_only:
        run["verify"] = verify(scene_dirs, args.verify_n, args.seed, log=lambda m: print(m, flush=True))
        return 0 if run["verify"]["intent_corrected_rate"] == 1.0 and run["verify"]["context_enum_ok_rate"] == 1.0 else 1

    if not args.skip_clean:
        print("  [clean] normalising context enums" + ("" if apply else " (nothing is written)"), flush=True)
        res = run_clean_labels(roots, apply, args.limit)
        print(f"  [clean] files scanned={res['n_files_scanned']} would_change={res['n_files_changed']} "
              f"by_kind={res['by_kind']} scenario={res['scenario_transitions']} weather={res['weather_transitions']} "
              f"events={res['event_types_removed_or_mapped']}  {res['elapsed_s']}s", flush=True)
        run["clean_labels"] = res
    if not args.skip_intent:
        run["intent"] = run_intent_fix(intent_scenes, apply, args.changed_only, args.workers, log=lambda m: print(m, flush=True))
    if apply:
        run["verify"] = verify(scene_dirs, args.verify_n, args.seed, log=lambda m: print(m, flush=True))
        append_changelog(args.changelog, run)
        print(f"changelog appended -> {args.changelog}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(run, fh, indent=0)
        print(f"DRY-RUN: nothing modified. report -> {args.report}  (re-run with --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
