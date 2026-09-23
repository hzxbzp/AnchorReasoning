#!/usr/bin/env python3
"""Chain causal-intervention experiments CI-1 / CI-2 / CI-3.

All three re-use the frozen decoding protocol of ``eval_dev.py`` with a *teacher-forced assistant
prefix* (``run_dev_eval(prefix_fn=...)``): the prefix is appended to the prompt as assistant
tokens and the model continues from there.

CI-1  plan/motion intervention (does the emitted trajectory follow the chain?): take the model's own
      chain (base pass), keep everything up to ``<final_plan>``, replace ``<final_plan>``+``<motion>``
      with a *counterfactual* trend class (keep<->stop, accelerate->stop, decelerate->accelerate;
      ``--ci1-targets all`` tries every alternative), force ``<traj>`` and let the model emit the
      waypoints. ``follow_rate`` = share of continuations whose trajectory-derived lon class
      equals the injected class (gate >= 0.70). A *control* re-forces the ORIGINAL plan/motion
      (``self_follow_rate``, reproduction ADE vs the base trajectory).
CI-2  chain vs no chain: ``full`` (the base Stage-2 pass) and ``truncated`` (the same prompt with
      ``<traj>`` forced immediately -- an out-of-distribution truncation of the chain). ADE/FDE,
      start-frame stats, RFS, paired CIs.
CI-3  GT-prefix upper bound: ``understanding`` = GT ego_state + scene/objects (+ n_objects) forced,
      model writes reason/plan/motion/traj; ``chain`` = the whole GT chain through ``<motion>`` forced,
      model writes only ``<traj>``. GT points are SAM2-mask stable interior points (adapter-encoded).

The base Stage-2 pass can be re-used from an ``eval_val456.py`` output directory
(``--base-dir <EVAL/run>`` reads ``chains.json`` and skips that pass).

Outputs (``--out-dir``, default ``EVAL/<run>/ci``)::

    ci_metrics.json      {ci1: {...}, ci2: {...}, ci3: {...}, base: <s2 metrics>}
    ci1_rows.json / ci2_<variant>_rows.json / ci3_<level>_rows.json / base_rows.json
    rfs_<variant>/       run_rfs.py file sets (unless --no-rfs)

CLI::

    python -m evaluation.ci_eval --ckpt <dir> [--adapter <n>] [--which 1,2,3] [--base-dir <EVAL/run>]
        [--subset N] [--ci1-targets one|all] [--no-rfs] [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from training.core.paths import CLUSTER_JSON, EVAL, PANO_W
from training.core import target_builder as TB
from evaluation import eval_dev as ED
from evaluation import run_rfs as RR
from evaluation.eval_val456 import run_name_of, ade5_by_frame, paired

LON_CLASSES = ("accelerate", "decelerate", "keep", "stop")
#: default counterfactual per predicted lon class (``--ci1-targets one``)
INTERVENTION_MAP = {"keep": "stop", "stop": "accelerate", "accelerate": "stop", "decelerate": "accelerate"}
CI1_FOLLOW_GATE = 0.70


# ======================================================================================
# prefix construction
# ======================================================================================
def plan_text_for(lon: str, lat: Optional[str], v0: float) -> str:
    """A canonical final_plan sentence for a trend class (fixed verb set + lateral wording)."""
    lat_txt = {"left_turn": "turn left", "right_turn": "turn right"}.get(lat or "", "keep lane")
    if lon == "stop":
        return "remain stopped" if v0 < 0.5 else "stop and wait"
    if lon == "accelerate":
        return f"{'proceed' if v0 < 0.5 else 'accelerate'} and {lat_txt}"
    if lon == "decelerate":
        return f"decelerate and {lat_txt}"
    return f"{lat_txt} and cruise"


def chain_head(text: str) -> Optional[str]:
    """Everything of a generated S2 answer before ``<final_plan>`` (``None`` if the tag is absent)."""
    if not text:
        return None
    i = text.find("<final_plan>")
    if i < 0:
        i = text.find("<motion>")       # plan missing but motion present: intervene from there
    return text[:i] if i >= 0 else None


def intervention_prefix(base_text: str, lon: str, lat: Optional[str], v0: float) -> Optional[str]:
    head = chain_head(base_text)
    if head is None:
        return None
    lat = lat if lat in ("straight", "left_turn", "right_turn") else "straight"
    return (f"{head}<final_plan>{plan_text_for(lon, lat, v0)}</final_plan>"
            f"<motion>lon={lon} | lat={lat}</motion><traj>")


def gt_prefix(frame: dict, pano: dict, coco: list, scale, point_codec, level: str) -> Optional[str]:
    """GT chain prefix from ``target_builder.build_target(task='s2')``.

    ``level='understanding'`` -> through ``</n_objects>`` (or ``</has_objects>`` when no objects);
    ``level='chain'`` -> through ``</motion>`` plus the opening ``<traj>``.
    """
    pano = json.loads(json.dumps(pano))      # build_target reads _point; do not mutate the caller's dict
    ED.attach_points(pano, coco, point_codec, scale)
    segs = TB.build_target(frame, pano, "s2", point_codec=point_codec)
    texts = [t for t, _, _ in segs]
    if level == "chain":
        try:
            i = texts.index("<traj>")
        except ValueError:
            return None
        return "".join(texts[:i + 1])
    if level == "understanding":
        try:
            i = texts.index("<reason>")
        except ValueError:                  # GT reason missing on this frame: cut before final_plan / motion
            i = next((k for k, t in enumerate(texts) if t in ("<final_plan>", "<motion>")), None)
            if i is None:
                return None
        return "".join(texts[:i])
    raise ValueError(f"unknown CI-3 level {level!r}")


# ======================================================================================
# record helpers
# ======================================================================================
def derived_lon(rec: dict) -> Optional[str]:
    pm = rec.get("pred_motion") or {}
    return pm.get("lon")


def rows_to_records(rows: Sequence[dict], pb: ED.PromptBuilder, adapter, cmap: dict, task: str = "s2",
                    max_new_tokens: int = 1024) -> List[dict]:
    """Rebuild evaluation records from a ``chains*.json`` (re-parses the stored text; re-processes the
    image only to recover ``scale``)."""
    out = []
    for r in rows:
        fdir = r["fdir"]
        try:
            frame, pano, coco = ED.load_frame_bundle(fdir)
            img = pb.image(os.path.join(fdir, "panorama_geo.png"))
        except Exception as e:
            print(f"[ci] skip {fdir}: {e!r}", flush=True)
            continue
        rec = ED.make_record(fdir, frame, pano, coco, img["scale"], adapter.point_codec, r.get("text") or "", task=task,
                             n_new_tokens=int(r.get("n_new_tokens") or 0), hit_cap=bool(r.get("hit_cap")),
                             cluster=cmap.get(ED.scene_frame_ids(frame, fdir)[0]), max_new_tokens=max_new_tokens,
                             prefix=r.get("prefix"))
        out.append(rec)
    return out


def _rate(xs: Sequence[bool]) -> Optional[float]:
    return float(np.mean([bool(x) for x in xs])) if len(xs) else None


def rfs_of(records: Sequence[dict], out_dir: str, tag: str, cluster_json: str) -> Optional[dict]:
    try:
        path = ED.dump_rfs_preds(records, os.path.join(out_dir, f"rfs_preds_{tag}.json"))
        r = RR.run_rfs(path, os.path.join(out_dir, f"rfs_{tag}"), cluster_json=cluster_json, tag=tag, verbose=False)
        print(f"[ci] RFS {tag}: frame-mean={r['frame_mean']:.4f} leaderboard={r['leaderboard']}", flush=True)
        return {"frame_mean": r["frame_mean"], "leaderboard": r["leaderboard"], "per_cluster": r["per_cluster"],
                "per_frame": r["per_frame"], "ade5": r["summary"]["waymo_ade_5s"]}
    except Exception as e:
        print(f"[ci] RFS {tag} failed: {e!r}", flush=True)
        return {"error": repr(e)}


def variant_summary(records: Sequence[dict], task: str = "s2") -> dict:
    m = ED.score_records(records, task=task, by_cluster=False)
    te = m.get("traj_extra") or {}
    return {"n": len(records), "pred_rate": te.get("pred_rate"), "ade1": te.get("ade1"), "ade3": te.get("ade3"),
            "ade5": te.get("ade5"), "fde5": te.get("fde5"), "traj_score": te.get("traj_score"),
            "start_frame_acc": te.get("start_frame_acc"), "start_pred_static": te.get("start_pred_static"),
            "static_frame_acc": te.get("static_frame_acc"), "by_sample_class": te.get("by_sample_class"),
            "gen_len": m.get("gen_len"), "chain_extra": m.get("chain_extra"), "composite": m.get("composite")}


# ======================================================================================
# CI-1
# ======================================================================================
def run_ci1(model, processor, adapter, base: Sequence[dict], out_dir: str, *, targets: str = "one",
            max_new_tokens: int = 128, bad_ids=None, long_edge: int = PANO_W, pb=None, cmap=None) -> dict:
    by_fdir = {r["fdir"]: r for r in base}
    plan: List[Tuple[str, str, str]] = []          # (fdir, kind, target lon)
    for r in base:
        pl = r["pred"].get("motion") or {}
        p_lon = pl.get("lon")
        if chain_head(r["text"]) is None:
            continue
        if p_lon in LON_CLASSES:
            plan.append((r["fdir"], "control", p_lon))
            alts = [c for c in LON_CLASSES if c != p_lon] if targets == "all" else [INTERVENTION_MAP[p_lon]]
        else:                                    # no/unknown predicted motion: intervene by the ego state
            alts = ["stop" if r["v0"] >= 0.5 else "accelerate"]
        plan.extend((r["fdir"], "intervene", t) for t in alts)
    print(f"[ci1] {len(plan)} continuations on {len(set(f for f, _, _ in plan))} frames (targets={targets})", flush=True)

    rows: List[dict] = []
    for kind, target in sorted({(k, t) for _, k, t in plan}):
        fdirs = [f for f, k, t in plan if k == kind and t == target]

        def prefix_fn(fdir, frame, pano, coco, _t=target):
            b = by_fdir[fdir]
            lat = (b["pred"].get("motion") or {}).get("lat")
            return intervention_prefix(b["text"], _t, lat, b["v0"]) or ED.SKIP_FRAME

        _, recs = ED.run_dev_eval(model, processor, adapter, fdirs, task="s2", max_new_tokens=max_new_tokens,
                                  bad_ids=bad_ids, long_edge=long_edge, cluster_map=cmap or {}, verbose=True,
                                  log_every=50, prefix_fn=prefix_fn, prompt_builder=pb)
        for rec in recs:
            b = by_fdir[rec["fdir"]]
            d_lon = derived_lon(rec)
            base_d = derived_lon(b)
            row = {"fdir": rec["fdir"], "scene_id": rec["scene_id"], "frame_id": rec["frame_id"], "kind": kind,
                   "target_lon": target, "base_pred_motion_lon": (b["pred"].get("motion") or {}).get("lon"),
                   "base_derived_lon": base_d, "derived_lon": d_lon, "follow": d_lon == target if d_lon else False,
                   "changed_vs_base": (d_lon != base_d) if (d_lon and base_d) else None,
                   "gt_motion_lon": (rec.get("motion_gt") or {}).get("lon"), "sample_class": rec.get("sample_class"),
                   "v0": rec["v0"], "traj5": rec.get("traj5"), "base_traj5": b.get("traj5"), "ade5": rec.get("ade5"),
                   "base_ade5": b.get("ade5"), "hit_cap": rec.get("hit_cap"), "text": rec["text"]}
            if rec.get("pred_xy") is not None and b.get("pred_xy") is not None:
                a, c = np.asarray(rec["pred_xy"]), np.asarray(b["pred_xy"])
                row["ade_vs_base"] = float(np.hypot(*(a - c).T).mean())
            rows.append(row)
    json.dump(rows, open(os.path.join(out_dir, "ci1_rows.json"), "w"), indent=1, ensure_ascii=False)

    inter = [r for r in rows if r["kind"] == "intervene"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    by_target = {t: {"n": len(x), "follow_rate": _rate([r["follow"] for r in x])}
                 for t in LON_CLASSES for x in [[r for r in inter if r["target_lon"] == t]] if x}
    by_cls = {c: {"n": len(x), "follow_rate": _rate([r["follow"] for r in x])}
              for c in ("stay", "start", "stop", "decel", "accel", "keep")
              for x in [[r for r in inter if r["sample_class"] == c]] if x}
    follow = _rate([r["follow"] for r in inter])
    m = {"n_intervened": len(inter), "n_control": len(ctrl), "targets": targets,
         "follow_rate": follow, "gate_ge_0.70": None if follow is None else follow >= CI1_FOLLOW_GATE,
         "changed_vs_base_rate": _rate([r["changed_vs_base"] for r in inter if r["changed_vs_base"] is not None]),
         "pred_rate": _rate([r["derived_lon"] is not None for r in inter]),
         "by_target": by_target, "by_gt_sample_class": by_cls,
         "control": {"self_follow_rate": _rate([r["follow"] for r in ctrl]),
                     "reproduction_ade_vs_base": float(np.mean([r["ade_vs_base"] for r in ctrl if "ade_vs_base" in r]))
                     if any("ade_vs_base" in r for r in ctrl) else None},
         "intervened_ade5_vs_gt": float(np.nanmean([r["ade5"] for r in inter if r["ade5"] is not None])) if inter else None}
    print(f"[ci1] follow_rate={follow} (gate >= {CI1_FOLLOW_GATE}) self_follow={m['control']['self_follow_rate']} "
          f"by_target={ {k: round(v['follow_rate'], 3) for k, v in by_target.items() if v['follow_rate'] is not None} }", flush=True)
    return m


# ======================================================================================
# CI-2
# ======================================================================================
def run_ci2(model, processor, adapter, base: Sequence[dict], frames: Sequence[str], out_dir: str, *,
            do_rfs: bool = True, bad_ids=None, long_edge: int = PANO_W,
            pb=None, cmap=None, cluster_json: str = CLUSTER_JSON, n_boot: int = 2000) -> dict:
    variants: Dict[str, List[dict]] = {"full": list(base)}
    print("[ci2] truncated chain: SYSTEM_S2 prompt + forced '<traj>'", flush=True)
    _, trunc = ED.run_dev_eval(model, processor, adapter, frames, task="s2", max_new_tokens=128, bad_ids=bad_ids,
                               long_edge=long_edge, cluster_map=cmap or {}, log_every=50, prompt_builder=pb,
                               prefix_fn=lambda *a: "<traj>")
    variants["truncated"] = trunc
    m: Dict[str, Any] = {"variants": {}, "paired_ade5": {}, "paired_rfs": {}}
    for name, recs in variants.items():
        m["variants"][name] = variant_summary(recs)
        json.dump(ED.records_to_rows(recs), open(os.path.join(out_dir, f"ci2_{name}_rows.json"), "w"), indent=1,
                  ensure_ascii=False)
        if do_rfs:
            m["variants"][name]["rfs"] = rfs_of(recs, out_dir, f"ci2_{name}", cluster_json)
    m["paired_ade5"]["full_minus_truncated"] = paired(ade5_by_frame(variants["full"]),
                                                      ade5_by_frame(variants["truncated"]), n_boot)
    rf, rn = (m["variants"]["full"].get("rfs") or {}), (m["variants"]["truncated"].get("rfs") or {})
    if rf.get("per_frame") and rn.get("per_frame"):
        m["paired_rfs"]["full_minus_truncated"] = paired(rf["per_frame"], rn["per_frame"], n_boot)
    for name, v in m["variants"].items():
        print(f"[ci2] {name:10s} ADE5={v['ade5']} FDE5={v['fde5']} pred_rate={v['pred_rate']} "
              f"start_static={v['start_pred_static']} RFS={(v.get('rfs') or {}).get('frame_mean')}", flush=True)
    for k, v in m["variants"].items():          # per_frame RFS maps are large; keep them out of the summary
        if isinstance(v.get("rfs"), dict):
            v["rfs"].pop("per_frame", None)
    return m


# ======================================================================================
# CI-3
# ======================================================================================
def run_ci3(model, processor, adapter, base: Sequence[dict], frames: Sequence[str], out_dir: str, *,
            levels: Sequence[str] = ("understanding", "chain"), do_rfs: bool = True, bad_ids=None,
            long_edge: int = PANO_W, pb=None, cmap=None, cluster_json: str = CLUSTER_JSON, n_boot: int = 2000) -> dict:
    m: Dict[str, Any] = {"levels": {}, "paired_ade5": {}, "paired_rfs": {}}
    base_ade = ade5_by_frame(base)
    base_rfs = None
    if do_rfs:
        base_rfs = rfs_of(base, out_dir, "ci3_base", cluster_json)
    for level in levels:
        print(f"[ci3] GT prefix level={level}", flush=True)
        codec = adapter.point_codec

        def prefix_fn(fdir, frame, pano, coco, _lvl=level):
            scale = pb.image(os.path.join(fdir, "panorama_geo.png"))["scale"] if pb else (1.0, 1.0)
            p = gt_prefix(frame, pano, coco, scale, codec, _lvl)
            return p if p else ED.SKIP_FRAME

        _, recs = ED.run_dev_eval(model, processor, adapter, frames, task="s2",
                                  max_new_tokens=128 if level == "chain" else 512, bad_ids=bad_ids,
                                  long_edge=long_edge, cluster_map=cmap or {}, log_every=50, prompt_builder=pb,
                                  prefix_fn=prefix_fn)
        summ = variant_summary(recs, task="s2")
        if level == "understanding":            # the model's own reason/plan/motion given GT understanding
            sm = ED.score_records(recs, task="s2", by_cluster=False)
            summ["chain"] = {k: (sm.get("chain") or {}).get(k) for k in
                             ("motion_lon_acc", "motion_traj_consistency", "plan_traj_consistency",
                              "plan_traj_inconsistency_rate", "start_frame_motion_acc", "reason_missing_rate",
                              "plan_missing_rate")}
        json.dump(ED.records_to_rows(recs), open(os.path.join(out_dir, f"ci3_{level}_rows.json"), "w"), indent=1,
                  ensure_ascii=False)
        if do_rfs:
            summ["rfs"] = rfs_of(recs, out_dir, f"ci3_{level}", cluster_json)
            if base_rfs and base_rfs.get("per_frame") and summ["rfs"].get("per_frame"):
                m["paired_rfs"][f"{level}_minus_base"] = paired(summ["rfs"]["per_frame"], base_rfs["per_frame"], n_boot)
            summ["rfs"].pop("per_frame", None)
        m["paired_ade5"][f"{level}_minus_base"] = paired(ade5_by_frame(recs), base_ade, n_boot)
        m["levels"][level] = summ
        print(f"[ci3] {level:14s} ADE5={summ['ade5']} FDE5={summ['fde5']} pred_rate={summ['pred_rate']} "
              f"RFS={(summ.get('rfs') or {}).get('frame_mean')}", flush=True)
    if base_rfs:
        base_rfs.pop("per_frame", None)
        m["base_rfs"] = base_rfs
    return m


# ======================================================================================
# main
# ======================================================================================
def load_rows(path: str) -> Optional[List[dict]]:
    return json.load(open(path)) if path and os.path.isfile(path) else None


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="chain causal-intervention experiments CI-1 (plan/motion), CI-2 (no-chain), CI-3 (GT prefix)")
    ap.add_argument("--ckpt", required=True, help="S2 checkpoint dir; 'base' = adapter base weights")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out-dir", default=None, help="default EVAL/<run>/ci")
    ap.add_argument("--base-dir", default=None, help="eval_val456 output dir: re-use its chains.json")
    ap.add_argument("--frames", default="rfs")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--which", default="1,2,3", help="comma list of experiments to run")
    ap.add_argument("--ci1-targets", choices=["one", "all"], default="one")
    ap.add_argument("--ci3-levels", default="understanding,chain")
    ap.add_argument("--max-new-tokens", type=int, default=ED.GEN_DEFAULTS["max_new_tokens"])
    ap.add_argument("--long-edge", type=int, default=PANO_W)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-rfs", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cluster-json", default=CLUSTER_JSON)
    ap.add_argument("--no-hash-check", action="store_true")
    args = ap.parse_args(argv)

    which = {w.strip() for w in args.which.split(",") if w.strip()}
    frames = ED.load_frames(args.frames)
    if args.subset:
        frames = frames[:args.subset]
    ckpt = None if args.ckpt == "base" else args.ckpt
    run = run_name_of(ckpt, args.run)
    out_dir = args.out_dir or os.path.join(EVAL, run, "ci")
    os.makedirs(out_dir, exist_ok=True)
    model, processor, adapter = ED.load_model_and_processor(ckpt, args.adapter, attn=args.attn, device=args.device)
    if ckpt:
        ED.check_prompt_hash(ckpt, adapter, strict=not args.no_hash_check)
    bad_ids = adapter.bad_token_ids(processor.tokenizer)
    cmap = ED.load_cluster_map(args.cluster_json)
    pb = ED.PromptBuilder(adapter, processor, args.long_edge)
    t0 = time.time()
    print(f"[ci] run={run} adapter={adapter.name} frames={len(frames)} which={sorted(which)} out={out_dir}", flush=True)

    # ---- base S2 pass (or re-use)
    base_rows = load_rows(os.path.join(args.base_dir, "chains.json")) if args.base_dir else None
    fset = set(frames)
    if base_rows is not None:
        base_rows = [r for r in base_rows if r["fdir"] in fset]
        print(f"[ci] re-using {len(base_rows)} base chains from {args.base_dir}", flush=True)
        base = rows_to_records(base_rows, pb, adapter, cmap, task="s2", max_new_tokens=args.max_new_tokens)
    else:
        print("[ci] base SYSTEM_S2 pass", flush=True)
        _, base = ED.run_dev_eval(model, processor, adapter, frames, task="s2", max_new_tokens=args.max_new_tokens,
                                  bad_ids=bad_ids, long_edge=args.long_edge, cluster_map=cmap, log_every=50, prompt_builder=pb)
    json.dump(ED.records_to_rows(base), open(os.path.join(out_dir, "base_rows.json"), "w"), indent=1, ensure_ascii=False)
    base_frames = [r["fdir"] for r in base]

    out: Dict[str, Any] = {"run": run, "ckpt": ckpt, "adapter": adapter.name, "n_frames": len(base_frames),
                           "base": variant_summary(base, task="s2")}
    common = dict(bad_ids=bad_ids, long_edge=args.long_edge, pb=pb, cmap=cmap)
    if "1" in which:
        out["ci1"] = run_ci1(model, processor, adapter, base, out_dir, targets=args.ci1_targets, **common)
    if "2" in which:
        out["ci2"] = run_ci2(model, processor, adapter, base, base_frames, out_dir,
                             do_rfs=not args.no_rfs, cluster_json=args.cluster_json, n_boot=args.n_boot, **common)
    if "3" in which:
        levels = [l.strip() for l in args.ci3_levels.split(",") if l.strip()]
        out["ci3"] = run_ci3(model, processor, adapter, base, base_frames, out_dir, levels=levels,
                             do_rfs=not args.no_rfs, cluster_json=args.cluster_json, n_boot=args.n_boot, **common)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(out_dir, "ci_metrics.json"), "w"), indent=2)
    print(f"[ci] done in {out['seconds']:.0f}s -> {out_dir}/ci_metrics.json", flush=True)


if __name__ == "__main__":
    main()
