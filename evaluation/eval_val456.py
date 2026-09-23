#!/usr/bin/env python3
"""Rater-feedback evaluation of a Stage-2 checkpoint on the 456 rated validation frames.

One generation pass with the Stage-2 system prompt: the chain is parsed, the 5 text waypoints are
PCHIP-upsampled (``traj_codec.upsample_pchip``, history anchors) to the official 20-point grid, and
everything is scored (understanding / trajectory / chain / composite, per scenario cluster).

The official rater-feedback score (``run_rfs.py``: frame-mean + leaderboard cluster-mean, ADE/FDE)
is then computed for that pass and for the reference rows (ground truth / constant velocity /
static / format ceiling), every reference row is compared with the model through a paired per-frame
bootstrap CI, and the acceptance gates are evaluated.

Outputs (``--out-dir``, default ``EVAL/<run>``)::

    chains.json     per-frame rows of the generation pass (incl. the raw text)
    rfs_preds.json  [{scene_id, frame_id, pred_xy(20)}]   (missing -> zeros)
    metrics.json    {s2, rfs, reference, gates, ...}
    official/  reference/<kind>/     run_rfs.py file sets

CLI::

    python -m evaluation.eval_val456 --ckpt <dir> [--adapter <name>] [--run <name>] [--out-dir <dir>]
        [--subset N] [--no-rfs] [--no-reference] [--judge --key-file <f>] [--no-hash-check]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Sequence

from training.core.paths import CLUSTER_JSON, EVAL, PANO_W, API_KEY_FILE
from training.core import prompts as P
from evaluation import eval_dev as ED
from evaluation import run_rfs as RR

REFERENCE_KINDS = RR.REFERENCE_KINDS


# ======================================================================================
# helpers
# ======================================================================================
def run_name_of(ckpt: Optional[str], explicit: Optional[str]) -> str:
    """``--run`` or a name derived from the checkpoint path (``<run>/best/best_stage_s2`` -> ``<run>_best_stage_s2``)."""
    if explicit:
        return explicit
    if not ckpt:
        return "base"
    parts = [p for p in os.path.abspath(ckpt).split(os.sep) if p]
    tail = parts[-3:] if len(parts) >= 3 and parts[-2] == "best" else parts[-2:]
    return "_".join(tail).replace("checkpoint-", "ckpt")


def ade5_by_frame(records: Sequence[dict]) -> Dict[str, Optional[float]]:
    return {f"{r['scene_id']}_{r['frame_id']}": (None if r.get("ade5") is None or r["ade5"] != r["ade5"] else r["ade5"])
            for r in records}


def paired(a_map: Dict[str, Any], b_map: Dict[str, Any], n_boot: int) -> dict:
    keys = [k for k in a_map if k in b_map]
    return ED.paired_bootstrap_ci([a_map[k] for k in keys], [b_map[k] for k in keys], n_boot=n_boot)


def _g(d: Optional[dict], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def gates(m: dict) -> Dict[str, dict]:
    """Acceptance gates -> ``{gate: {value, threshold, op, pass}}`` (``pass=None`` = no support).

    The trajectory and rater-feedback gates are expressed against the reference rows scored in the
    same run (constant velocity, format ceiling) rather than against fixed numbers, so they are
    self-contained; they report ``pass=None`` when ``--no-reference`` skipped those rows.
    """
    s2 = m.get("s2") or {}
    ce, te, ch, u = s2.get("chain_extra") or {}, s2.get("traj_extra") or {}, s2.get("chain") or {}, s2.get("understanding") or {}
    miss = ce.get("missing") or {}
    gl = s2.get("gen_len") or {}
    n = s2.get("n_frames") or 0
    rfs = _g(m, "rfs", "frame_mean")
    cv_ade5 = _g(m, "reference", "cv", "ade5")
    cv_rfs = _g(m, "reference", "cv", "frame_mean")
    out: Dict[str, dict] = {}

    def add(name, value, thr, op):
        ok = None
        if value is not None and thr is not None:
            ok = (value <= thr) if op == "<=" else (value >= thr) if op == ">=" else (value < thr) if op == "<" else (value > thr)
        out[name] = {"value": value, "threshold": thr, "op": op, "pass": ok}

    add("missing_reason", miss.get("reason"), 0.01, "<=")
    add("missing_final_plan", miss.get("final_plan"), 0.02, "<=")
    add("missing_implication", ch.get("implication_missing_rate"), 0.02, "<=")
    add("start_pred_static_rate", te.get("start_pred_static"), 0.40, "<")
    n_stay, stay_acc = te.get("n_stay"), te.get("static_frame_acc")
    add("stay_frames_correct", None if (n_stay is None or stay_acc is None) else round(stay_acc * n_stay), 20, ">=")
    add("ego_state_lon_acc", ce.get("ego_state_lon_acc"), 0.95, ">=")
    add("motion_traj_consistency", ce.get("motion_traj_consistency"), 0.95, ">=")
    add("plan_traj_inconsistency", ch.get("plan_traj_inconsistency_rate"), 0.10, "<=")
    add("rfs_vs_cv", rfs, cv_rfs, ">")
    add("ade5_vs_cv", te.get("ade5"), cv_ade5, "<")
    add("attr_intention_state_pooled", u.get("attr_intention_state_pooled_acc"), 0.70, ">=")
    add("attr_location_lane", u.get("attr_location_lane_acc"), 0.80, ">=")
    add("direction_pixel_consistency", u.get("direction_pixel_consistency"), 0.95, ">=")
    add("degenerate_rate", ch.get("degenerate_rate"), 0.02, "<=")
    add("cap_hit_rate", (gl.get("hit_cap") / n) if n and gl.get("hit_cap") is not None else None, 0.01, "<=")
    add("format_ceiling_vs_cv", _g(m, "reference", "ceiling", "frame_mean"), cv_rfs, ">")
    out["_all_pass"] = {"value": all(v["pass"] for k, v in out.items() if v["pass"] is not None),
                        "n_unsupported": sum(1 for v in out.values() if v["pass"] is None)}
    return out


def print_gates(g: Dict[str, dict]) -> None:
    print("  gates:")
    for k, v in g.items():
        if k.startswith("_"):
            continue
        val = v["value"]
        vs = "n/a" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val))
        st = "PASS" if v["pass"] else ("FAIL" if v["pass"] is False else "----")
        thr = v["threshold"]
        ts = "n/a" if thr is None else (f"{thr:.4f}" if isinstance(thr, float) else str(thr))
        print(f"    {st:4s} {k:32s} {vs:>10s} {v['op']} {ts}")
    print(f"    all supported gates pass: {g['_all_pass']['value']} (unsupported: {g['_all_pass']['n_unsupported']})")


# ======================================================================================
# main
# ======================================================================================
def evaluate(model, processor, adapter, frames: Sequence[str], out_dir: str, *,
             do_rfs: bool = True, reference: bool = True, max_new_tokens: int = 1024, long_edge: int = PANO_W,
             no_repeat_ngram: int = 12, impl_judge=None, reason_judge=None,
             n_boot: int = 2000, cluster_json: str = CLUSTER_JSON,
             ckpt: Optional[str] = None, run: str = "") -> dict:
    """Library entry point behind the CLI: run the Stage-2 pass, the official RFS, the reference
    rows and the acceptance gates, and write the output files into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    cmap = ED.load_cluster_map(cluster_json)
    bad_ids = adapter.bad_token_ids(processor.tokenizer)
    m: Dict[str, Any] = {"run": run, "ckpt": ckpt, "adapter": adapter.name, "n_frames": len(frames),
                         "prompt_hash": P.prompt_hash(adapter.point_codec.desc),
                         "max_new_tokens": max_new_tokens,
                         "long_edge": long_edge, "out_dir": out_dir}

    print(f"[val456] Stage-2 chain pass on {len(frames)} frames", flush=True)
    m_s2, rec_s2 = ED.run_dev_eval(model, processor, adapter, frames, task="s2", max_new_tokens=max_new_tokens,
                                   bad_ids=bad_ids, long_edge=long_edge, no_repeat_ngram=no_repeat_ngram,
                                   impl_judge=impl_judge, reason_judge=reason_judge,
                                   cluster_map=cmap, by_cluster=True)
    m["s2"] = m_s2
    json.dump(ED.records_to_rows(rec_s2), open(os.path.join(out_dir, "chains.json"), "w"), indent=1, ensure_ascii=False)
    preds_path = ED.dump_rfs_preds(rec_s2, os.path.join(out_dir, "rfs_preds.json"))
    ED.print_summary(m_s2)

    if do_rfs:
        try:
            r = RR.run_rfs(preds_path, os.path.join(out_dir, "official"), cluster_json=cluster_json, tag="s2")
            m["rfs"] = {k: r[k] for k in ("n_frames", "frame_mean", "leaderboard", "per_cluster", "summary")}
            m["rfs"]["per_frame"] = r["per_frame"]
        except Exception as e:  # RFS must never lose the generation results
            print(f"[val456] RFS failed: {e!r}", flush=True)
            m["rfs_error"] = repr(e)
        if reference:
            m["reference"] = {}
            for kind in REFERENCE_KINDS:
                try:
                    rr = RR.run_rfs(RR.reference_preds(kind, frames), os.path.join(out_dir, "reference", kind),
                                    cluster_json=cluster_json, tag=kind, verbose=False)
                    m["reference"][kind] = {"frame_mean": rr["frame_mean"], "leaderboard": rr["leaderboard"],
                                            "ade5": rr["summary"]["waymo_ade_5s"], "fde5": rr["summary"]["waymo_fde_5s"]}
                    print(f"[reference] {kind:8s} RFS={rr['frame_mean']:.4f} lb={rr['leaderboard']} "
                          f"ADE5={rr['summary']['waymo_ade_5s']:.3f}", flush=True)
                    if "rfs" in m:
                        m["reference"][kind]["paired_model_minus_ref"] = paired(m["rfs"]["per_frame"], rr["per_frame"], n_boot)
                except Exception as e:
                    m["reference"][kind] = {"error": repr(e)}
    m["gates"] = gates(m)
    m["seconds"] = time.time() - t0
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print_gates(m["gates"])
    print(f"[val456] done in {m['seconds']:.0f}s -> {out_dir}/metrics.json", flush=True)
    return m


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="456-frame rater-feedback evaluation: Stage-2 chain, PCHIP upsampling, official RFS")
    ap.add_argument("--ckpt", required=True, help="S2 checkpoint dir (save_pretrained); 'base' = adapter base weights")
    ap.add_argument("--adapter", default=None, help="adapter name (default: <ckpt>/adapter_name.json)")
    ap.add_argument("--run", default=None, help="run name (default derived from --ckpt); output dir = EVAL/<run>")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--frames", default="rfs", help="'rfs' (456 val frames) | json list | directory")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=ED.GEN_DEFAULTS["max_new_tokens"])
    ap.add_argument("--no-repeat-ngram", type=int, default=ED.GEN_DEFAULTS["no_repeat_ngram"])
    ap.add_argument("--long-edge", type=int, default=PANO_W)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-rfs", action="store_true", help="skip the official RFS computation")
    ap.add_argument("--no-reference", action="store_true", help="skip the GT/CV/static/ceiling reference rows")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cluster-json", default=CLUSTER_JSON)
    ap.add_argument("--judge", action="store_true", help="score implication with the LLM judge")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--key-file", default=API_KEY_FILE, help="OpenAI key file for --judge")
    ap.add_argument("--no-hash-check", action="store_true", help="warn instead of exit on prompt_hash mismatch")
    args = ap.parse_args(argv)

    frames = ED.load_frames(args.frames)
    if args.subset:
        frames = frames[:args.subset]
    ckpt = None if args.ckpt == "base" else args.ckpt
    run = run_name_of(ckpt, args.run)
    out_dir = args.out_dir or os.path.join(EVAL, run)
    model, processor, adapter = ED.load_model_and_processor(ckpt, args.adapter, attn=args.attn, device=args.device)
    if ckpt:
        ED.check_prompt_hash(ckpt, adapter, strict=not args.no_hash_check)
    judge = ED.make_impl_judge(args.judge_model, args.key_file) if args.judge else None
    rjudge = ED.make_reason_judge_(args.judge_model, args.key_file) if args.judge else None
    print(f"[val456] run={run} adapter={adapter.name} ckpt={ckpt} frames={len(frames)} out={out_dir}", flush=True)
    evaluate(model, processor, adapter, frames, out_dir, do_rfs=not args.no_rfs,
             reference=not args.no_reference, max_new_tokens=args.max_new_tokens, long_edge=args.long_edge,
             no_repeat_ngram=args.no_repeat_ngram, impl_judge=judge, reason_judge=rjudge,
             n_boot=args.n_boot,
             cluster_json=args.cluster_json, ckpt=ckpt, run=run)


if __name__ == "__main__":
    main()
