#!/usr/bin/env python3
"""Capability-aware re-scoring of a released model's outputs.

Input is the per-frame dump written by ``eval_val456`` (``chains.json``), or a native-format dump
converted to that same schema.  For every row the text goes through ``text_extract.extract``
(strict grammar first, prose fallback), a record is rebuilt with ``eval_dev.make_record`` and
everything is scored with the shared ``eval_dev.score_records``.
Then ``capabilities.detect_capabilities`` decides, FROM THE OUTPUT ITSELF, what is reported:

    field absent / constant across frames -> block set to None (printed as '—'), never 0
    types found but no points             -> ``objects`` replaced by the geometry-free ``typeonly`` block
    everything else                       -> the standard metric, with ``_provenance`` counts recorded

Outputs in ``--out``: ``metrics_baseline.json`` (plus an ``official/`` rater-feedback run when the
model produces trajectories) and ``chains_baseline_parsed.json``.

    python -m evaluation.baselines.rescore --chains <run>/chains.json --profile qwen25_7b \
        --out <run>/baseline [--judge] [--no-rfs]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.core import traj_codec as TC
from training.core.paths import API_KEY_FILE, CLUSTER_JSON, RFS_DIR
from evaluation import eval_dev as ED
from evaluation import run_rfs as RR
from training.core.metrics.judges import JUDGE_MAX_DROP
from evaluation.baselines.capabilities import MIN_COVERAGE, MIN_DISTINCT, caps_true, detect_capabilities
from evaluation.baselines.profiles import Profile, get_profile
from evaluation.baselines.text_extract import extract
from evaluation.baselines.typeonly import evaluate_types_only

__all__ = ["QWEN25_SCALE", "point_codec_for", "resolve_fdir", "build_records", "score_baseline", "main"]

# The image is resized before it reaches the backbone, so a pixel coordinate the model writes is
# expressed in the resized frame: smart_resize(1079, 2916, factor=28) -> (2912, 1064).
QWEN25_SCALE = (2912 / 2916, 1064 / 1079)

# metric blocks gated by each capability (block -> profile field)
_GATES: Dict[str, Tuple[str, ...]] = {
    "objects": ("points",), "behaviour": ("behaviour",), "reason": ("reason",),
}
_UNDERSTANDING_IMPL_KEYS = ("implication", "implication_cause", "implication_effect", "implication_n_with_gt",
                            "implication_n_scored", "implication_n_dropped", "implication_missing_rate")
_PLAN_KEY_PREFIXES = ("plan_tax_", "plan_motion_", "plan_traj_", "plan_unspecified", "plan_unmatched", "plan_missing")


def point_codec_for(mode: str):
    """``'abs_pixel'`` / ``'norm1000'`` -> the matching point codec."""
    from training.adapters import make_point_codec
    return make_point_codec(mode)


def resolve_fdir(fdir: str, scene_id: str, frame_id: str) -> Optional[str]:
    """Frame directory recorded in a dump -> a directory that exists here, else ``None``."""
    for c in (fdir, os.path.join(RFS_DIR, f"{scene_id}-{frame_id}")):
        if c and os.path.isfile(os.path.join(c, "frame.json")) and os.path.isfile(os.path.join(c, "panorama_geo.json")):
            return c
    return None


def build_records(rows: Sequence[dict], profile: Profile, subset: Optional[int] = None,
                  cluster_json: str = CLUSTER_JSON) -> Tuple[List[dict], Dict[str, int]]:
    """chains rows -> evaluation records, with tolerant extraction of the chain fields."""
    codec = point_codec_for(profile.point_mode)
    scale = QWEN25_SCALE if profile.point_mode == "abs_pixel" else (1.0, 1.0)
    cmap = ED.load_cluster_map(cluster_json)
    stats = {"n": 0, "skipped": 0, "traj_from_prose": 0, "objects_from_prose": 0, "reason_from_prose": 0,
             "plan_from_prose": 0, "no_pred_xy": 0}
    records: List[dict] = []
    for r in (rows[:subset] if subset else rows):
        sid, fid = str(r.get("scene_id")), str(r.get("frame_id"))
        fdir = resolve_fdir(r.get("fdir") or "", sid, fid)
        if fdir is None:
            stats["skipped"] += 1
            continue
        frame, pano, coco = ED.load_frame_bundle(fdir)
        text = r.get("text") or ""
        pred = extract(text, profile.point_mode)
        prov = pred.get("_provenance", {})
        cot = ((r.get("native") or {}).get("cot") or "").strip()
        if cot and not pred.get("reason"):
            # a native chain-of-causation field IS the model's reason (thinking text that reads as a
            # reason is scored as a reason) -- the whole sentence, not only the clause with a causal marker
            pred["reason"] = cot; prov["reason"] = "native:cot"
        xy = r.get("pred_xy")
        if (xy is None or len(xy) == 0) and pred.get("traj") and prov.get("traj"):
            try:
                xy = TC.upsample_pchip(pred["traj"], TC.hist_xy_from_frame(frame)).tolist()
                stats["traj_from_prose"] += 1
            except Exception:
                xy = None
        if xy is None or len(xy) == 0:
            stats["no_pred_xy"] += 1
            xy = []
        rec = ED.make_record(fdir, frame, pano, coco, scale, codec, text, task="s2",
                             n_new_tokens=int(r.get("n_new_tokens") or 0), hit_cap=bool(r.get("hit_cap")),
                             cluster=cmap.get(sid), pred_xy_override=xy)
        rec["pred"] = pred                                   # tolerant parse replaces the strict one
        rec["prompt_len"] = r.get("prompt_len")
        for k in ("objects", "reason", "final_plan"):
            if prov.get(k):
                stats[f"{k if k != 'final_plan' else 'plan'}_from_prose"] += 1
        if prov.get("reason") == "native:cot":
            stats["reason_from_native_cot"] = stats.get("reason_from_native_cot", 0) + 1
        records.append(rec)
        stats["n"] += 1
    return records, stats


def _mask(m: Dict[str, Any], profile: Profile, records: Sequence[dict]) -> Dict[str, Any]:
    """Score only what the model's OWN output supports (baselines.capabilities), '—' (None) otherwise.

    The static profile forces nothing: a released model whose thinking text turns out to be a real,
    varying reason is scored on reason; one whose reason is the same sentence on every frame gets
    '—'.  ``metrics['capability']['detected']`` records the evidence for each decision."""
    detected = detect_capabilities(records)
    caps: Dict[str, Any] = dict(profile.as_dict())
    scored = caps_true(detected)
    caps.update(scored)
    applied: Dict[str, str] = {}
    for block, fields in _GATES.items():
        if block in m and not all(caps[f] for f in fields):
            m[block] = None; applied[block] = "absent -> None"
    if not caps["points"] and caps["types"]:                 # types from text, no geometry
        m["objects"] = evaluate_types_only(records); applied["objects"] = "typeonly (no points)"
    u = m.get("understanding")
    if isinstance(u, dict) and not caps["implication"]:
        for k in _UNDERSTANDING_IMPL_KEYS:
            if k in u:
                u[k] = None
        applied["understanding.implication_*"] = "absent -> None"
    ch = m.get("chain")
    if isinstance(ch, dict) and not caps["plan"]:
        for k in list(ch):
            if k.startswith(_PLAN_KEY_PREFIXES):
                ch[k] = None
        applied["chain.plan_*"] = "absent -> None"
    if not caps["trajectory"]:
        for k in ("trajectory", "traj", "traj_extra"):
            if k in m:
                m[k] = None
        applied["trajectory"] = "absent -> None"
    prov = {}
    for rec in records:
        for k in (rec.get("pred") or {}).get("_provenance", {}):
            prov[k] = prov.get(k, 0) + 1
    m["capability"] = {"profile": profile.as_dict(), "detected": detected, "scored": scored,
                       "applied": applied, "provenance_counts": prov,
                       "rule": f"scored iff coverage >= {MIN_COVERAGE} and distinct/covered >= {MIN_DISTINCT}"}
    return m


def score_baseline(rows: Sequence[dict], profile: Profile, *, impl_judge=None, reason_judge=None,
                   subset: Optional[int] = None, do_rfs: bool = True, out_dir: Optional[str] = None,
                   tag: str = "s2", cluster_json: str = CLUSTER_JSON) -> Tuple[Dict[str, Any], List[dict]]:
    records, stats = build_records(rows, profile, subset=subset, cluster_json=cluster_json)
    m = ED.score_records(records, task="s2", impl_judge=impl_judge, reason_judge=reason_judge, by_cluster=True)
    m = _mask(m, profile, records)
    m["extraction"] = stats
    if do_rfs and profile.has("trajectory") and out_dir and any(r.get("pred_xy") for r in records):
        try:
            sub = [{"scene_id": r["scene_id"], "frame_id": r["frame_id"],
                    "pred_xy": r["pred_xy"] if r["pred_xy"] is not None else [[0.0, 0.0]] * 20} for r in records]
            rr = RR.run_rfs(sub, os.path.join(out_dir, "official"), cluster_json=cluster_json, tag=tag, verbose=False)
            m["rfs"] = {"frame_mean": rr["frame_mean"], "leaderboard": rr["leaderboard"],
                        "per_cluster": rr["per_cluster"], "summary": rr["summary"]}
        except Exception as e:                                # never lose the text metrics over RFS
            m["rfs"] = {"error": repr(e)}
    return m, records


def _worth_reusing(old: Dict[str, Any], cur: Dict[str, Any], n_key: str, miss_key: str) -> bool:
    """Reuse a previous judge block only if the fresh one scored clearly fewer items (API outage) AND the
    old block saw the same predictions (its missing rate is not worse: an older extraction that found no
    reason at all scores every item 0 'missing' -- reusing that would bury the new numbers)."""
    n_old, n_cur = old.get(n_key) or 0, cur.get(n_key) or 0
    m_old, m_cur = old.get(miss_key), cur.get(miss_key)
    if n_cur >= 0.8 * n_old:
        return False
    if isinstance(m_old, (int, float)) and isinstance(m_cur, (int, float)) and m_old > m_cur + 0.05:
        return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True, help="chains.json of the run to re-score")
    ap.add_argument("--profile", required=True, help="adapter / model name (see baselines.profiles)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-judges", default=None,
                    help="previous metrics_baseline.json: copy its judge results (reason block, understanding.implication_*) "
                         "into this run instead of calling the judges again (re-masking without API credits)")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--judge", action="store_true", help="score implication / reason with the LLM judges")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--key-file", default=API_KEY_FILE)
    ap.add_argument("--no-rfs", action="store_true")
    ap.add_argument("--cluster-json", default=CLUSTER_JSON)
    args = ap.parse_args(argv)

    profile = get_profile(args.profile)
    os.makedirs(args.out, exist_ok=True)
    impl_judge = reason_judge = None
    if args.judge:                                        # capability is decided from the output (capabilities.py),
        impl_judge = ED.make_impl_judge(args.judge_model, args.key_file)      # not from the static profile, so
        reason_judge = ED.make_reason_judge_(args.judge_model, args.key_file) # both judges are always available
    t0 = time.time()
    rows = json.load(open(args.chains))
    m, records = score_baseline(rows, profile, impl_judge=impl_judge, reason_judge=reason_judge,
                                subset=args.subset, do_rfs=not args.no_rfs, out_dir=args.out,
                                cluster_json=args.cluster_json)
    json.dump(ED.records_to_rows(records), open(os.path.join(args.out, "chains_baseline_parsed.json"), "w"),
              indent=1, ensure_ascii=False)
    n = len(records)
    out: Dict[str, Any] = {"profile": profile.as_dict(), "chains": args.chains, "n_frames": n, "metrics": m}
    if args.reuse_judges:
        prev = json.load(open(args.reuse_judges))
        old = prev.get("metrics")
        if isinstance(old, dict):
            if m.get("reason") is not None and old.get("reason") is not None and _worth_reusing(
                    old["reason"], m["reason"], "reason_n_scored", "reason_missing_rate"):
                rb = dict(old["reason"])
                n_s, n_d = rb.get("reason_n_scored") or 0, rb.get("reason_n_dropped") or 0
                if n_d > JUDGE_MAX_DROP * max(n_s + n_d, 1):        # an outage-biased block is not worth reusing
                    for k in ("reason_cause", "reason_effect", "reason_overall"):
                        rb[k] = None
                    rb["reason_judge_incomplete"] = True
                m["reason"] = rb
            cu, ou = m.get("understanding"), old.get("understanding")
            if isinstance(cu, dict) and isinstance(ou, dict) and cu.get("implication_cause") is None \
                    and ou.get("implication_cause") is not None and m.get("capability", {}).get("scored", {}).get("implication") \
                    and _worth_reusing(ou, cu, "implication_n_scored", "implication_missing_rate"):
                n_s, n_d = ou.get("implication_n_scored") or 0, ou.get("implication_n_dropped") or 0
                if n_d <= JUDGE_MAX_DROP * max(n_s + n_d, 1):
                    for k in _UNDERSTANDING_IMPL_KEYS:
                        if k in ou:
                            cu[k] = ou[k]
            m.setdefault("judges_reused_from", args.reuse_judges)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(args.out, "metrics_baseline.json"), "w"), indent=2)
    print(f"[baseline] {profile.display}: n={n} extraction={m['extraction']}")
    ED.print_summary(m)
    if "rfs" in m:
        print(f"  rfs: frame_mean={m['rfs'].get('frame_mean')} leaderboard={m['rfs'].get('leaderboard')}")
    print(f"[baseline] done in {out['seconds']:.0f}s -> {args.out}/metrics_baseline.json", flush=True)


if __name__ == "__main__":
    main()
