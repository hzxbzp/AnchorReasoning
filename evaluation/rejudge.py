#!/usr/bin/env python3
"""CPU re-scoring of a finished evaluation run (eval_val456 / eval_dev output) from its saved chains --
the LLM-judge columns (implication cause/effect, reason cause/effect) can be added or refreshed WITHOUT
re-running the model, because the judges read only the stored text and the ground truth.

    python -m evaluation.rejudge --run-dir eval_results/<run> [--judge] [--stage s1|s2]

Reads  <run>/metrics.json + <run>/chains.json          (eval_val456; stage s2)
       <run>/dev_s1_metrics.json + *_chains.json       (eval_dev --stage s1; --stage s1)
Writes the same files in place (previous copies kept as *.pre_rejudge.json).  Everything except the
judge blocks is recomputed deterministically from the text, so the numbers stay identical.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import List, Optional

from training.adapters import get_adapter
from training.core.paths import API_KEY_FILE, CLUSTER_JSON, RFS_DIR
from evaluation import eval_dev as ED

#: Panorama -> model-input scale for adapters that emit absolute pixels: the image processor's
#: smart_resize maps the 2916 x 1079 panorama onto a 2912 x 1064 grid.
QWEN25_SCALE = (2912 / 2916, 1064 / 1079)

_IMPL_KEYS = ("implication", "implication_cause", "implication_effect", "implication_n_with_gt", "implication_n_scored",
              "implication_n_dropped", "implication_missing_rate")


def resolve_fdir(fdir: str, scene_id: str, frame_id: str) -> Optional[str]:
    """Frame dir stored in a chains row -> an existing frame dir (the rated-frame folder is the fallback)."""
    cands = [fdir]
    b = os.path.basename(fdir.rstrip("/"))
    if b.startswith("DONE_"):
        cands.append(os.path.join(os.path.dirname(fdir), b[5:]))
    cands += [os.path.join(RFS_DIR, f"{scene_id}-{frame_id}"), os.path.join(RFS_DIR, f"DONE_{scene_id}-{frame_id}")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "frame.json")) and os.path.isfile(os.path.join(c, "panorama_geo.json")):
            return c
    return None


def rows_to_records(rows: List[dict], adapter, task: str, cmap: dict) -> List[dict]:
    scale = QWEN25_SCALE if adapter.point_mode == "abs_pixel" else (1.0, 1.0)
    out = []
    for r in rows:
        sid, fid = str(r.get("scene_id")), str(r.get("frame_id"))
        fdir = resolve_fdir(r.get("fdir") or "", sid, fid)
        if fdir is None:
            continue
        frame, pano, coco = ED.load_frame_bundle(fdir)
        rec = ED.make_record(fdir, frame, pano, coco, scale, adapter.point_codec, r.get("text") or "", task=task,
                             n_new_tokens=int(r.get("n_new_tokens") or 0), hit_cap=bool(r.get("hit_cap")),
                             cluster=cmap.get(sid), pred_xy_override=r.get("pred_xy") if r.get("pred_xy") else None,
                             prefix=r.get("prefix"))
        rec["prompt_len"] = r.get("prompt_len")
        out.append(rec)
    return out


def _merge_judged(old: Optional[dict], new: dict) -> dict:
    """The new block replaces the old one; judge results that the OLD block had and the new one lost
    (API failures) are kept."""
    if not isinstance(old, dict):
        return new
    o_r, n_r = old.get("reason"), new.get("reason")
    if isinstance(o_r, dict) and isinstance(n_r, dict) and (o_r.get("reason_n_scored") or 0) > (n_r.get("reason_n_scored") or 0):
        new["reason"] = o_r
    ou, nu = old.get("understanding"), new.get("understanding")
    if isinstance(ou, dict) and isinstance(nu, dict) and (ou.get("implication_n_scored") or 0) > (nu.get("implication_n_scored") or 0):
        for k in _IMPL_KEYS:
            if k in ou:
                nu[k] = ou[k]
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", choices=["s1", "s2"], default="s2")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--key-file", default=API_KEY_FILE)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--cluster-json", default=CLUSTER_JSON)
    args = ap.parse_args()
    rd = args.run_dir
    if args.stage == "s1":
        mpath, cpath, key = os.path.join(rd, "dev_s1_metrics.json"), os.path.join(rd, "dev_s1_metrics_chains.json"), None
    else:
        mpath, cpath, key = os.path.join(rd, "metrics.json"), os.path.join(rd, "chains.json"), "s2"
    mj = json.load(open(mpath))
    adapter = get_adapter(mj["adapter"])
    cmap = ED.load_cluster_map(args.cluster_json)
    impl_judge = ED.make_impl_judge(args.judge_model, args.key_file) if args.judge else None
    reason_judge = ED.make_reason_judge_(args.judge_model, args.key_file) if (args.judge and args.stage == "s2") else None
    t0 = time.time()
    rows = json.load(open(cpath))[: args.subset or None]
    recs = rows_to_records(rows, adapter, args.stage, cmap)
    m = ED.score_records(recs, task=args.stage, impl_judge=impl_judge, reason_judge=reason_judge, by_cluster=True)
    old_block = mj.get(key) if key else mj
    m = _merge_judged(old_block, m)
    shutil.copy(mpath, mpath.replace(".json", ".pre_rejudge.json"))
    if key:
        for k in ("seconds", "n_skipped"):
            if isinstance(old_block, dict) and k in old_block:
                m.setdefault(k, old_block[k])
        mj[key] = m
    else:
        keep = {k: mj[k] for k in ("ckpt", "adapter") if k in mj}
        mj = m; mj.update(keep)
    mj["rejudged"] = {"when": time.strftime("%Y-%m-%d %H:%M"), "judge": bool(args.judge), "n": len(recs)}
    json.dump(mj, open(mpath, "w"), indent=2)
    ED.print_summary(m)
    print(f"[rejudge] {rd}: stage={args.stage} n={len(recs)} judge={args.judge} -> {mpath}", flush=True)
    print(f"[rejudge] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
