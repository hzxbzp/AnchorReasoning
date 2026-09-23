#!/usr/bin/env python3
"""Render the per-frame trajectory and reasoning visualization of an evaluation run (no GPU needed).

Inputs, all written by ``evaluation.eval_val456`` into the run directory:

    <run>/official/rater_feedback_inputs.json   20-point predictions, ``clip_id`` = ``<scene>_<frame>``
    <run>/official/per_sample_results.jsonl     per-frame RFS / scenario cluster
    <run>/chains.json                           final_plan / reason / objects for the text box

One PNG per frame is written, named ``<scene>-<frame>.png``, through
``evaluation.viz_common.make_figure``: panorama with the ground-truth (green), the predicted (red)
and the rater (score-coloured) trajectories, plus an intent banner and a BEV inset.

CLI::

    python -m evaluation.render_rfs_viz --eval-dir <run dir> [--model-name NAME] [--out-dir DIR]
        [--limit N] [--clips <scene>_<frame>,...]
"""
from __future__ import annotations
import argparse, json, os

import numpy as np

from evaluation import viz_common as V
from training.core.parse_output import parse_output
from training.core.paths import RFS_DIR, ROOT, VAL_ROOT

#: Index of the rated validation frames (one JSON object per line with scene_id / frame_id /
#: partition), used only to locate a frame that is not in the flattened rated-frame folder.
#: Same file and override as ``run_rfs.py``.
RATED_INDEX = os.environ.get("ANCHOR_RATED_INDEX",
                             os.path.join(ROOT, "waymo_e2e_index", "val_rated_only.jsonl"))
ANCHOR_IDX = [3, 7, 11, 15, 19]   # t = 1..5 s in the 20-point 0.25 s grid (traj_codec.GT_1HZ_IDX)


def part_map():
    """``{(scene_id, frame_id): partition}`` from the rated index (empty when the file is missing)."""
    m = {}
    if os.path.isfile(RATED_INDEX):
        for line in open(RATED_INDEX):
            line = line.strip()
            if line:
                e = json.loads(line)
                m[(str(e["scene_id"]), str(e["frame_id"]))] = e["partition"]
    return m


def frame_dir(scene: str, frame: str, part: str | None) -> str | None:
    """Folder holding ``frame.json`` + ``panorama_geo.png`` of one frame (``None`` if not found)."""
    cands = [os.path.join(RFS_DIR, f"{scene}-{frame}"),
             os.path.join(RFS_DIR, f"DONE_{scene}-{frame}")]
    if part:
        cands.append(os.path.join(VAL_ROOT, part, scene, f"{scene}-{frame}"))
    for d in cands:
        if os.path.isfile(os.path.join(d, "frame.json")) and os.path.isfile(os.path.join(d, "panorama_geo.png")):
            return d
    return None


def cot_text(ch: dict | None, rfs: float | None, cluster: str | None, ade: float,
             pred_max_disp: float | None = None) -> str:
    """Text box. ``make_figure`` wraps every line at 60 columns itself, so no pre-wrapping here."""
    lines = [f"ADE@5s = {ade:.2f} m   RFS = {rfs:.2f}" if rfs is not None else f"ADE@5s = {ade:.2f} m",
             f"cluster: {cluster or '-'}",
             "red: big labeled dots = the 5 emitted waypoints (t=1..5 s); small hollow = PCHIP-interpolated (0.25 s)"]
    if pred_max_disp is not None:
        lines.append(f"prediction: max displacement {pred_max_disp:.1f} m over 5 s"
                     + ("  (stationary -> not visible in camera view, see BEV)" if pred_max_disp < 1.0 else ""))
    if ch:
        es = ch.get("ego_state") or {}
        if isinstance(es, dict) and es.get("lon"):
            lines.append(f"ego_state: {es.get('lon')} | {es.get('lat')}")
        pm = ch.get("pred_motion") or {}
        if isinstance(pm, dict) and pm.get("text"):
            lines.append(f"motion: {pm['text']}")
        parsed = parse_output(ch.get("text") or "") or {}
        objs = parsed.get("objects") or []
        lines += ["", "final_plan: " + (ch.get("final_plan") or parsed.get("final_plan") or "-"),
                  "", "reason: " + (ch.get("reason") or parsed.get("reason") or "-"),
                  "", f"objects (pred {len(objs)} / says {ch.get('n_objects')}):"]
        for i, o in enumerate(objs, 1):
            attrs = " ".join(f"{k}={o[k]}" for k in ("location", "intention", "state", "content")
                             if o.get(k) not in (None, "", "None"))
            lines.append(f" {i}. {o.get('type')}" + (f" [{attrs}]" if attrs else ""))
            im = (o.get("implication") or "").strip()
            if im:
                lines.append("    -> " + im)
        if ch.get("hit_cap"):
            lines += ["", "!! generation hit the token cap (trajectory missing -> zeros)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clips", default=None, help="comma-separated scene_frame ids to render (subset)")
    a = ap.parse_args()
    E = a.eval_dir.rstrip("/")
    out_dir = a.out_dir or os.path.join(E, "viz")
    os.makedirs(out_dir, exist_ok=True)
    name = a.model_name or os.path.basename(E)

    recs = json.load(open(os.path.join(E, "official", "rater_feedback_inputs.json")))
    per = {}
    for ln in open(os.path.join(E, "official", "per_sample_results.jsonl")):
        d = json.loads(ln); per[d["clip_id"]] = d
    chains = {}
    p = os.path.join(E, "chains.json")
    if os.path.isfile(p):
        for r in json.load(open(p)):
            chains[f"{r['scene_id']}_{r['frame_id']}"] = r
    intr, extr = V.load_calib(); pm = part_map()
    if a.clips:
        want = set(a.clips.split(",")); recs = [r for r in recs if r["clip_id"] in want]
    if a.limit:
        recs = recs[:a.limit]
    n_ok = n_miss = 0
    for i, r in enumerate(recs):
        cid = r["clip_id"]; scene, frame = cid.rsplit("_", 1)
        fd = frame_dir(scene, frame, pm.get((scene, frame)))
        if not fd:
            n_miss += 1; continue
        fj = json.load(open(f"{fd}/frame.json"))
        pred = np.asarray(r["prediction_xy"], dtype=np.float64)
        pred_ego = np.concatenate([pred, np.zeros((len(pred), 1))], axis=1) if len(pred) else None
        gt = V.gt_future_xyz(fj); hist = V.hist_xyz(fj)
        fs = fj["future_states"]; gt20 = np.stack([fs["pos_x"][:20], fs["pos_y"][:20]], axis=1).astype(np.float64)
        m = min(len(pred), len(gt20)); ade = float(np.linalg.norm(pred[:m] - gt20[:m], axis=1).mean()) if m else float("nan")
        ps = per.get(cid, {}); rfs = ps.get("rfs"); cluster = ps.get("cluster")
        intent = fj.get("intent", "")
        pmd = float(np.linalg.norm(pred, axis=1).max()) if len(pred) else None
        cot = cot_text(chains.get(cid), rfs, cluster, ade, pmd)
        hdr = [f"model: {name}", f"scene {scene[:12]}  frame {frame}   cluster {cluster or '-'}",
               f"intent {intent}   ADE@5s {ade:.2f} m   RFS {rfs:.2f}" if rfs is not None else f"intent {intent}   ADE@5s {ade:.2f} m"]
        out = f"{out_dir}/{scene}-{frame}.png"
        try:
            V.make_figure(out, f"{fd}/panorama_geo.png", gt, pred_ego, hist, cot, intent, hdr, intr, extr,
                          prefs=V.pref_trajectories(fj),
                          pred_anchor_idx=ANCHOR_IDX if (pred_ego is not None and len(pred) == 20) else None)
            n_ok += 1
        except Exception as e:  # keep going; report at the end
            print(f"[err] {cid}: {e}", flush=True); n_miss += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(recs)}] rendered {n_ok}", flush=True)
    print(f"[done] {name}: rendered {n_ok}, missing/err {n_miss} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
