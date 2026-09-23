#!/usr/bin/env python3
"""Official Waymo rater-feedback score (RFS) + ADE/FDE on ``rfs_preds.json``.

Importable, pure-numpy implementation (no torch, so it also runs on a login node) of the two steps
of the official evaluation: build the metric inputs from the predictions, then score them with the
vendored Waymo scoring function. ``rater_feedback_utils.py`` is used unchanged, exactly as vendored
from the official Waymo Open Dataset end-to-end driving release.

Input schema::

    [{"scene_id": str, "frame_id": str, "pred_xy": [[x, y]] * 20}]      # t = 0.25 .. 5.0 s, ego frame
    {"results": [{"id": "<scene>-<frame>", "pred_xy": ...}]}             # official results format also accepted

``pred_xy`` may be ``null``/empty (-> stationary fallback, counted in ``n_missing_pred``) or shorter
than 20 points (linearly resampled/padded like the official preparation step: ``--src-dt`` is the
step of the given points; ``--has-origin`` drops a leading (0, 0)).

Outputs in ``--out-dir``::

    results_for_official.json    the preds in the official results format
    rater_feedback_inputs.json   20-pt preds + rater trajectories/scores + init speed (input of the Waymo code)
    per_sample_results.jsonl     per-frame official ADE@1/3/5s, FDE@5s, intent, speed, RFS, cluster
    metrics_summary.json         aggregate ADE/FDE + ``rater_feedback`` block (frame-mean, leaderboard cluster-mean)
    rater_feedback_results.json  per-frame RFS + per-cluster table

Reference rows (``--reference``): ``gt`` (the ground-truth future), ``cv`` (constant velocity),
``static`` (all zeros) and ``ceiling`` (GT -> 5 rounded 1 Hz waypoints -> PCHIP with history
anchors -> 20 points; the upper bound imposed by the output format itself).

CLI::

    python -m evaluation.run_rfs --preds <rfs_preds.json> --out-dir <dir> [--src-dt 0.25] [--has-origin]
    python -m evaluation.run_rfs --reference ceiling --out-dir <dir> [--subset N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from training.core.paths import CLUSTER_JSON, DT, FUT_STEPS, RFS_DIR, ROOT, VAL_ROOT
from training.core import traj_codec as TC

#: Directory holding ``rater_feedback_utils.py``, vendored unchanged from the official Waymo Open
#: Dataset end-to-end driving release; override with ``WAYMO_METRICS_SRC``.
VENDORED_SRC = os.environ.get("WAYMO_METRICS_SRC", os.path.join(ROOT, "third_party", "waymo_metrics"))
#: Index of the rated validation frames ({scene_id, frame_id, partition} rows), used only to locate
#: a frame that is not in the flattened rated-frame folder.
RATED_INDEX = os.environ.get("ANCHOR_RATED_INDEX", os.path.join(ROOT, "waymo_e2e_index", "val_rated_only.jsonl"))
REFERENCE_KINDS = ("gt", "cv", "static", "ceiling")
FREQUENCY = 4
LENGTH_SECONDS = 5


# ======================================================================================
# vendored Waymo code
# ======================================================================================
def get_rater_feedback_score_fn():
    """The official ``get_rater_feedback_score`` (imported lazily; numpy only).

    Resolved from the import path first, then from :data:`VENDORED_SRC`.
    """
    try:
        from rater_feedback_utils import get_rater_feedback_score  # type: ignore
    except ImportError:
        if VENDORED_SRC not in sys.path:
            sys.path.insert(0, VENDORED_SRC)
        from rater_feedback_utils import get_rater_feedback_score  # type: ignore
    return get_rater_feedback_score


# ======================================================================================
# frame lookup (flattened RFS_DIR first, partitioned val tree as fallback)
# ======================================================================================
_PART_MAP: Optional[Dict[Tuple[str, str], str]] = None


def partition_map() -> Dict[Tuple[str, str], str]:
    """``{(scene_id, frame_id): partition}`` from the rated index (empty if the file is missing)."""
    global _PART_MAP
    if _PART_MAP is None:
        _PART_MAP = {}
        if os.path.isfile(RATED_INDEX):
            for line in open(RATED_INDEX):
                line = line.strip()
                if line:
                    e = json.loads(line)
                    _PART_MAP[(str(e["scene_id"]), str(e["frame_id"]))] = e["partition"]
    return _PART_MAP


def frame_json_path(scene_id: str, frame_id: str) -> Optional[str]:
    """Path of ``frame.json`` for a rated val frame (``None`` if not found anywhere)."""
    cands = [os.path.join(RFS_DIR, f"{scene_id}-{frame_id}", "frame.json"),
             os.path.join(RFS_DIR, f"DONE_{scene_id}-{frame_id}", "frame.json")]
    part = partition_map().get((str(scene_id), str(frame_id)))
    if part:
        cands.append(os.path.join(VAL_ROOT, part, scene_id, f"{scene_id}-{frame_id}", "frame.json"))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def load_cluster_map(path: str = CLUSTER_JSON) -> Dict[str, str]:
    """``{scene_id: scenario_cluster}`` (empty when the json is missing)."""
    if not path or not os.path.isfile(path):
        return {}
    raw = json.load(open(path))
    return {k: (v.get("scenario_cluster") if isinstance(v, dict) else v) for k, v in raw.items()}


# ======================================================================================
# predictions
# ======================================================================================
def load_preds(path: str) -> List[dict]:
    """Read either the plain list schema or the official ``{"results": [...]}`` schema -> list of rows."""
    obj = json.load(open(path))
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        rows = []
        for r in obj["results"]:
            sid, fid = str(r["id"]).rsplit("-", 1)
            rows.append({"scene_id": sid, "frame_id": fid, "pred_xy": r.get("pred_xy", r.get("pred_traj_xy"))})
        return rows
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a list of {{scene_id, frame_id, pred_xy}} rows")
    return [{"scene_id": str(r["scene_id"]), "frame_id": str(r["frame_id"]), "pred_xy": r.get("pred_xy")} for r in obj]


def resample20(pred_pts, src_dt: float = DT, has_origin: bool = False) -> np.ndarray:
    """(N,2) at t = src_dt, 2*src_dt, ... -> (20,2) at t = 0.25 .. 5.0 (linear, origin at t = 0,
    last point held beyond the horizon). Empty/None -> stationary zeros (the official rule)."""
    if pred_pts is None:
        return np.zeros((FUT_STEPS, 2))
    pts = np.asarray(pred_pts, dtype=np.float64).reshape(-1, 2)
    if has_origin and pts.shape[0]:
        pts = pts[1:]
    if pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
        return np.zeros((FUT_STEPS, 2))
    n = pts.shape[0]
    if n == FUT_STEPS and abs(src_dt - DT) < 1e-9:
        return pts.copy()
    src_t = np.concatenate([[0.0], np.arange(1, n + 1) * src_dt])
    src_xy = np.concatenate([[[0.0, 0.0]], pts], axis=0)
    tgt = np.arange(1, FUT_STEPS + 1) * DT
    return np.stack([np.interp(tgt, src_t, src_xy[:, 0]), np.interp(tgt, src_t, src_xy[:, 1])], axis=1)


def init_speed(frame: dict) -> float:
    vx, vy = TC.ego_velocity(frame)
    return float(math.hypot(vx, vy))


def rated_trajectories(frame: dict) -> Tuple[List[List[List[float]]], List[float]]:
    """Rater alternatives ``([[x, y] * T_i], scores)`` of a frame (only entries with a score)."""
    trajs, scores = [], []
    for p in frame.get("preference_trajectories") or []:
        if not isinstance(p, dict) or p.get("preference_score", -1) == -1 or not isinstance(p.get("pos_x"), list):
            continue
        trajs.append(np.stack([p["pos_x"], p["pos_y"]], axis=-1).tolist())
        scores.append(float(p["preference_score"]))
    return trajs, scores


def prepare_inputs(preds: Sequence[dict], src_dt: float = DT, has_origin: bool = False,
                   cluster_map: Optional[Dict[str, str]] = None) -> dict:
    """Official preparation step: preds -> 20-pt trajectories, official ADE/FDE, rater inputs.

    Returns ``dict(rater_inputs, per_sample, summary, skipped)``; frames whose ``frame.json`` is not
    found or has < 20 GT points are skipped (listed in ``skipped``).
    """
    cmap = cluster_map or {}
    rater_inputs: List[dict] = []
    per_sample: List[dict] = []
    skipped: List[str] = []
    ade = {"1": [], "3": [], "5": []}
    fde: List[float] = []
    by_intent: Dict[str, List[float]] = defaultdict(list)
    n_missing = 0
    for r in preds:
        sid, fid = str(r["scene_id"]), str(r["frame_id"])
        fp = frame_json_path(sid, fid)
        if fp is None:
            skipped.append(f"{sid}-{fid}: frame.json not found")
            continue
        frame = json.load(open(fp))
        fs = frame.get("future_states") or {}
        px, py = fs.get("pos_x") or [], fs.get("pos_y") or []
        if min(len(px), len(py)) < FUT_STEPS:
            skipped.append(f"{sid}-{fid}: GT shorter than {FUT_STEPS}")
            continue
        gt = np.stack([px[:FUT_STEPS], py[:FUT_STEPS]], axis=1).astype(np.float64)
        raw = r.get("pred_xy")
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            n_missing += 1
        pred20 = resample20(raw, src_dt, has_origin)
        a1, a3, a5, f5 = TC.traj_ade_fde(pred20, gt)
        ade["1"].append(a1); ade["3"].append(a3); ade["5"].append(a5); fde.append(f5)
        intent = frame.get("intent_corrected") or frame.get("intent") or ""
        spd = init_speed(frame)
        by_intent[intent].append(a5)
        clip = f"{sid}_{fid}"
        per_sample.append({"clip_id": clip, "scene_id": sid, "frame_id": fid, "intent": intent,
                           "waymo_ade_1s": a1, "waymo_ade_3s": a3, "waymo_ade_5s": a5, "waymo_fde": f5,
                           "initial_speed_mps": spd, "cluster": cmap.get(sid), "missing_pred": raw is None})
        trajs, scores = rated_trajectories(frame)
        if trajs:
            rater_inputs.append({"clip_id": clip, "intent": intent, "prediction_xy": pred20.tolist(),
                                 "prediction_proba": 1.0, "rater_trajectories": trajs, "rater_scores": scores,
                                 "initial_speed_mps": spd})
    n = len(per_sample)
    summary = {"n_frames": n, "n_missing_pred": n_missing, "n_skipped": len(skipped),
               "waymo_ade_1s": float(np.mean(ade["1"])) if n else None,
               "waymo_ade_3s": float(np.mean(ade["3"])) if n else None,
               "waymo_ade_5s": float(np.mean(ade["5"])) if n else None,
               "waymo_fde_5s": float(np.mean(fde)) if n else None,
               "by_intent_ade5s": {k: float(np.mean(v)) for k, v in by_intent.items()}}
    return {"rater_inputs": rater_inputs, "per_sample": per_sample, "summary": summary, "skipped": skipped}


# ======================================================================================
# RFS
# ======================================================================================
def compute_rfs(rater_inputs: Sequence[dict], cluster_map: Optional[Dict[str, str]] = None,
                frequency: int = FREQUENCY, length_seconds: int = LENGTH_SECONDS) -> dict:
    """Official rater-feedback scoring of in-memory rater inputs.

    Returns ``dict(n_rated_frames, frame_mean_rater_feedback_score, leaderboard_overall_rater_feedback_score,
    per_cluster, per_frame_scores, per_frame (clip_id/scene_id/frame_id/score/cluster), percentiles)``.
    """
    if not rater_inputs:
        raise ValueError("empty rater inputs -- nothing to compute")
    fn = get_rater_feedback_score_fn()
    B = len(rater_inputs)
    inference = np.stack([np.asarray(r["prediction_xy"], dtype=np.float64)[None] for r in rater_inputs], axis=0)
    probs = np.ones((B, 1), dtype=np.float64)
    rater_trajs = [[np.asarray(t, dtype=np.float64) for t in r["rater_trajectories"]] for r in rater_inputs]
    rater_labels = [np.asarray(r["rater_scores"], dtype=np.float64) for r in rater_inputs]
    speed = np.asarray([r["initial_speed_mps"] for r in rater_inputs], dtype=np.float64)
    res = fn(inference_trajectories=inference, inference_probs=probs, rater_specified_trajectories=rater_trajs,
             rater_feedback_labels=rater_labels, init_speed=speed, frequency=frequency,
             length_seconds=length_seconds, output_trust_region_visualization=False)
    scores = np.asarray(res["rater_feedback_score"], dtype=np.float64).reshape(-1)
    cmap = cluster_map or {}
    per_frame: List[dict] = []
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    unknown = 0
    for r, s in zip(rater_inputs, scores.tolist()):
        sid, fid = r["clip_id"].rsplit("_", 1)
        c = cmap.get(sid)
        if c is None:
            unknown += 1
            c = "Unknown"
        by_cluster[c].append(s)
        per_frame.append({"clip_id": r["clip_id"], "scene_id": sid, "frame_id": fid, "score": s, "cluster": c})
    per_cluster = {c: {"n": len(v), "mean_RFS": float(np.mean(v))}
                   for c, v in sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))}
    means = [d["mean_RFS"] for c, d in per_cluster.items() if c != "Unknown"]
    return {"n_rated_frames": B, "frequency_hz": frequency, "length_seconds": length_seconds,
            "frame_mean_rater_feedback_score": float(scores.mean()),
            "leaderboard_overall_rater_feedback_score": float(np.mean(means)) if means else None,
            "n_unknown_cluster": unknown, "per_cluster": per_cluster,
            "percentiles": {f"p{q}": float(np.percentile(scores, q)) for q in (10, 25, 50, 75, 90)},
            "per_frame_scores": scores.tolist(), "per_frame": per_frame}


def run_rfs(preds, out_dir: Optional[str], cluster_json: str = CLUSTER_JSON, src_dt: float = DT,
            has_origin: bool = False, tag: str = "", verbose: bool = True) -> dict:
    """Full pipeline on ``preds`` (path or list of rows) -> result dict; writes the file set to
    ``out_dir`` when given. Result keys: ``summary`` (ADE/FDE), ``rfs`` (compute_rfs output),
    ``frame_mean``, ``leaderboard``, ``per_frame`` ({clip_id: score}), ``n_frames``, ``skipped``."""
    rows = load_preds(preds) if isinstance(preds, str) else list(preds)
    cmap = load_cluster_map(cluster_json)
    prep = prepare_inputs(rows, src_dt=src_dt, has_origin=has_origin, cluster_map=cmap)
    if not prep["rater_inputs"]:
        raise ValueError(f"no rated frames among {len(rows)} predictions (skipped: {prep['skipped'][:3]})")
    rfs = compute_rfs(prep["rater_inputs"], cmap)
    score_by_clip = {p["clip_id"]: p["score"] for p in rfs["per_frame"]}
    for s in prep["per_sample"]:
        s["rfs"] = score_by_clip.get(s["clip_id"])
    summary = dict(prep["summary"])
    summary["rater_feedback"] = {
        "computed": True, "n_rated_frames": rfs["n_rated_frames"],
        "frame_mean_rater_feedback_score": rfs["frame_mean_rater_feedback_score"],
        "leaderboard_overall_rater_feedback_score": rfs["leaderboard_overall_rater_feedback_score"],
        "per_cluster": rfs["per_cluster"],
        "source": "vendored waymo-open-dataset rater_feedback_utils; "
                  "per-cluster from val_sequence_name_to_scenario_cluster.json",
    }
    out = {"tag": tag, "n_frames": summary["n_frames"], "summary": summary, "rfs": rfs,
           "frame_mean": rfs["frame_mean_rater_feedback_score"],
           "leaderboard": rfs["leaderboard_overall_rater_feedback_score"],
           "per_cluster": rfs["per_cluster"], "per_frame": score_by_clip, "skipped": prep["skipped"]}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"results": [{"id": f"{r['scene_id']}-{r['frame_id']}", "pred_xy": r.get("pred_xy")} for r in rows]},
                  open(os.path.join(out_dir, "results_for_official.json"), "w"))
        json.dump(prep["rater_inputs"], open(os.path.join(out_dir, "rater_feedback_inputs.json"), "w"))
        with open(os.path.join(out_dir, "per_sample_results.jsonl"), "w") as f:
            for s in prep["per_sample"]:
                f.write(json.dumps(s) + "\n")
        json.dump(summary, open(os.path.join(out_dir, "metrics_summary.json"), "w"), indent=1)
        res = {k: v for k, v in rfs.items() if k != "per_frame"}
        res["per_frame"] = rfs["per_frame"]
        json.dump(res, open(os.path.join(out_dir, "rater_feedback_results.json"), "w"), indent=1)
        out["out_dir"] = out_dir
    if verbose:
        print_rfs(out)
    return out


def print_rfs(res: dict) -> None:
    s = res["summary"]
    print(f"[rfs{(':' + res['tag']) if res.get('tag') else ''}] n={s['n_frames']} missing_pred={s['n_missing_pred']} "
          f"skipped={s['n_skipped']} ADE1={s['waymo_ade_1s']:.3f} ADE3={s['waymo_ade_3s']:.3f} "
          f"ADE5={s['waymo_ade_5s']:.3f} FDE5={s['waymo_fde_5s']:.3f}", flush=True)
    lb = res["leaderboard"]
    print(f"      RFS frame-mean={res['frame_mean']:.4f}  leaderboard(cluster-mean)={lb if lb is None else round(lb, 4)}"
          f"  (rated={res['rfs']['n_rated_frames']})", flush=True)
    for c, d in res["per_cluster"].items():
        print(f"      {c:>28s} n={d['n']:>4d} RFS={d['mean_RFS']:.4f}")


# ======================================================================================
# reference rows
# ======================================================================================
def rfs_frame_ids(root: str = RFS_DIR) -> List[Tuple[str, str, str]]:
    """``[(scene_id, frame_id, frame.json path)]`` of all rated val frames in ``root``."""
    out = []
    for b in sorted(os.listdir(root)):
        d = os.path.join(root, b)
        fp = os.path.join(d, "frame.json")
        if b.startswith("._") or not os.path.isdir(d) or not os.path.isfile(fp):
            continue
        name = b[5:] if b.startswith("DONE_") else b
        sid, _, fid = name.rpartition("-")
        out.append((sid, fid, fp))
    return out


def reference_pred(kind: str, frame: dict) -> Optional[List[List[float]]]:
    """One reference trajectory (20x2 list) for ``frame``; ``None`` if the GT is unusable."""
    fs = frame.get("future_states") or {}
    px, py = fs.get("pos_x") or [], fs.get("pos_y") or []
    if min(len(px), len(py)) < FUT_STEPS:
        return None
    gt = np.stack([px[:FUT_STEPS], py[:FUT_STEPS]], axis=1).astype(np.float64)
    if kind == "gt":
        return gt.tolist()
    if kind == "static":
        return np.zeros((FUT_STEPS, 2)).tolist()
    if kind == "cv":
        return TC.cv_extrapolation(frame).tolist()
    if kind == "ceiling":
        pts5 = [(round(x, 1), round(y, 1)) for x, y in TC.points_1hz(gt)]
        return TC.upsample_pchip(pts5, TC.hist_xy_from_frame(frame)).tolist()
    raise ValueError(f"unknown reference kind {kind!r}; expected one of {REFERENCE_KINDS}")


def reference_preds(kind: str, frames: Optional[Sequence[str]] = None) -> List[dict]:
    """Rows ``{scene_id, frame_id, pred_xy}`` of a reference row for the given frame dirs (default: all RFS frames)."""
    items: List[Tuple[str, str, str]]
    if frames is None:
        items = rfs_frame_ids()
    else:
        items = []
        for fdir in frames:
            b = os.path.basename(fdir.rstrip("/"))
            name = b[5:] if b.startswith("DONE_") else b
            sid, _, fid = name.rpartition("-")
            items.append((sid, fid, os.path.join(fdir, "frame.json")))
    rows = []
    for sid, fid, fp in items:
        frame = json.load(open(fp))
        sid = str(frame.get("scene_id") or sid)
        fid = str(frame.get("frame_id") or fid)
        rows.append({"scene_id": sid, "frame_id": fid, "pred_xy": reference_pred(kind, frame)})
    return rows


# ======================================================================================
# CLI
# ======================================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="official Waymo RFS + ADE/FDE on rfs_preds.json (frame-mean + cluster-mean)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preds", help="rfs_preds.json: [{scene_id, frame_id, pred_xy(20x[x,y])}] or {'results': [...]}")
    g.add_argument("--reference", choices=REFERENCE_KINDS, help="score a reference row instead of a prediction file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src-dt", type=float, default=DT, help="time step of the given points (0.25 = 20-pt official grid)")
    ap.add_argument("--has-origin", action="store_true", help="pred[0] is the current position (0,0) -> dropped")
    ap.add_argument("--cluster-json", default=CLUSTER_JSON)
    ap.add_argument("--subset", type=int, default=None, help="reference rows: first N RFS frames only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.reference:
        rows = reference_preds(args.reference)
        if args.subset:
            rows = rows[:args.subset]
        tag = args.tag or args.reference
    else:
        rows = load_preds(args.preds)
        if args.subset:
            rows = rows[:args.subset]
        tag = args.tag
    run_rfs(rows, args.out_dir, cluster_json=args.cluster_json, src_dt=args.src_dt, has_origin=args.has_origin, tag=tag)
    print(f"[run_rfs] done -> {args.out_dir}/rater_feedback_results.json + metrics_summary.json", flush=True)


if __name__ == "__main__":
    main()
