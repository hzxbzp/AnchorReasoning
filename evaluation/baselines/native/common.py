"""Environment-agnostic helpers shared by the native runners (stdlib + numpy only; py3.9 compatible).

The runners load this file by path (``importlib``) so that each released model can run inside its own
virtual environment without ever importing the ``training`` package.

Locations come from environment variables. ``ANCHOR_ROOT``, ``WAYMO_VAL_ROOT`` and ``ANCHOR_WEIGHTS``
carry the same names and defaults as ``training.core.paths``; ``ANCHOR_RATED_INDEX`` names the JSONL
index of the rated validation frames and can be overridden per run with ``--index``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# evaluation/baselines/native/common.py -> repository root
ROOT = os.environ.get("ANCHOR_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
VAL_ROOT = os.environ.get("WAYMO_VAL_ROOT", os.path.join(ROOT, "datasets", "waymo_e2e_processed_val"))
RFS_DIR = os.path.join(VAL_ROOT, "rater_feedback_frames")
VAL456_INDEX = os.environ.get("ANCHOR_RATED_INDEX", os.path.join(ROOT, "waymo_e2e_index", "val_rated_only.jsonl"))
WEIGHTS = os.environ.get("ANCHOR_WEIGHTS", os.path.join(ROOT, "weights", "base"))
MAX_NEW_TOKENS = 1024                    # generation cap, identical for every model (see eval_dev)
WAYMO_DT = 0.25
N_FUTURE = 20                            # 5 s @ 4 Hz
T_OUT = np.arange(1, N_FUTURE + 1) * WAYMO_DT


def staged_or(path: str) -> str:
    """``ANCHOR_BASE_PATH`` (node-local copy made by the launcher) if set, else ``path``."""
    s = os.environ.get("ANCHOR_BASE_PATH", "").strip()
    return s if s and os.path.isdir(s) else path


def unique_snapshot(repo_dir: str) -> str:
    snaps = sorted(d for d in (os.path.join(repo_dir, "snapshots", x)
                               for x in os.listdir(os.path.join(repo_dir, "snapshots"))) if os.path.isdir(d))
    if len(snaps) != 1:
        raise FileNotFoundError(f"expected one snapshot under {repo_dir}, found {snaps}")
    return snaps[0]


def load_val456(index: str = VAL456_INDEX, subset: Optional[int] = None) -> List[Dict[str, str]]:
    """The rated validation frames: partition / scene_id / frame_id / fdir (rater_feedback_frames) /
    clip (the partition dir that also holds the neighbouring frames for temporal inputs)."""
    out = []
    for line in open(index):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        sid, fid, part = e["scene_id"], e["frame_id"], e["partition"]
        fdir = os.path.join(RFS_DIR, f"{sid}-{fid}")
        clip = os.path.join(VAL_ROOT, part, sid, f"{sid}-{fid}")
        if not os.path.isfile(os.path.join(fdir, "frame.json")):
            fdir = clip
        if not os.path.isfile(os.path.join(clip, "frame.json")):
            continue
        out.append({"partition": part, "scene_id": sid, "frame_id": fid, "fdir": fdir, "clip": clip,
                    "scene_dir": os.path.join(VAL_ROOT, part, sid)})
    out.sort(key=lambda r: (r["scene_id"], r["frame_id"]))
    return out[:subset] if subset else out


def resample_to_4hz(points: Sequence[Sequence[float]], dt_in: float) -> Optional[List[List[float]]]:
    """Waypoints at t = dt_in, 2 dt_in, ... (ego frame, origin at t = 0) -> 20 x (x, y) @ 0.25 s.

    Linear interpolation through the origin; a horizon shorter than 5 s is HELD at the last point
    (the same 'short predictions are penalised, not extrapolated' rule as ``traj_codec.upsample_pchip``).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2) if points is not None else np.zeros((0, 2))
    if pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
        return None
    t_in = np.concatenate([[0.0], dt_in * np.arange(1, pts.shape[0] + 1)])
    xy = np.concatenate([np.zeros((1, 2)), pts], axis=0)
    out = np.stack([np.interp(T_OUT, t_in, xy[:, 0]), np.interp(T_OUT, t_in, xy[:, 1])], axis=1)
    return out.tolist()


def make_row(fr: Dict[str, str], text: str, pred_xy: Optional[List[List[float]]], n_new_tokens: int,
             hit_cap: bool, prompt_len: Optional[int],
             native: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One row in the ``eval_val456`` chains.json schema (only the keys ``rescore`` reads + native extras)."""
    return {"scene_id": fr["scene_id"], "frame_id": fr["frame_id"], "fdir": fr["fdir"], "task": "s2",
            "text": text or "", "pred_xy": pred_xy, "n_new_tokens": int(n_new_tokens), "hit_cap": bool(hit_cap),
            "prompt_len": prompt_len, "native": native or {}}


class RowSink:
    """Append rows to ``<out>.partial.jsonl`` as they come (resumable) and write the final JSON list."""

    def __init__(self, out_json: str, resume: bool = True):
        self.out_json = out_json
        self.partial = out_json + ".partial.jsonl"
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        self.rows: List[dict] = []
        if resume and os.path.isfile(self.partial):
            for line in open(self.partial):
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.done = {(r["scene_id"], r["frame_id"]) for r in self.rows}
        self._f = open(self.partial, "a")

    def has(self, fr: Dict[str, str]) -> bool:
        return (fr["scene_id"], fr["frame_id"]) in self.done

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.done.add((row["scene_id"], row["frame_id"]))
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> str:
        self._f.close()
        self.rows.sort(key=lambda r: (r["scene_id"], r["frame_id"]))
        json.dump(self.rows, open(self.out_json, "w"), indent=1, ensure_ascii=False)
        return self.out_json


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def summarize(rows: List[dict]) -> Dict[str, Any]:
    tok = [r["n_new_tokens"] for r in rows]
    return {"n": len(rows), "n_pred_xy": sum(1 for r in rows if r.get("pred_xy")),
            "n_hit_cap": sum(1 for r in rows if r.get("hit_cap")),
            "mean_new_tokens": float(np.mean(tok)) if tok else None,
            "n_text": sum(1 for r in rows if (r.get("text") or "").strip())}
