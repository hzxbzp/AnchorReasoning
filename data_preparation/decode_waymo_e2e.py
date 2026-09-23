"""Decode Waymo End-to-End Driving tfrecord shards into per-frame folders.

The same script handles both dataset splits (``--split train`` / ``--split
val``). Every E2EDFrame record becomes one frame folder::

    <out>/p<group>/<scene_id>/<scene_id>-<frame_id>/
        panorama_geo.png   FRONT_LEFT | FRONT | FRONT_RIGHT, stitched
        frame.json         intent, past/future ego states, preference trajectories

Notes
-----
* The three forward camera images (972x1079 each) are undistorted and
  reprojected onto one common forward-facing virtual pinhole, which yields a
  2916x1079 panorama.
* ``frame.json`` holds ``scene_id``, ``frame_id``, ``past_states``,
  ``future_states``, ``intent``, ``preference_trajectories`` and a placeholder
  ``ego_behavior``. The raw records carry no ego-behaviour labels, and the
  camera calibrations (intrinsics / extrinsics) are intentionally not written.
* ``preference_trajectories`` and the ego trajectory states are copied verbatim
  from the proto; they are the fields used at evaluation time.
* Scenes are grouped 100 per ``p<N>`` folder, in order of first appearance.
* The shards are read with a small pure-python TFRecord reader, so neither
  TensorFlow nor the official dataset wheel is required: decoding needs only
  the ``protobuf`` runtime plus a generated descriptor set, which one command
  produces::

      bash data_preparation/make_wod_desc.sh

Usage::

    python -m data_preparation.decode_waymo_e2e --split val --src <shard-dir>
    python -m data_preparation.decode_waymo_e2e --split train --src <shard-dir> --out <out-root>
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # a missing descriptor set is a setup problem: report it, not a traceback
    from data_preparation.wod_proto import E2EDFrame, camera_name, intent_name  # noqa: E402
except RuntimeError as err:  # raised by wod_proto when no proto source resolves
    raise SystemExit(f"error: {err}") from None
from training.core.paths import DATA_ROOT, VAL_ROOT  # noqa: E402

PANORAMA_CAMERAS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")
SCENES_PER_GROUP = 100

# Permutation mapping the Waymo camera frame (x=out of lens, y=left, z=up) to
# the standard pinhole frame (x=right, y=down, z=forward).
_P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
_PINV = np.linalg.inv(_P)

# Trajectory fields, in the order frame.json lists them.
_STATE_FIELDS = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y")


def iter_tfrecord(path: Path):
    """Yield raw record bytes from a (uncompressed) TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # crc32 of length (ignored)
            data = f.read(length)
            f.read(4)  # crc32 of data (ignored)
            yield data


def states_to_dict(states) -> dict:
    """Serialize an EgoTrajectoryStates, keeping only populated fields."""
    out = {}
    for field in _STATE_FIELDS:
        values = list(getattr(states, field))
        if values:
            out[field] = values
    return out


def preference_traj_to_dict(traj) -> dict:
    d = states_to_dict(traj)
    d["preference_score"] = traj.preference_score
    return d


def build_panorama(frame, out_w: int = 2916, out_h: int = 1079) -> Image.Image:
    """Undistort -> project all three cameras onto ONE common forward-facing,
    leveled virtual pinhole in the vehicle frame (no per-strip yaw spacing).

    Because the three cameras share a single image plane, the same object seen
    by two cameras lands at the same panorama pixel, so seams are continuous
    (no offset). The trade-off is that content far off-axis stretches and
    anything beyond the virtual camera's FOV is cropped.
    """
    images, calib = {}, {}
    for img in frame.images:
        name = camera_name(img.name)
        if name in PANORAMA_CAMERAS:
            images[name] = np.array(Image.open(io.BytesIO(img.image)).convert("RGB"))
    for c in frame.context.camera_calibrations:
        name = camera_name(c.name)
        if name in PANORAMA_CAMERAS:
            calib[name] = c
    missing = [c for c in PANORAMA_CAMERAS if c not in images or c not in calib]
    if missing:
        raise ValueError(f"missing camera images/calibration: {missing}")

    # virtual pinhole: forward, leveled, focal = FRONT's fx, centred output.
    f_pano = calib["FRONT"].intrinsic[0]
    cx_p, cy_p = out_w / 2.0, calib["FRONT"].intrinsic[3]
    uu, vv = np.meshgrid(np.arange(out_w), np.arange(out_h))
    # ray in vehicle frame (x fwd, y left, z up) for each output pixel
    std = np.stack([(uu - cx_p) / f_pano, (vv - cy_p) / f_pano, np.ones_like(uu)], axis=-1)
    ray = std @ _PINV.T          # (H,W,3) vehicle-frame direction
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    best_cos = np.full((out_h, out_w), -1.0)   # pick the most on-axis camera
    for cam in PANORAMA_CAMERAS:
        src = images[cam]
        h, w = src.shape[:2]
        c = calib[cam]
        fx, fy, cx, cy = c.intrinsic[0], c.intrinsic[1], c.intrinsic[2], c.intrinsic[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([*c.intrinsic[4:9]], dtype=np.float64)
        undist = cv2.undistort(src, K, dist, None, K)

        R_cam = np.array(c.extrinsic.transform).reshape(4, 4)[:3, :3]
        d_cam = ray @ R_cam            # ray expressed in camera frame (= ray @ (R^T)^T)
        depth = d_cam[..., 0]
        std_cam = d_cam @ _P.T         # to standard pinhole frame
        zc = std_cam[..., 2]
        us = fx * (std_cam[..., 0] / zc) + cx
        vs = fy * (std_cam[..., 1] / zc) + cy
        valid = (depth > 1e-6) & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        # on-axis score = forward component (depth), higher = closer to centre
        better = valid & (depth > best_cos)
        mapx = np.where(better, us, -1).astype(np.float32)
        mapy = np.where(better, vs, -1).astype(np.float32)
        sampled = cv2.remap(undist, mapx, mapy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out[better] = sampled[better]
        best_cos[better] = depth[better]
    return Image.fromarray(out)


def build_frame_json(efr: "E2EDFrame", scene_id: str, frame_id: str) -> dict:
    # camera_calibrations (intrinsics + extrinsics) are intentionally omitted;
    # ego_behavior is a placeholder because the records do not carry it.
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "future_states": states_to_dict(efr.future_states),
        "past_states": states_to_dict(efr.past_states),
        "intent": intent_name(efr.intent),
        "preference_trajectories": [
            preference_traj_to_dict(t) for t in efr.preference_trajectories
        ],
        "ego_behavior": {"longitudinal": "none", "lateral": "none"},
    }


def process_record(raw: bytes, out_root: Path, group_name: str) -> Path:
    efr = E2EDFrame()
    efr.ParseFromString(raw)
    scene_id, frame_id = efr.frame.context.name.rsplit("-", 1)

    frame_dir = out_root / group_name / scene_id / f"{scene_id}-{frame_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    build_panorama(efr.frame).save(frame_dir / "panorama_geo.png")
    (frame_dir / "frame.json").write_text(
        json.dumps(build_frame_json(efr, scene_id, frame_id), indent=2)
    )
    return frame_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--split",
        required=True,
        choices=("train", "val"),
        help="dataset split to decode; picks the default shard pattern and output root",
    )
    ap.add_argument(
        "--src",
        required=True,
        help="directory holding the tfrecord shards of that split",
    )
    ap.add_argument(
        "--input-glob",
        default=None,
        help="shard filename pattern inside --src (default: '<split>*.tfrecord*')",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output root (default: $WAYMO_TRAIN_ROOT for train, $WAYMO_VAL_ROOT for val)",
    )
    ap.add_argument("--max-scenes", type=int, default=None, help="stop after N scenes")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--shards", type=int, default=None, help="only first N shards")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out or (DATA_ROOT if args.split == "train" else VAL_ROOT))
    pattern = args.input_glob or f"{args.split}*.tfrecord*"
    shards = sorted(src.glob(pattern))
    if args.shards is not None:
        shards = shards[: args.shards]
    if not shards:
        raise SystemExit(f"no tfrecord shards matching {pattern!r} under {src}")
    print(f"{len(shards)} shard(s) from {src} -> {out_root}")

    scene_ordinal = {}  # scene_id -> 0-based index of first appearance
    n_frames = 0
    for shard in shards:
        for raw in iter_tfrecord(shard):
            # Peek scene id to assign a group folder.
            efr = E2EDFrame()
            efr.ParseFromString(raw)
            scene_id = efr.frame.context.name.rsplit("-", 1)[0]
            if scene_id not in scene_ordinal:
                if args.max_scenes is not None and len(scene_ordinal) >= args.max_scenes:
                    print(f"reached max-scenes={args.max_scenes}")
                    return
                scene_ordinal[scene_id] = len(scene_ordinal)
            group_name = f"p{scene_ordinal[scene_id] // SCENES_PER_GROUP + 1}"

            frame_dir = process_record(raw, out_root, group_name)
            n_frames += 1
            print(f"[{n_frames}] {frame_dir}")

            if args.max_frames is not None and n_frames >= args.max_frames:
                print(f"reached max-frames={args.max_frames}")
                return

    print(f"done: {n_frames} frames, {len(scene_ordinal)} scenes")


if __name__ == "__main__":
    main()
