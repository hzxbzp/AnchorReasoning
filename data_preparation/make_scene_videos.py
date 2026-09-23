"""Generate one MP4 preview per scene folder from its panorama_geo.png frames.

Layout (input == output root)::

    <root>/p<N>/<scene_id>/<scene_id>-<frame_id>/panorama_geo.png

For each <scene_id> folder we collect its frame subfolders, sort them by the
numeric frame_id, and encode the panoramas into <scene_id>/<scene_id>.mp4.

Usage::

    python -m data_preparation.make_scene_videos                      # all scenes, 10 fps
    python -m data_preparation.make_scene_videos --fps 10 --jobs 8
    python -m data_preparation.make_scene_videos --root <decoded-root> --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import VAL_ROOT  # noqa: E402

PANO_NAME = "panorama_geo.png"


def list_scene_dirs(root: Path) -> list[Path]:
    scenes = []
    for group in sorted(root.glob("p*")):
        if not group.is_dir():
            continue
        for scene in sorted(group.iterdir()):
            if scene.is_dir():
                scenes.append(scene)
    return scenes


def frame_folders_sorted(scene_dir: Path) -> list[Path]:
    folders = [p for p in scene_dir.iterdir() if p.is_dir()]
    folders.sort(key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    return folders


def build_video(scene_dir: Path, fps: float, codec: str, overwrite: bool) -> str:
    out_path = scene_dir / f"{scene_dir.name}.mp4"
    if out_path.exists() and not overwrite:
        return f"skip (exists) {scene_dir.name}"

    frames = frame_folders_sorted(scene_dir)
    pano_paths = [f / PANO_NAME for f in frames if (f / PANO_NAME).exists()]
    if not pano_paths:
        return f"skip (no frames) {scene_dir.name}"

    first = cv2.imread(str(pano_paths[0]))
    if first is None:
        return f"ERROR unreadable first frame {scene_dir.name}"
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened():
        return f"ERROR VideoWriter failed {scene_dir.name}"

    n = 0
    for p in pano_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        n += 1
    writer.release()
    return f"ok {scene_dir.name}: {n} frames -> {out_path.name}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=VAL_ROOT,
                    help="decoded frame root, train or val "
                         "(default: $WAYMO_VAL_ROOT; pass $WAYMO_TRAIN_ROOT for the training split)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--codec", default="mp4v", help="fourcc, e.g. mp4v / avc1")
    ap.add_argument("--jobs", type=int, default=4, help="parallel scenes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    scenes = list_scene_dirs(root)
    print(f"found {len(scenes)} scene folders under {root}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(build_video, s, args.fps, args.codec, args.overwrite): s
            for s in scenes
        }
        for fut in as_completed(futs):
            done += 1
            msg = fut.result()
            if msg.startswith("ERROR") or done % 20 == 0 or done == len(scenes):
                print(f"[{done}/{len(scenes)}] {msg}")

    print(f"done: {done} scenes processed")


if __name__ == "__main__":
    main()
