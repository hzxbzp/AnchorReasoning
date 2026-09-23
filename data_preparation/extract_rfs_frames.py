"""Extract the rater-feedback (RFS) frames into one separate folder.

An RFS frame is one whose frame.json has real rater-feedback preference
trajectories: at least one entry in ``preference_trajectories`` carries an
actual trajectory (a ``pos_x`` list). Placeholder frames only have
``{"preference_score": -1.0}`` with no positions.

Each matching frame folder (e.g. ``<scene_id>-<frame_id>/`` with its
panorama_geo.png + frame.json) is copied verbatim into::

    <root>/<dest>/<scene_id>-<frame_id>/

together with ``<root>/<dest>/rfs_index.json``, which lists the extracted
frames and their preference scores.

Usage::

    python -m data_preparation.extract_rfs_frames --dry-run   # just count + list
    python -m data_preparation.extract_rfs_frames             # copy them
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.core.paths import VAL_ROOT  # noqa: E402


def load_frame(frame_json: Path) -> dict | None:
    try:
        return json.loads(frame_json.read_text())
    except Exception:
        return None


def is_rfs(d: dict) -> bool:
    for pt in d.get("preference_trajectories", []):
        if pt.get("pos_x"):  # non-empty trajectory => real rater feedback
            return True
    return False


def index_entry(frame: Path, root: Path, dest_name: str, d: dict) -> dict:
    prefs = d.get("preference_trajectories", [])
    return {
        "scene_id": d.get("scene_id"),
        "frame_id": d.get("frame_id"),
        "frame_name": frame.name,
        "intent": d.get("intent"),
        "num_preference_trajectories": len(prefs),
        "preference_scores": [pt.get("preference_score") for pt in prefs],
        "source_path": str(frame.relative_to(root)),
        "extracted_path": f"{dest_name}/{frame.name}",
    }


def iter_frame_dirs(root: Path):
    for group in sorted(root.glob("p*")):
        if not group.is_dir():
            continue
        for scene in sorted(group.iterdir()):
            if not scene.is_dir():
                continue
            for frame in sorted(scene.iterdir()):
                if frame.is_dir() and (frame / "frame.json").exists():
                    yield frame


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=VAL_ROOT,
                    help="decoded frame root to scan (default: $WAYMO_VAL_ROOT)")
    ap.add_argument("--dest", default="rater_feedback_frames",
                    help="destination folder name under --root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    dest = root / args.dest

    matches = []          # list of (frame_dir, frame_json_dict)
    scanned = 0
    for frame in iter_frame_dirs(root):
        scanned += 1
        d = load_frame(frame / "frame.json")
        if d and is_rfs(d):
            matches.append((frame, d))
        if scanned % 10000 == 0:
            print(f"  scanned {scanned}, RFS so far {len(matches)}")

    print(f"scanned {scanned} frames; RFS frames found: {len(matches)}")
    for frame, _ in matches[:5]:
        print("   e.g.", frame.relative_to(root))

    if args.dry_run:
        return

    dest.mkdir(parents=True, exist_ok=True)

    index = []
    copied = 0
    for frame, d in matches:
        target = dest / frame.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(frame, target)
        index.append(index_entry(frame, root, args.dest, d))
        copied += 1
        if copied % 50 == 0 or copied == len(matches):
            print(f"  copied {copied}/{len(matches)}")

    # sort index for stable, easy lookup
    index.sort(key=lambda e: (e["scene_id"] or "", e["frame_id"] or ""))
    index_path = dest / "rfs_index.json"
    index_path.write_text(json.dumps(
        {"count": len(index), "frames": index}, indent=2, ensure_ascii=False))
    print(f"done: copied {copied} RFS frames -> {dest}")
    print(f"index: {index_path} ({len(index)} entries)")


if __name__ == "__main__":
    main()
