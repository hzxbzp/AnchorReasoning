#!/usr/bin/env python3
"""Render one MP4 per scene showing the projected future trajectory and the intent label.

For every scene folder under --data_dir the script walks the frame folders in
frame order, draws the ego future trajectory (green, projected onto the stitched
panorama through the camera calibration) plus an "Intent: ..." banner at the top
of the image, and encodes the frames into a single video per scene.

Expected layout:

    <data_dir>/
        <scene_folder>/
            <frame_folder>/
                frame.json
                panorama_geo.png   (any *.png / *.jpg in the folder also works)
            ...
        ...

Usage:
    python -m data_preparation.visualization.generate_trajectory_videos \
        --data_dir <data-root>/p<group>
    python -m data_preparation.visualization.generate_trajectory_videos \
        --data_dir <data-root>/p<group> --output_dir <out-dir> --fps 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the camera projection of the interactive visualizer
from data_preparation.visualization.trajectory_visualizer import CameraProjector  # noqa: E402


def _natural_sort_key(path: Path) -> Tuple[int, str]:
    """Sort folder names naturally so 001, 002, ..., 010 keep their order."""
    name = path.name
    # Prefer the last numeric chunk of the name (e.g. xxx-078 -> 78)
    parts = re.split(r"[-_\s]+", name)
    for part in reversed(parts):
        if part.isdigit():
            return (0, int(part))
    return (1, name)


def _get_sorted_frame_folders(video_folder: Path) -> List[Path]:
    """Return every frame folder holding a frame.json, in playback order."""
    frame_folders = []
    for item in video_folder.iterdir():
        if not item.is_dir():
            continue
        if (item / "frame.json").exists():
            frame_folders.append(item)
    return sorted(frame_folders, key=_natural_sort_key)


def _find_image_path(frame_folder: Path) -> Optional[Path]:
    """Find one image in the frame folder, preferring panorama_geo.png."""
    for ext in [".png", ".jpg", ".jpeg"]:
        candidates = list(frame_folder.glob(f"*{ext}"))
        if candidates:
            # Prefer the geo-referenced panorama
            for p in candidates:
                if "panorama" in p.name.lower() or "geo" in p.name.lower():
                    return p
            return candidates[0]
    return None


def _draw_trajectory_and_intent(
    image: np.ndarray,
    frame_data: Dict[str, Any],
) -> np.ndarray:
    """Draw the future trajectory and the intent banner on a BGR image in place."""
    h, w = image.shape[:2]
    calibrations = frame_data.get("camera_calibrations", {})
    intent = frame_data.get("intent", "N/A")

    # Intent banner: centred at the top of the image, easy to read while playing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    text = f"Intent: {intent}"
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 24, 16
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x1 = (w - box_w) // 2
    y1 = 10
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text_x = (w - tw) // 2
    text_y = y1 + pad_y + th
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    if not calibrations:
        return image

    projector = CameraProjector(calibrations, w, h)
    future_states = frame_data.get("future_states", {})

    # Future trajectory (green)
    if "pos_x" in future_states and "pos_y" in future_states:
        pos_x = future_states["pos_x"]
        pos_y = future_states["pos_y"]
        pos_z = future_states.get("pos_z", [0] * len(pos_x))
        future_points = []
        for x, y, z in zip(pos_x, pos_y, pos_z):
            u, v, _ = projector.project_point(np.array([x, y, z]))
            if u is not None and v is not None and 0 <= u < w and 0 <= v < h:
                future_points.append((int(u), int(v)))
        for i, pt in enumerate(future_points):
            intensity = int(255 * (1 - i / max(len(future_points), 1)))
            color = (0, 255, intensity)  # BGR green
            cv2.circle(image, pt, 10, color, -1)
            cv2.circle(image, pt, 10, (0, 100, 0), 2)
            if i > 0:
                cv2.line(image, future_points[i - 1], pt, (0, 200, 0), 4)

    return image


def _collect_video_folders(data_dir: Path) -> List[Path]:
    """Collect every scene folder under data_dir that has at least one frame folder."""
    video_folders = []
    for item in sorted(data_dir.iterdir()):
        if not item.is_dir():
            continue
        frame_folders = _get_sorted_frame_folders(item)
        if frame_folders:
            video_folders.append(item)
    return video_folders


def generate_video_for_folder(
    video_folder: Path,
    output_path: Path,
    fps: int = 10,
) -> bool:
    """Encode one scene folder into a video at output_path.

    Frames are written in the order returned by _get_sorted_frame_folders.
    """
    frame_folders = _get_sorted_frame_folders(video_folder)
    if not frame_folders:
        print(f"  [Skip] No valid frames: {video_folder}")
        return False

    # The first frame fixes the video size
    first_folder = frame_folders[0]
    first_image_path = _find_image_path(first_folder)
    if not first_image_path:
        print(f"  [Skip] No image found: {first_folder}")
        return False

    first_frame = cv2.imread(str(first_image_path))
    if first_frame is None:
        print(f"  [Skip] Unreadable image: {first_image_path}")
        return False

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"  [Fail] Cannot create video: {output_path}")
        return False

    ok_count = 0
    for i, frame_folder in enumerate(frame_folders):
        image_path = _find_image_path(frame_folder)
        if not image_path:
            continue
        frame_json_path = frame_folder / "frame.json"
        if not frame_json_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        try:
            with open(frame_json_path, "r", encoding="utf-8") as f:
                frame_data = json.load(f)
        except Exception:
            continue

        # Every frame must have the same size
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        _draw_trajectory_and_intent(image, frame_data)
        writer.write(image)
        ok_count += 1

    writer.release()
    if ok_count == 0:
        output_path.unlink(missing_ok=True)
        print(f"  [Fail] No frame written: {video_folder}")
        return False

    print(f"  [OK] {output_path.name} ({ok_count} frames)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render videos with the projected future trajectory and the intent label"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root holding scene folders: <data_dir>/<scene>/<frame>/frame.json + image",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write the videos (default: <data_dir>/trajectory_videos)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate of the output videos (default: 10)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"Directory does not exist: {data_dir}")
        return 1

    output_dir = args.output_dir or (data_dir / "trajectory_videos")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_folders = _collect_video_folders(data_dir)
    if not video_folders:
        print(f"No scene folder (with frame.json subfolders) found under {data_dir}")
        return 1

    print(f"{len(video_folders)} scene folders, output directory: {output_dir}\n")

    success = 0
    for vf in video_folders:
        # Name each video after its scene folder to avoid collisions
        out_name = f"{vf.name}.mp4"
        out_path = output_dir / out_name
        print(f"Generating: {vf.name} -> {out_path.name}")
        if generate_video_for_folder(vf, out_path, fps=args.fps):
            success += 1

    print(f"\nDone: {success}/{len(video_folders)} videos generated")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
