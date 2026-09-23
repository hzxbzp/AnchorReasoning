#!/usr/bin/env python3
"""Render an annotated review video: one composite frame per annotated frame folder.

Each video frame is laid out as:
  - top left:  the panorama with segmentation masks, object boxes, the projected
               ego future trajectory and the intent banner;
  - top right: the panorama_geo.json annotations rendered as readable text;
  - bottom:    the past/future trajectory in bird's-eye view and the past speed.

--data_dir is either a root holding scene folders, or a single scene folder whose
subfolders are frame folders (frame.json + image + panorama_geo.json). One video
is written per scene.

Usage:
  python -m data_preparation.visualization.generate_annotated_video \
      --data_dir <data-root>/p<group>
  python -m data_preparation.visualization.generate_annotated_video \
      --data_dir <data-root>/p<group>/<scene> --output out.mp4 --fps 10
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

from data_preparation.visualization.trajectory_visualizer import (  # noqa: E402
    CameraProjector,
    decode_rle_mask,
    format_panorama_geo_for_display,
    load_coco_annotations,
    MASK_COLORS,
)


def _natural_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    parts = re.split(r"[-_\s]+", name)
    for part in reversed(parts):
        if part.isdigit():
            return (0, int(part))
    return (1, name)


def _get_sorted_frame_folders(video_folder: Path) -> List[Path]:
    frame_folders = []
    for item in video_folder.iterdir():
        if not item.is_dir():
            continue
        if (item / "frame.json").exists():
            frame_folders.append(item)
    return sorted(frame_folders, key=_natural_sort_key)


def _collect_video_folders(data_dir: Path) -> List[Path]:
    """Return list of video folders. If data_dir itself contains frame folders, treat it as single video."""
    video_folders = []
    for item in sorted(data_dir.iterdir()):
        if not item.is_dir():
            continue
        frame_folders = _get_sorted_frame_folders(item)
        if frame_folders:
            video_folders.append(item)
    if not video_folders:
        frame_folders = _get_sorted_frame_folders(data_dir)
        if frame_folders:
            video_folders = [data_dir]
    return video_folders


def _find_image_path(frame_folder: Path) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg"]:
        candidates = list(frame_folder.glob(f"*{ext}"))
        if candidates:
            for p in candidates:
                if "panorama" in p.name.lower() or "geo" in p.name.lower():
                    return p
            return candidates[0]
    return None


def _draw_main_image(
    image: np.ndarray,
    frame_data: Dict[str, Any],
    frame_folder: Path,
) -> np.ndarray:
    """Draw bbox, mask, future trajectory, and intent (center) on image. BGR in/out."""
    image = image.copy()
    h, w = image.shape[:2]
    calibrations = frame_data.get("camera_calibrations", {})
    intent = frame_data.get("intent", "N/A")

    # COCO masks (blend first), then draw bbox and labels on top so they stay clear
    bbox_labels = []  # (bbox xyxy, color_bgr, label) drawn after blend
    coco_path = frame_folder / "panorama_geo_sam2_coco.json"
    if coco_path.exists():
        annotations, categories = load_coco_annotations(coco_path)
        mask_overlay = image.copy()
        for i, ann in enumerate(annotations):
            if "segmentation" not in ann:
                continue
            seg = ann["segmentation"]
            if isinstance(seg, dict) and "counts" in seg:
                mask = decode_rle_mask(seg)
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                color_bgr = MASK_COLORS[i % len(MASK_COLORS)][::-1]  # RGB -> BGR
                mask_indices = mask > 0
                mask_overlay[mask_indices] = color_bgr
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cat_id = ann.get("category_id", 0)
                    label = ann.get("label", categories.get(cat_id, f"Obj{i+1}"))
                    bbox_labels.append(((x, y, x + bw, y + bh), color_bgr, str(label)))
        image = cv2.addWeighted(mask_overlay, 0.3, image, 0.7, 0)
        # Draw bbox and label on top of blended image so they are clearly visible
        for (x1, y1, x2, y2), color_bgr, label in bbox_labels:
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)
            cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr, 2, cv2.LINE_AA)

    # Intent banner: horizontally centred near the top, with a fixed top margin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3
    text = f"Intent: {intent}"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 28, 18
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x1 = (w - box_w) // 2
    y1 = 20
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text_x = (w - tw) // 2
    text_y = y1 + pad_y + th
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Ego future trajectory (green, clearly visible)
    if calibrations and "future_states" in frame_data:
        fs = frame_data["future_states"]
        if "pos_x" in fs and "pos_y" in fs:
            projector = CameraProjector(calibrations, w, h)
            pos_x, pos_y = fs["pos_x"], fs["pos_y"]
            pos_z = fs.get("pos_z", [0] * len(pos_x))
            future_points = []
            for x, y, z in zip(pos_x, pos_y, pos_z):
                u, v, _ = projector.project_point(np.array([x, y, z]))
                if u is not None and v is not None and 0 <= u < w and 0 <= v < h:
                    future_points.append((int(u), int(v)))
            for i, pt in enumerate(future_points):
                intensity = int(255 * (1 - i / max(len(future_points), 1)))
                cv2.circle(image, pt, 12, (0, 255, intensity), -1)
                cv2.circle(image, pt, 12, (0, 120, 0), 2)
                if i > 0:
                    cv2.line(image, future_points[i - 1], pt, (0, 220, 0), 5)
    return image


def _render_text_panel(text: str, width: int, height: int) -> np.ndarray:
    """Render multi-line text to a BGR image (white bg, black text)."""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    x0, y0 = 12, 28
    for line in text.split("\n"):
        if y0 > height - 25:
            break
        cv2.putText(panel, line, (x0, y0), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y0 += line_height
    return panel


def _render_trajectory_chart(past_states: Dict, future_states: Dict, width: int, height: int) -> np.ndarray:
    """Render past/future trajectory X-Y plot to BGR image using matplotlib."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use fixed dpi and calculate figsize to get exact output dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    if "pos_x" in past_states and "pos_y" in past_states:
        ax.plot(past_states["pos_x"], past_states["pos_y"], "o-", color="#FF6600", linewidth=2, markersize=4, label="Past")
    if "pos_x" in future_states and "pos_y" in future_states:
        ax.plot(future_states["pos_x"], future_states["pos_y"], "o-", color="#00FF00", linewidth=2, markersize=4, label="Future")
    ax.plot(0, 0, "o", color="#00FFFF", markersize=10, label="Current")
    ax.set_xlabel("X (forward, m)", fontsize=9)
    ax.set_ylabel("Y (left, m)", fontsize=9)
    ax.set_title("Past / Future Trajectory", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-15, 30)
    ax.set_ylim(-10, 10)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    # Note: removed set_aspect("equal") to avoid squeezing the plot in a wide chart
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    # Do NOT use bbox_inches="tight" - it changes output size; use exact figsize
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
    return img


def _render_speed_chart(past_states: Dict, width: int, height: int) -> np.ndarray:
    """Render past speed (vel_x, vel_y, magnitude) to BGR image."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use fixed dpi and calculate figsize to get exact output dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    if "vel_x" in past_states and "vel_y" in past_states:
        vel_x = np.array(past_states["vel_x"])
        vel_y = np.array(past_states["vel_y"])
        speed = np.sqrt(vel_x**2 + vel_y**2)
        n = len(speed)
        t = np.linspace(-4, 0, n)
        ax.plot(t, vel_x, "b-", linewidth=2, label="Vel X")
        ax.plot(t, vel_y, "g-", linewidth=2, label="Vel Y")
        ax.plot(t, speed, "r--", linewidth=2, label="Speed")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Velocity (m/s)", fontsize=9)
    ax.set_title("Past Speed", fontsize=10)
    ax.set_xlim(-4, 0)
    ax.set_ylim(-5, 25)
    ax.tick_params(axis='both', labelsize=8)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    # Do NOT use bbox_inches="tight" - it changes output size; use exact figsize
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
    return img


def build_frame(
    frame_folder: Path,
    main_img_h: int,
    main_img_w: int,
    panel_width: int,
    chart_w: int,
    chart_h: int,
) -> Optional[np.ndarray]:
    """Build one composite frame. Returns BGR image or None on failure."""
    image_path = _find_image_path(frame_folder)
    frame_json_path = frame_folder / "frame.json"
    if not image_path or not frame_json_path.exists():
        return None
    try:
        with open(frame_json_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)
    except Exception:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    # Draw on ORIGINAL image first (bbox/mask/trajectory coords are based on original size)
    image = _draw_main_image(image, frame_data, frame_folder)
    # Then resize to target size
    main_img = cv2.resize(image, (main_img_w, main_img_h), interpolation=cv2.INTER_LINEAR)

    # Panorama panel (right)
    panorama_path = frame_folder / "panorama_geo.json"
    if panorama_path.exists():
        try:
            with open(panorama_path, "r", encoding="utf-8") as f:
                panorama_data = json.load(f)
            text = format_panorama_geo_for_display(panorama_data)
        except Exception:
            text = "(Load error)"
    else:
        text = "(No panorama_geo.json)"
    panel = _render_text_panel(text, panel_width, main_img_h)
    top_row = np.hstack([main_img, panel])

    # Bottom: trajectory + speed charts (past only for "history")
    past_states = frame_data.get("past_states", {})
    future_states = frame_data.get("future_states", {})
    traj_img = _render_trajectory_chart(past_states, future_states, chart_w, chart_h)
    speed_img = _render_speed_chart(past_states, chart_w, chart_h)
    bottom_row = np.hstack([traj_img, speed_img])

    # Ensure same total width
    top_w, bottom_w = top_row.shape[1], bottom_row.shape[1]
    if top_w > bottom_w:
        pad = np.ones((chart_h, top_w - bottom_w, 3), dtype=np.uint8) * 255
        bottom_row = np.hstack([bottom_row, pad])
    elif bottom_w > top_w:
        pad = np.ones((main_img_h, bottom_w - top_w, 3), dtype=np.uint8) * 255
        top_row = np.hstack([top_row, pad])
    frame = np.vstack([top_row, bottom_row])
    return frame


def generate_video(
    video_folder: Path,
    output_path: Path,
    fps: int = 10,
    main_height: int = 720,
    panel_width: int = 380,
    chart_height: int = 260,
) -> bool:
    frame_folders = _get_sorted_frame_folders(video_folder)
    if not frame_folders:
        print(f"  [Skip] No frame folders in {video_folder}")
        return False
    # Get main image size from first frame (keep aspect or fixed height)
    first_img_path = _find_image_path(frame_folders[0])
    if not first_img_path:
        print(f"  [Skip] No image in {frame_folders[0]}")
        return False
    sample = cv2.imread(str(first_img_path))
    if sample is None:
        return False
    h0, w0 = sample.shape[:2]
    main_img_w = int(w0 * main_height / h0)
    main_img_h = main_height
    chart_w = (main_img_w + panel_width) // 2
    total_w = main_img_w + panel_width
    total_h = main_img_h + chart_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        print(f"  [Fail] Cannot create {output_path}")
        return False
    ok = 0
    for i, ff in enumerate(frame_folders):
        frame = build_frame(ff, main_img_h, main_img_w, panel_width, chart_w, chart_height)
        if frame is not None:
            if frame.shape[1] != total_w or frame.shape[0] != total_h:
                frame = cv2.resize(frame, (total_w, total_h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            ok += 1
    writer.release()
    if ok == 0:
        output_path.unlink(missing_ok=True)
        print(f"  [Fail] No frames written")
        return False
    print(f"  [OK] {output_path.name} ({ok} frames)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate annotated video (bbox, mask, trajectory, intent, panorama, charts)")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root dir or single scene folder (frame folders with frame.json, image, panorama_geo.json)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output video path (default: an annotated_videos folder named after each scene)")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    parser.add_argument("--main_height", type=int, default=720, help="Main image height in pixels")
    parser.add_argument("--panel_width", type=int, default=380, help="Panorama text panel width")
    parser.add_argument("--chart_height", type=int, default=320, help="Height of each bottom chart")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"Directory does not exist: {data_dir}")
        return 1
    video_folders = _collect_video_folders(data_dir)
    if not video_folders:
        print(f"No video folders (with frame.json subdirs) found under {data_dir}")
        return 1

    output_dir = data_dir if len(video_folders) > 1 else data_dir.parent
    output_dir = output_dir / "annotated_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        out_path = args.output.resolve()
        if out_path.suffix.lower() != ".mp4":
            out_path = out_path.with_suffix(".mp4")
        if len(video_folders) == 1:
            generate_video(video_folders[0], out_path, fps=args.fps, main_height=args.main_height, panel_width=args.panel_width, chart_height=args.chart_height)
        else:
            for vf in video_folders:
                op = out_path.parent / f"{out_path.stem}_{vf.name}{out_path.suffix}"
                generate_video(vf, op, fps=args.fps, main_height=args.main_height, panel_width=args.panel_width, chart_height=args.chart_height)
        return 0

    success = 0
    for vf in video_folders:
        out_name = f"{vf.name}.mp4"
        out_path = output_dir / out_name
        print(f"Generating: {vf.name} -> {out_path.name}")
        if generate_video(vf, out_path, fps=args.fps, main_height=args.main_height, panel_width=args.panel_width, chart_height=args.chart_height):
            success += 1
    print(f"Done: {success}/{len(video_folders)} videos")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
