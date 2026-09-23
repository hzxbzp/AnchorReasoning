#!/usr/bin/env python3
r"""Render one annotated frame into a single high-resolution poster (PNG).

The poster combines, for one frame folder:
  - the stitched front panorama with segmentation masks, numbered object boxes
    and the ego future (green) / past (orange) trajectory projected onto it;
  - one card per annotated object with its attributes and driving implication,
    color-coded and numbered to match the boxes on the panorama;
  - the frame-level reasoning rationale and the final plan;
  - bird's-eye trajectory, ego speed and ego acceleration charts.

The poster is written as a PNG and, unless --no-show is given, also opened in a
window.

Usage:
    # render one specific frame folder
    python -m data_preparation.visualization.annotation_showcase \
        --frame_dir <data-root>/p<group>/<scene>/<scene>-<frame>

    # or scan a directory and render the frame with the richest annotations
    python -m data_preparation.visualization.annotation_showcase \
        --data_dir <data-root>/p<group>

    # write the PNG only, without opening a window
    python -m data_preparation.visualization.annotation_showcase \
        --frame_dir <frame-dir> --no-show
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# Theme / Palette  (Nord-inspired, clean & high-contrast)
# =============================================================================

BG        = "#ECEFF4"   # page background
CARD      = "#FFFFFF"   # card fill
INK       = "#2E3440"   # primary text (dark slate)
SUBINK    = "#4C566A"   # secondary text
HAIRLINE  = "#D8DEE9"   # card border
BLUE      = "#5E81AC"
TEAL      = "#88C0D0"
GREEN     = "#5B9279"
ORANGE    = "#D08770"
RED       = "#BF616A"
PURPLE    = "#9D7CB8"
YELLOW    = "#EBCB8B"

# Object palette: the box drawn on the panorama and the implication card on the
# right share the same color and the same index.
OBJECT_COLORS = [
    (0xBF, 0x61, 0x6A),  # red
    (0x5E, 0x81, 0xAC),  # blue
    (0x5B, 0x92, 0x79),  # green
    (0xD0, 0x87, 0x70),  # orange
    (0x9D, 0x7C, 0xB8),  # purple
    (0xEB, 0xCB, 0x8B),  # yellow
    (0x88, 0xC0, 0xD0),  # teal
    (0xB4, 0x8E, 0xAD),  # mauve
    (0xA3, 0xBE, 0x8C),  # sage
    (0xD0, 0x8F, 0x6E),  # clay
]


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % rgb


# =============================================================================
# Camera projection (same geometry as the trajectory visualizer)
# =============================================================================

class CameraProjector:
    def __init__(self, calibrations: Dict[str, Any], image_width: int, image_height: int):
        self.calibrations = calibrations
        self.image_width = image_width
        self.image_height = image_height
        self.camera_order = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT']
        self.camera_width = image_width // len(self.camera_order)
        self.camera_offsets = {
            'FRONT_LEFT': 0,
            'FRONT': self.camera_width,
            'FRONT_RIGHT': self.camera_width * 2,
        }

    def _intrinsic(self, cam):
        return np.array(self.calibrations[cam]['intrinsic'])

    def _extrinsic(self, cam):
        return np.array(self.calibrations[cam]['extrinsic']['transform']).reshape(4, 4)

    def project_to_cam(self, p, cam):
        try:
            intr = self._intrinsic(cam)
            extr = self._extrinsic(cam)
        except (KeyError, TypeError):
            return None, None, False
        R, t = extr[:3, :3], extr[:3, 3]
        pc = R.T @ (np.array([p[0], p[1], p[2]]) - t)
        depth = pc[0]
        if depth <= 0.5:
            return None, None, False
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        u = fx * (-pc[1] / depth) + cx
        v = fy * (-pc[2] / depth) + cy
        ok = (0 <= u < self.camera_width) and (0 <= v < self.image_height)
        return u, v, ok

    def project(self, p):
        best = None
        for cam in self.camera_order:
            if cam not in self.calibrations:
                continue
            u, v, ok = self.project_to_cam(p, cam)
            if ok and u is not None:
                pu = u + self.camera_offsets[cam]
                if best is None:
                    best = (pu, v, cam)
                else:
                    cur = abs(u - self.camera_width / 2)
                    bst = abs(best[0] - self.camera_offsets[best[2]] - self.camera_width / 2)
                    if cur < bst:
                        best = (pu, v, cam)
        return best if best else (None, None, None)


# =============================================================================
# Data loading
# =============================================================================

def load_frame(frame_dir: Path) -> Dict[str, Any]:
    frame_data, pano_data = {}, {}
    fj = frame_dir / "frame.json"
    if fj.exists():
        frame_data = json.loads(fj.read_text(encoding="utf-8"))
    pj = frame_dir / "panorama_geo.json"
    if pj.exists():
        pano_data = json.loads(pj.read_text(encoding="utf-8"))
    img_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        c = list(frame_dir.glob(f"panorama_geo{ext}"))
        if c:
            img_path = c[0]
            break
    return {"frame_data": frame_data, "pano_data": pano_data, "image_path": img_path}


def ego_speed_kmh(frame_data: Dict) -> float:
    ps = frame_data.get("past_states", {})
    vx = (ps.get("vel_x") or [0.0])[-1]
    vy = (ps.get("vel_y") or [0.0])[-1]
    return float(np.hypot(vx, vy) * 3.6)


def frame_complexity_score(pano: Dict) -> int:
    anns = pano.get("annotations", [])
    impl = sum(1 for a in anns if a.get("driving_implication"))
    has_rp = bool(pano.get("reason")) and bool(pano.get("final_plan"))
    return impl * 10 + (5 if has_rp else 0)


def auto_pick_frame(data_dir: Path) -> Optional[Path]:
    """Recursively pick the frame under data_dir with the most annotations that
    also carries a reason and a final plan."""
    best, best_score = None, -1
    for pj in data_dir.rglob("panorama_geo.json"):
        try:
            d = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (pj.parent / "frame.json").exists():
            continue
        s = frame_complexity_score(d)
        if s > best_score and len(d.get("annotations", [])) >= 3:
            best_score, best = s, pj.parent
    return best


# =============================================================================
# Segmentation (COCO RLE) decode
# =============================================================================

def decode_rle(size: List[int], counts) -> np.ndarray:
    """Decode an uncompressed COCO RLE (counts is a list of ints, column-major)
    into an (h, w) uint8 mask."""
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        c = int(c)
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
        if pos >= flat.size:
            break
    return flat.reshape((h, w), order="F")


def overlay_segmentation(img: np.ndarray, frame_dir: Path, pano: Dict, alpha: float = 0.45) -> np.ndarray:
    """Blend the segmentation masks from panorama_geo_sam2_coco.json onto the
    image, using the color of the matching object index."""
    coco_path = frame_dir / "panorama_geo_sam2_coco.json"
    if not coco_path.exists():
        return img
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
    except Exception:
        return img
    h, w = img.shape[:2]
    class_to_idx = {a.get("class"): i for i, a in enumerate(pano.get("annotations", []))}
    overlay = img.copy()
    contours_to_draw = []
    for ca in coco.get("annotations", []):
        seg = ca.get("segmentation")
        if not isinstance(seg, dict) or "counts" not in seg:
            continue
        try:
            mask = decode_rle(seg.get("size", [h, w]), seg["counts"])
        except Exception:
            continue
        if mask.shape != (h, w):
            continue
        label = ca.get("label", "")
        idx = class_to_idx.get(label, ca.get("category_id", 1) - 1)
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        overlay[mask > 0] = color
        contours_to_draw.append((mask, color))
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # Outline the masks so their borders stay readable after blending
    for mask, color in contours_to_draw:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 2, cv2.LINE_AA)
    return img


# =============================================================================
# Annotated panorama image
# =============================================================================

def build_annotated_image(bundle: Dict) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Return (RGB image, top-left pixel of each annotation box)."""
    img = cv2.imread(str(bundle["image_path"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    frame_data = bundle["frame_data"]
    pano = bundle["pano_data"]

    # Segmentation masks go underneath the trajectory and the boxes
    img = overlay_segmentation(img, Path(bundle["image_path"]).parent, pano)

    cals = frame_data.get("camera_calibrations", {})
    if cals:
        proj = CameraProjector(cals, w, h)

        # Past trajectory (orange)
        ps = frame_data.get("past_states", {})
        if "pos_x" in ps and "pos_y" in ps:
            pts = []
            for x, y in zip(ps["pos_x"], ps["pos_y"]):
                u, v, _ = proj.project(np.array([x, y, 0]))
                if u is not None and 0 <= u < w and 0 <= v < h:
                    pts.append((int(u), int(v)))
            for i, pt in enumerate(pts):
                inten = int(110 + 145 * i / max(len(pts), 1))
                if i > 0:
                    cv2.line(img, pts[i - 1], pt, (255, inten, 60), 4)
                cv2.circle(img, pt, 7, (255, inten, 60), -1)

        # Future trajectory (green gradient with an outline so it stands out)
        fs = frame_data.get("future_states", {}) or pano.get("future_states", {})
        if "pos_x" in fs and "pos_y" in fs:
            px, py = fs["pos_x"], fs["pos_y"]
            pz = fs.get("pos_z", [0.0] * len(px))
            pts = []
            for x, y, z in zip(px, py, pz):
                u, v, _ = proj.project(np.array([x, y, z]))
                if u is not None and 0 <= u < w and 0 <= v < h:
                    pts.append((int(u), int(v)))
            for i, pt in enumerate(pts):
                inten = int(255 * (1 - i / max(len(pts), 1)))
                if i > 0:
                    cv2.line(img, pts[i - 1], pt, (0, 210, 90), 7)
            for i, pt in enumerate(pts):
                inten = int(255 * (1 - i / max(len(pts), 1)))
                cv2.circle(img, pt, 11, (0, 255, inten), -1)
                cv2.circle(img, pt, 11, (0, 90, 30), 2)

    # Boxes plus index badges, matching the implication cards on the right
    bbox_origins: List[Tuple[int, int]] = []
    for i, ann in enumerate(pano.get("annotations", [])):
        bbox = ann.get("bbox", {})
        if not bbox:
            bbox_origins.append((0, 0))
            continue
        x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
        bw, bh = int(bbox.get("w", 0)), int(bbox.get("h", 0))
        color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 4)
        # Index badge
        badge = str(i + 1)
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        bx, by = x, max(0, y - th - 16)
        cv2.rectangle(img, (bx, by), (bx + tw + 18, by + th + 16), color, -1)
        cv2.putText(img, badge, (bx + 9, by + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        bbox_origins.append((x, y))
    return img, bbox_origins


# =============================================================================
# Text helpers
# =============================================================================

def _pick_font() -> Optional[str]:
    """Pick an installed font with broad glyph coverage, or None for the default."""
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                 "Source Han Sans SC", "DejaVu Sans"):
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def describe_attrs(attrs: Dict) -> str:
    keep = []
    for k in ("type", "location", "state", "intention", "content", "description"):
        v = attrs.get(k)
        if not v:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        keep.append(f"{k}: {v}")
    return "  ·  ".join(keep)


def wrap(text: str, width_in: float, fontsize_pt: float, factor: float = 0.50) -> List[str]:
    """Wrap text after estimating how many characters fit in the card width."""
    char_w_in = fontsize_pt * factor / 72.0
    n = max(8, int(width_in / max(char_w_in, 1e-3)))
    out: List[str] = []
    for para in str(text).splitlines() or [""]:
        out.extend(textwrap.wrap(para, width=n) or [""])
    return out


# =============================================================================
# Poster rendering
# =============================================================================

class Inches:
    """Convert an inch rectangle into a figure fraction for absolute placement."""
    def __init__(self, W, H):
        self.W, self.H = W, H

    def rect(self, x, y, w, h):
        return [x / self.W, y / self.H, w / self.W, h / self.H]


def add_card(ax, x, y, w, h, *, face=CARD, edge=HAIRLINE, lw=1.4, radius=0.10, z=1):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
                       mutation_aspect=1.0)
    ax.add_patch(p)


def chip(ax, x, y, w, h, label, value, color):
    add_card(ax, x, y, w, h, face=CARD, edge=HAIRLINE, lw=1.2, radius=0.09, z=3)
    ax.add_patch(FancyBboxPatch((x, y), 0.12, h, boxstyle="round,pad=0,rounding_size=0.09",
                                linewidth=0, facecolor=color, zorder=4))
    ax.text(x + 0.32, y + h * 0.66, label.upper(), fontsize=11, color=SUBINK,
            va="center", ha="left", zorder=5, weight="bold")
    ax.text(x + 0.32, y + h * 0.30, value, fontsize=15, color=INK,
            va="center", ha="left", zorder=5, weight="bold")


def build_poster(bundle: Dict, out_png: Path, font: Optional[str]):
    frame_data = bundle["frame_data"]
    pano = bundle["pano_data"]
    anns = pano.get("annotations", [])

    img, _ = build_annotated_image(bundle)
    ih, iw = img.shape[:2]
    img_aspect = iw / ih

    # ---- Canvas size (inches) ----
    W = 22.0
    mx = 0.55                      # left/right outer margin
    content_w = W - 2 * mx
    img_w = content_w
    img_h = img_w / img_aspect     # the panorama is very wide; keep its aspect

    title_h = 1.05
    chips_h = 0.95
    bottom_h = 5.9     # middle band: implications on the left, plan + reason on the right
    charts_h = 2.75    # full-width chart band: bird's-eye trajectory, speed, acceleration
    top_m, bot_m = 0.45, 0.5
    H = (top_m + title_h + 0.10 + chips_h + 0.30 + img_h + 0.40
         + bottom_h + 0.42 + charts_h + bot_m)

    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig = Figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor(BG)
    U = Inches(W, H)

    # Background canvas: every card and label is drawn here, in inches, y upwards
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    bg.add_patch(plt.Rectangle((0, 0), W, H, facecolor=BG, edgecolor="none", zorder=0))

    cursor_y = H - top_m   # lay the sections out from the top down

    # ---- Title bar ----
    ty = cursor_y - title_h
    add_card(bg, mx, ty, content_w, title_h, face=INK, edge=INK, radius=0.12, z=1)
    scene = frame_data.get("scene_id", "")
    frame_id = frame_data.get("frame_id", Path(bundle["image_path"]).parent.name if bundle["image_path"] else "")
    bg.text(mx + 0.45, ty + title_h * 0.62, "Driving Annotation Showcase",
            fontsize=27, color="white", va="center", ha="left", weight="bold", zorder=5)
    bg.text(mx + 0.45, ty + title_h * 0.24,
            "panorama  ·  bbox  ·  segmentation  ·  projected future trajectory  ·  "
            "driving implication  ·  reasoning  ·  final plan",
            fontsize=13, color="#D8DEE9", va="center", ha="left", zorder=5)
    sub = f"scene {str(scene)[:16]}…   frame {frame_id}" if scene else str(frame_id)
    bg.text(mx + content_w - 0.45, ty + title_h * 0.5, sub,
            fontsize=12.5, color="#A3B0C2", va="center", ha="right", zorder=5, family="monospace")
    cursor_y = ty - 0.10

    # ---- Context chips ----
    cy = cursor_y - chips_h
    ctx = pano.get("context", {})
    intent = frame_data.get("intent", "N/A")
    ego = frame_data.get("ego_behavior", {})
    chip_specs = [
        ("Intent", str(intent), BLUE),
        ("Longitudinal", str(ego.get("longitudinal", "N/A")), GREEN),
        ("Lateral", str(ego.get("lateral", "N/A")), PURPLE),
        ("Speed", f"{ego_speed_kmh(frame_data):.1f} km/h", ORANGE),
        ("Scene", " / ".join(filter(None, [ctx.get("scenario"), ctx.get("road")])) or "—", TEAL),
        ("Weather", " / ".join(filter(None, [ctx.get("weather"), ctx.get("daytime")])) or "—", YELLOW),
    ]
    n = len(chip_specs)
    cgap = 0.22
    cw = (content_w - cgap * (n - 1)) / n
    for i, (lab, val, col) in enumerate(chip_specs):
        chip(bg, mx + i * (cw + cgap), cy, cw, chips_h, lab, val, col)
    cursor_y = cy - 0.30

    # ---- Panorama ----
    iy = cursor_y - img_h
    add_card(bg, mx - 0.12, iy - 0.12, content_w + 0.24, img_h + 0.24,
             face="#FFFFFF", edge=HAIRLINE, lw=1.6, radius=0.06, z=1)
    img_ax = fig.add_axes(U.rect(mx, iy, img_w, img_h), zorder=2)
    img_ax.imshow(img)
    img_ax.axis("off")
    # Legend inside the image
    leg = [
        (0.110, "#00E060", "future trajectory (~5s)", "o"),
        (0.055, "#FF9030", "past trajectory (behind ego)", "o"),
        (0.000, "#B48EAD", "segmentation mask + bbox", "s"),
    ]
    for ly, col, txt, mk in leg:
        img_ax.scatter([0.012], [ly + 0.018], s=130, c=col, marker=mk, transform=img_ax.transAxes,
                       edgecolors="white", linewidths=1.0, zorder=6, clip_on=False)
        img_ax.text(0.028, ly + 0.018, txt, transform=img_ax.transAxes,
                    color="white", fontsize=12.5, va="center", ha="left", weight="bold",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="#2E3440D0", ec="none"))
    cursor_y = iy - 0.40

    # ---- Middle band: implications on the left, plan and reasoning on the right ----
    by = cursor_y - bottom_h
    left_w = content_w * 0.575
    right_w = content_w - left_w - 0.40
    right_x = mx + left_w + 0.40

    # Left column: object driving implications
    bg.text(mx + 0.04, by + bottom_h + 0.06, "Object Driving Implications",
            fontsize=16, color=INK, va="bottom", ha="left", weight="bold")
    n_obj = max(1, len(anns))
    ogap = 0.18
    card_h = (bottom_h - ogap * (n_obj - 1)) / n_obj
    # Font size adapts to the number of objects but never drops below 10.5 pt
    impl_fs = float(np.clip(15.5 - 0.7 * n_obj, 10.5, 14.0))
    head_fs = impl_fs + 1.5
    for i, ann in enumerate(anns):
        cyi = by + bottom_h - (i + 1) * card_h - i * ogap
        col = _hex(OBJECT_COLORS[i % len(OBJECT_COLORS)])
        add_card(bg, mx, cyi, left_w, card_h, face=CARD, edge=HAIRLINE, lw=1.3, radius=0.08, z=2)
        # Color bar on the left edge of the card
        bg.add_patch(FancyBboxPatch((mx, cyi), 0.14, card_h,
                     boxstyle="round,pad=0,rounding_size=0.08", linewidth=0,
                     facecolor=col, zorder=3))
        # Circled index
        bg.add_patch(plt.Circle((mx + 0.42, cyi + card_h - 0.30), 0.165, color=col, zorder=4))
        bg.text(mx + 0.42, cyi + card_h - 0.30, str(i + 1), fontsize=head_fs - 1,
                color="white", va="center", ha="center", weight="bold", zorder=5)
        # Class name
        cls = ann.get("class", f"object {i+1}")
        rank = ann.get("attributes", {}).get("impact_rank", "?")
        bg.text(mx + 0.70, cyi + card_h - 0.30, cls, fontsize=head_fs, color=INK,
                va="center", ha="left", weight="bold", zorder=5)
        bg.text(mx + left_w - 0.18, cyi + card_h - 0.30, f"impact rank {rank}",
                fontsize=impl_fs - 2, color=SUBINK, va="center", ha="right",
                style="italic", zorder=5)
        # Attribute line, drawn bold for readability
        attr_txt = describe_attrs(ann.get("attributes", {}))
        attr_fs = impl_fs - 0.5
        ytxt = cyi + card_h - 0.60
        if attr_txt:
            for ln in wrap(attr_txt, left_w - 1.0, attr_fs)[:2]:
                bg.text(mx + 0.70, ytxt, ln, fontsize=attr_fs, color="#434C5E",
                        va="top", ha="left", zorder=5, weight="bold")
                ytxt -= (attr_fs + 4.5) / 72.0
            ytxt -= 0.06
        # Implication body text
        impl = ann.get("driving_implication", "(no implication)")
        for ln in wrap(impl, left_w - 0.95, impl_fs):
            if ytxt < cyi + 0.10:
                break
            bg.text(mx + 0.70, ytxt, ln, fontsize=impl_fs, color="#3B4252",
                    va="top", ha="left", zorder=5)
            ytxt -= (impl_fs + 4) / 72.0

    # Right column: final plan and reasoning
    plan_h = 1.55
    rgap = 0.34
    reason_h = bottom_h - plan_h - rgap

    # Final plan (highlighted card)
    py = by + bottom_h - plan_h
    add_card(bg, right_x, py, right_w, plan_h, face=GREEN, edge=GREEN, radius=0.10, z=2)
    bg.text(right_x + 0.35, py + plan_h - 0.34, "FINAL  PLAN", fontsize=13,
            color="#EAF3EE", va="center", ha="left", weight="bold", zorder=5)
    plan_txt = pano.get("final_plan", "—")
    for j, ln in enumerate(wrap(plan_txt, right_w - 0.7, 20, factor=0.54)[:2]):
        bg.text(right_x + 0.35, py + plan_h - 0.78 - j * 0.40, ln, fontsize=20,
                color="white", va="center", ha="left", weight="bold", zorder=5)

    # Reasoning
    ry = py - rgap - reason_h
    add_card(bg, right_x, ry, right_w, reason_h, face=CARD, edge=HAIRLINE, lw=1.3, radius=0.08, z=2)
    bg.add_patch(FancyBboxPatch((right_x, ry), 0.14, reason_h,
                 boxstyle="round,pad=0,rounding_size=0.08", linewidth=0,
                 facecolor=ORANGE, zorder=3))
    bg.text(right_x + 0.35, ry + reason_h - 0.30, "Reasoning Rationale",
            fontsize=15.5, color=INK, va="center", ha="left", weight="bold", zorder=5)
    reason_txt = pano.get("reason", "") or pano.get("reasoning", "") or "—"
    rytxt = ry + reason_h - 0.66
    for ln in wrap(reason_txt, right_w - 0.7, 13.5):
        if rytxt < ry + 0.12:
            break
        bg.text(right_x + 0.35, rytxt, ln, fontsize=13.5, color="#3B4252",
                va="top", ha="left", zorder=5)
        rytxt -= (13.5 + 5) / 72.0

    # ---- Full-width chart band: bird's-eye trajectory, past speed, past acceleration ----
    past = frame_data.get("past_states", {})
    future = frame_data.get("future_states", {}) or pano.get("future_states", {})
    ch_y = by - 0.42 - charts_h
    cgap2 = 0.40
    panel_w = (content_w - 2 * cgap2) / 3
    panels = [
        ("Trajectory · bird's-eye view (history + future)", lambda a: _draw_bev(a, past, future)),
        ("Ego Speed · past 4s", lambda a: _draw_speed(a, past)),
        ("Ego Acceleration · past 4s", lambda a: _draw_accel(a, past)),
    ]
    for k, (title, draw) in enumerate(panels):
        px = mx + k * (panel_w + cgap2)
        add_card(bg, px, ch_y, panel_w, charts_h, face=CARD, edge=HAIRLINE, lw=1.3, radius=0.07, z=2)
        bg.text(px + 0.34, ch_y + charts_h - 0.30, title, fontsize=14.5, color=INK,
                va="center", ha="left", weight="bold", zorder=5)
        ax = fig.add_axes(U.rect(px + 0.62, ch_y + 0.52, panel_w - 1.02, charts_h - 1.05), zorder=3)
        draw(ax)

    fig.savefig(out_png, dpi=150, facecolor=BG)
    return fig, (W, H)


def _style_axes(ax):
    ax.tick_params(labelsize=11, colors=SUBINK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIRLINE)
    ax.grid(True, alpha=0.25)


def _draw_speed(ax, past: Dict):
    ax.set_facecolor("#FFFFFF")
    if "vel_x" in past and "vel_y" in past:
        vx = np.array(past["vel_x"], dtype=float)
        vy = np.array(past["vel_y"], dtype=float)
        spd = np.hypot(vx, vy)
        t = np.linspace(-4, 0, len(spd))
        ax.fill_between(t, spd, color=ORANGE, alpha=0.18, zorder=1)
        ax.plot(t, spd, color=ORANGE, lw=2.8, zorder=3, label="|speed|")
        ax.plot(t, vx, color=BLUE, lw=2.0, ls="--", zorder=2, label="vel x (fwd)")
        ax.scatter([0], [spd[-1]], s=46, color=ORANGE, zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(f"{spd[-1]*3.6:.0f} km/h", (0, spd[-1]), textcoords="offset points",
                    xytext=(-6, 6), ha="right", fontsize=12, color=INK, weight="bold")
        ax.set_xlim(-4, 0)
        ax.set_ylim(0, max(5, float(spd.max()) * 1.25))
        ax.legend(loc="lower left", fontsize=10, frameon=False)
    ax.set_xlabel("time (s)", fontsize=11, color=SUBINK)
    ax.set_ylabel("m/s", fontsize=11, color=SUBINK)
    _style_axes(ax)


def _draw_accel(ax, past: Dict):
    """Past acceleration: forward, lateral and magnitude."""
    ax.set_facecolor("#FFFFFF")
    if "accel_x" in past and "accel_y" in past:
        ax_ = np.array(past["accel_x"], dtype=float)
        ay_ = np.array(past["accel_y"], dtype=float)
        amag = np.hypot(ax_, ay_)
        t = np.linspace(-4, 0, len(ax_))
        ax.axhline(0, color=HAIRLINE, lw=1.2, zorder=1)
        ax.plot(t, ax_, color=BLUE, lw=2.4, zorder=3, label="accel x (fwd)")
        ax.plot(t, ay_, color=PURPLE, lw=2.4, zorder=3, label="accel y (lat)")
        ax.plot(t, amag, color=RED, lw=1.8, ls="--", zorder=2, label="|accel|")
        ax.scatter([0], [ax_[-1]], s=42, color=BLUE, zorder=4, edgecolors="white", linewidths=1.2)
        lim = max(0.6, float(np.abs(np.concatenate([ax_, ay_])).max()) * 1.4)
        ax.set_xlim(-4, 0)
        ax.set_ylim(-lim, lim)
        ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=1)
    ax.set_xlabel("time (s)", fontsize=11, color=SUBINK)
    ax.set_ylabel("m/s²", fontsize=11, color=SUBINK)
    _style_axes(ax)


def _draw_bev(ax, past: Dict, future: Dict):
    """Bird's-eye view of the past (orange) and future (green) trajectory in ego
    coordinates, where x is forward and y is left."""
    ax.set_facecolor("#F7F9FC")

    def _xy(states):
        if "pos_x" not in states or "pos_y" not in states:
            return None, None
        return np.array(states["pos_x"], float), np.array(states["pos_y"], float)

    pxx, pyy = _xy(past)
    fxx, fyy = _xy(future)
    all_x, all_y = [0.0], [0.0]
    if pxx is not None:
        # Horizontal axis: left (+) / right (-); vertical axis: forward.
        # Left should appear on the left, so plot -y horizontally.
        ax.plot(-pyy, pxx, color=ORANGE, lw=2.8, zorder=3, label="past (~4s)")
        ax.scatter(-pyy, pxx, s=18, color=ORANGE, zorder=4)
        all_x += list(-pyy); all_y += list(pxx)
    if fxx is not None:
        ax.plot(-fyy, fxx, color="#2FAE60", lw=3.0, zorder=3, label="future (~5s)")
        ax.scatter(-fyy, fxx, s=22, color="#2FAE60", zorder=4)
        all_x += list(-fyy); all_y += list(fxx)
    # Ego vehicle
    ax.scatter([0], [0], marker="^", s=200, color=INK, zorder=6, edgecolors="white", linewidths=1.5)
    ax.annotate("ego", (0, 0), textcoords="offset points", xytext=(8, -4),
                fontsize=11, color=INK, weight="bold")

    rng = max(8.0, max(abs(min(all_y)), abs(max(all_y))) * 1.15)
    xr = max(2.5, max(abs(min(all_x)), abs(max(all_x))) * 1.3)
    xr = max(xr, rng * 0.28)
    ax.set_xlim(-xr, xr)
    ax.set_ylim(min(all_y) - 2, max(all_y) + 3)
    ax.set_aspect("auto")
    ax.set_xlabel("← right    left →   (m)", fontsize=11, color=SUBINK)
    ax.set_ylabel("forward (m)", fontsize=11, color=SUBINK)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    _style_axes(ax)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Render one annotated frame into a polished showcase poster.")
    ap.add_argument("--frame_dir", type=str, default=None,
                    help="Folder of the frame to render (holds frame.json and panorama_geo.json)")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="Directory to scan when --frame_dir is omitted; the frame with the "
                         "richest annotations is rendered")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default: annotation_showcase.png inside the frame folder)")
    ap.add_argument("--no-show", action="store_true", help="Only write the PNG, do not open a window")
    args = ap.parse_args()

    if args.frame_dir:
        frame_dir = Path(args.frame_dir)
    elif args.data_dir:
        print(f"Scanning {args.data_dir} for a richly annotated frame...")
        frame_dir = auto_pick_frame(Path(args.data_dir))
        if frame_dir is None:
            print("No suitable frame found.")
            sys.exit(1)
        print(f"Picked: {frame_dir}")
    else:
        ap.error("pass either --frame_dir or --data_dir")

    if not (frame_dir / "panorama_geo.json").exists():
        print(f"Error: {frame_dir} has no panorama_geo.json")
        sys.exit(1)

    bundle = load_frame(frame_dir)
    if bundle["image_path"] is None:
        print("Error: no panorama image found.")
        sys.exit(1)

    out_png = Path(args.out) if args.out else (frame_dir / "annotation_showcase.png")
    font = _pick_font()
    fig, _ = build_poster(bundle, out_png, font)
    print(f"Saved poster -> {out_png}")

    if not args.no_show:
        try:
            matplotlib.use("Qt5Agg")
            import matplotlib.pyplot as _plt
            w, h = fig.get_size_inches()
            mgr = _plt.figure(figsize=(min(w, 19), min(h, 19 * h / w)))
            try:
                mgr.canvas.manager.set_window_title("Annotation Showcase")
            except Exception:
                pass
            # Show the rendered PNG itself, so the window matches the saved file
            shown = _plt.imread(str(out_png))
            ax = mgr.add_axes([0, 0, 1, 1])
            ax.imshow(shown)
            ax.axis("off")
            _plt.show()
        except Exception as e:
            print(f"(Cannot open a window: {e})  Use the saved PNG instead: {out_png}")


if __name__ == "__main__":
    main()
