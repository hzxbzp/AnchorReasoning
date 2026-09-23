#!/usr/bin/env python3
"""Interactive viewer for decoded driving frames, plus a headless statistics mode.

The GUI (PyQt5 + matplotlib) shows, for the frame it is parked on:
  - the panorama with object boxes and the ego past/future trajectory projected
    onto it through the camera calibration;
  - the trajectory in bird's-eye view, the past speed and the past acceleration;
  - the intent label and the panorama_geo.json annotations as readable text;
  - drop-downs for the current and future ego behavior; edits are written back to
    frame.json and propagated to the later frames of the same scene.
Arrow keys or A / W / S step through frames and scenes, +/-/F zoom the panorama.

With --analyze the script opens no window and instead prints the intent
distribution and trajectory statistics of the frames under --data_dir.

Usage:
    python -m data_preparation.visualization.trajectory_visualizer --data_dir <scene-root>
    python -m data_preparation.visualization.trajectory_visualizer --data_dir <scene-root> --analyze

Folder structure expected:
    data_dir/
        video_folder_1/
            frame_folder_1/
                panorama_geo.png (or other image file)
                frame.json
            frame_folder_2/
                ...
        video_folder_2/
            ...

Trajectory Projection Method:
    The future trajectory points are in the vehicle coordinate system (x: forward, y: left, z: up).
    To project these onto the panorama image:
    1. Use the camera extrinsic matrix to transform from vehicle coordinates to camera coordinates
    2. Use the camera intrinsic matrix to project from 3D camera coordinates to 2D image coordinates
    3. Determine which camera (FRONT_LEFT, FRONT, FRONT_RIGHT) can see each point
    4. Apply the horizontal offset based on the camera's position in the panorama
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix Qt plugin conflict between OpenCV and PyQt5
# Must be set before importing cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np


def decode_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO RLE (Run-Length Encoding) mask to binary mask.
    
    Args:
        rle: Dictionary with 'size' [height, width] and 'counts' (list of integers)
    
    Returns:
        Binary mask of shape (height, width)
    """
    size = rle['size']
    counts = rle['counts']
    
    h, w = size[0], size[1]
    
    # Handle string-encoded RLE (compressed format)
    if isinstance(counts, str):
        # This is compressed RLE, need pycocotools to decode
        try:
            from pycocotools import mask as mask_utils
            mask = mask_utils.decode(rle)
            return mask
        except ImportError:
            print("Warning: pycocotools not available, cannot decode compressed RLE")
            return np.zeros((h, w), dtype=np.uint8)
    
    # Uncompressed RLE format: list of integers
    # The counts alternate between 0s and 1s, starting with 0s
    mask_flat = np.zeros(h * w, dtype=np.uint8)
    
    pos = 0
    value = 0  # Start with 0 (background)
    for count in counts:
        mask_flat[pos:pos + count] = value
        pos += count
        value = 1 - value  # Toggle between 0 and 1
    
    # Reshape to 2D (column-major order as per COCO format)
    mask = mask_flat.reshape((w, h)).T  # Transpose because COCO uses column-major
    
    return mask


def load_coco_annotations(json_path: Path) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load COCO format annotations from JSON file.
    
    Returns:
        (annotations_list, category_id_to_name_dict)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load COCO annotations: {e}")
        return [], {}
    
    # Build category mapping
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    return data.get('annotations', []), categories


# Color palette for different object categories
MASK_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 128, 128),  # Light Red
]


class CameraProjector:
    """
    Projects 3D points in vehicle coordinate system to 2D image coordinates.
    
    The panorama image is a horizontal concatenation of three cameras:
    FRONT_LEFT | FRONT | FRONT_RIGHT
    
    Each camera has its own intrinsic and extrinsic parameters.
    
    Based on Waymo Open Dataset projection method.
    """
    
    def __init__(self, calibrations: Dict[str, Any], image_width: int, image_height: int):
        """
        Args:
            calibrations: Camera calibration data from frame.json
            image_width: Total width of the panorama image
            image_height: Height of the panorama image
        """
        self.calibrations = calibrations
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera order in the panorama (left to right): FRONT_LEFT, FRONT, FRONT_RIGHT
        # This matches Waymo's order [2, 1, 3] -> [FRONT_LEFT, FRONT, FRONT_RIGHT]
        self.camera_order = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT']
        self.camera_width = image_width // len(self.camera_order)
        
        # Precompute camera offsets in panorama
        self.camera_offsets = {
            'FRONT_LEFT': 0,
            'FRONT': self.camera_width,
            'FRONT_RIGHT': self.camera_width * 2
        }
    
    def _get_intrinsic(self, cam_name: str) -> np.ndarray:
        """Get intrinsic parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]"""
        return np.array(self.calibrations[cam_name]['intrinsic'])
    
    def _get_extrinsic_matrix(self, cam_name: str) -> np.ndarray:
        """Get 4x4 extrinsic matrix (camera to vehicle transform)."""
        transform = self.calibrations[cam_name]['extrinsic']['transform']
        return np.array(transform).reshape(4, 4)
    
    def project_point_to_camera(self, point_vehicle: np.ndarray, cam_name: str) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Project a 3D point in vehicle coordinates to a specific camera's image coordinates.
        
        Waymo coordinate systems:
        - Vehicle: X-forward, Y-left, Z-up
        - Camera (standard): X-right, Y-down, Z-forward
        
        The extrinsic matrix transforms from camera frame to vehicle frame.
        
        Args:
            point_vehicle: [x, y, z] in vehicle coordinates (x: forward, y: left, z: up)
            cam_name: Camera name ('FRONT_LEFT', 'FRONT', 'FRONT_RIGHT')
        
        Returns:
            (u, v, ok) - image coordinates and validity flag
        """
        intrinsic = self._get_intrinsic(cam_name)
        extrinsic = self._get_extrinsic_matrix(cam_name)
        
        # Extract rotation (R) and translation (t) from extrinsic
        # extrinsic is camera-to-vehicle: P_vehicle = R @ P_camera + t
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Transform from vehicle to camera: P_camera = R^T @ (P_vehicle - t)
        point_vehicle_3d = np.array([point_vehicle[0], point_vehicle[1], point_vehicle[2]])
        point_cam = R.T @ (point_vehicle_3d - t)
        
        # Now point_cam is in camera frame
        # For standard camera: X-right, Y-down, Z-forward
        # But Waymo's camera frame might be: X-forward, Y-left, Z-up (same as vehicle)
        # We need to check which interpretation is correct
        
        # The depth (distance along optical axis) should be positive for visible points
        # In standard camera coords, depth = Z
        # In Waymo camera coords (if same as vehicle), depth = X
        
        # Let's try assuming depth = X (forward direction in vehicle-like camera frame)
        depth = point_cam[0]  # X is forward in vehicle-like frame
        
        if depth <= 0.5:  # Point is behind camera
            return None, None, False
        
        # Project to image plane
        # If camera frame is vehicle-like: X-forward, Y-left, Z-up
        # Then image u corresponds to -Y (left->right), v corresponds to -Z (up->down)
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        
        # u = fx * (-Y/X) + cx  (Y positive = left, so negate for right in image)
        # v = fy * (-Z/X) + cy  (Z positive = up, so negate for down in image)
        u = fx * (-point_cam[1] / depth) + cx
        v = fy * (-point_cam[2] / depth) + cy
        
        # Check if within camera image bounds
        ok = (0 <= u < self.camera_width) and (0 <= v < self.image_height)
        
        return u, v, ok
    
    def project_point(self, point_vehicle: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Project a 3D point in vehicle coordinates to panorama image coordinates.
        Tries all three cameras and returns the best projection.
        
        Args:
            point_vehicle: [x, y, z] in vehicle coordinates
        
        Returns:
            (u, v, camera_name) if point is visible, (None, None, None) otherwise
        """
        best_result = None
        
        for cam_name in self.camera_order:
            u, v, ok = self.project_point_to_camera(point_vehicle, cam_name)
            
            if ok and u is not None and v is not None:
                # Add camera offset to get panorama coordinates
                panorama_u = u + self.camera_offsets[cam_name]
                
                # Keep the result that's most centered in its camera view
                if best_result is None:
                    best_result = (panorama_u, v, cam_name)
                else:
                    # Prefer point closer to camera center
                    curr_dist = abs(u - self.camera_width / 2)
                    best_dist = abs(best_result[0] - self.camera_offsets[best_result[2]] - self.camera_width / 2)
                    if curr_dist < best_dist:
                        best_result = (panorama_u, v, cam_name)
        
        if best_result:
            return best_result
        return None, None, None


def analyze_intents(data_dir: Path) -> Dict[str, int]:
    """
    Analyze all intents in the dataset to understand behavior distribution.
    
    Returns a dictionary of intent -> count.
    """
    intent_counts = {}
    
    # Scan all frame.json files
    for frame_json in data_dir.rglob('frame.json'):
        try:
            with open(frame_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            intent = data.get('intent', 'UNKNOWN')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        except Exception:
            continue
    
    return intent_counts


def analyze_trajectory_characteristics(data_dir: Path, max_samples: int = 1000) -> Dict[str, Any]:
    """
    Analyze trajectory characteristics to help understand ego vehicle behaviors.
    
    Returns statistics about trajectory patterns.
    """
    stats = {
        'total_frames': 0,
        'max_lateral_displacement': [],  # Max |y| in future trajectory
        'max_forward_displacement': [],  # Max x in future trajectory
        'avg_speed': [],
        'speed_change': [],  # Final speed - initial speed
        'lateral_direction': {'left': 0, 'right': 0, 'straight': 0},
        'curvature': [],  # Approximate curvature of trajectory
    }
    
    count = 0
    for frame_json in data_dir.rglob('frame.json'):
        if count >= max_samples:
            break
        
        try:
            with open(frame_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            future_states = data.get('future_states', {})
            past_states = data.get('past_states', {})
            
            if 'pos_x' in future_states and 'pos_y' in future_states:
                pos_x = np.array(future_states['pos_x'])
                pos_y = np.array(future_states['pos_y'])
                
                # Calculate characteristics
                max_y = np.max(np.abs(pos_y))
                max_x = np.max(pos_x)
                final_y = pos_y[-1] if len(pos_y) > 0 else 0
                
                stats['max_lateral_displacement'].append(max_y)
                stats['max_forward_displacement'].append(max_x)
                
                # Determine lateral direction
                if final_y > 0.5:
                    stats['lateral_direction']['left'] += 1
                elif final_y < -0.5:
                    stats['lateral_direction']['right'] += 1
                else:
                    stats['lateral_direction']['straight'] += 1
                
                # Approximate curvature (using endpoint)
                if max_x > 0.1:
                    curvature = abs(final_y) / max_x
                    stats['curvature'].append(curvature)
            
            if 'vel_x' in past_states:
                vel_x = np.array(past_states['vel_x'])
                vel_y = np.array(past_states.get('vel_y', [0] * len(vel_x)))
                speed = np.sqrt(vel_x**2 + vel_y**2)
                
                stats['avg_speed'].append(np.mean(speed))
                if len(speed) > 1:
                    stats['speed_change'].append(speed[-1] - speed[0])
            
            stats['total_frames'] += 1
            count += 1
            
        except Exception:
            continue
    
    return stats


def format_panorama_geo_for_display(data: Dict[str, Any]) -> str:
    """Format panorama_geo.json into readable text (no bbox)."""
    lines = []
    ctx = data.get("context") or {}
    if ctx:
        lines.append("[Context]")
        for key, val in ctx.items():
            key_display = key.replace("_", " ").title()
            lines.append(f"  {key_display}: {val}")
        lines.append("")
    events = data.get("traffic_events") or []
    if events:
        lines.append("[Traffic events]")
        for ev in events:
            types_list = ev.get("types") or []
            lines.append("  * " + ", ".join(str(t) for t in types_list))
        lines.append("")
    anns = data.get("annotations") or []
    if anns:
        lines.append("[Annotations]")
        for i, ann in enumerate(anns, 1):
            class_name = ann.get("class", "-")
            lines.append(f"  {i}. {class_name}")
            attrs = ann.get("attributes") or {}
            for k, v in attrs.items():
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                key_display = k.replace("_", " ").title()
                lines.append(f"      {key_display}: {v}")
            # Display chain if present
            chain = ann.get("chain")
            if chain:
                lines.append(f"      Chain: {chain}")
            lines.append("")
    if not lines:
        return "(No panorama_geo annotations)"
    return "\n".join(lines).rstrip()


def run_gui(data_dir: Path):
    """Run the GUI visualization."""
    # Import GUI libraries only when needed
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSplitter, QComboBox, QScrollArea,
        QSizePolicy, QShortcut, QTextEdit, QGroupBox, QDialog, QCheckBox,
        QGridLayout, QDialogButtonBox,
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
    
    class MplCanvas(FigureCanvas):
        """Matplotlib canvas for embedding plots in PyQt5."""
        
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    # Behavior options offered by the drop-downs; they mirror the ego_behavior
    # vocabulary stored in frame.json.
    LATERAL_BEHAVIORS = [
        "LANE_KEEPING",
        "LEFT_NUDGE",
        "RIGHT_NUDGE",
        "LEFT_LANE_CHANGE",
        "RIGHT_LANE_CHANGE",
        "LEFT_TURN",
        "RIGHT_TURN",
        "LANE_BORROWING",
        "PULL_OVER",
        "PULL_OUT",
    ]
    LONGITUDINAL_BEHAVIORS = [
        "STOP",
        "BACKING",
        "ACCELERATE",
        "DECELERATE",
        "CRUISING",
    ]
    
    class FutureBehaviorEditor(QDialog):
        """Dialog for editing future behavior sequence."""
        
        def __init__(self, sequence: list, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Edit Future Behavior Sequence")
            self.setMinimumWidth(800)
            self.setMinimumHeight(400)
            
            self.sequence = sorted(sequence, key=lambda x: x.get('t', 0)) if sequence else []
            self.combos = []  # List of (t, lat_combo, lon_combo)
            
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            
            # Scroll area for the grid
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_widget = QWidget()
            grid = QGridLayout(scroll_widget)
            grid.setSpacing(5)
            
            # Header row
            grid.addWidget(QLabel("Time"), 0, 0)
            grid.addWidget(QLabel("Lateral Behavior"), 0, 1)
            grid.addWidget(QLabel("Longitudinal Behavior"), 0, 2)
            
            # Create a row for each time step
            for row_idx, item in enumerate(self.sequence):
                t = item.get('t', row_idx)
                lat = item.get('lateral', 'LANE_KEEPING')
                lon = item.get('longitudinal', 'CRUISING')
                
                # Time label
                time_label = QLabel(f"t = {t}")
                time_label.setFont(QFont("Arial", 9, QFont.Bold))
                grid.addWidget(time_label, row_idx + 1, 0)
                
                # Lateral combo
                lat_combo = QComboBox()
                lat_combo.addItems(LATERAL_BEHAVIORS)
                idx = lat_combo.findText(lat)
                if idx >= 0:
                    lat_combo.setCurrentIndex(idx)
                else:
                    lat_combo.addItem(lat)
                    lat_combo.setCurrentIndex(lat_combo.count() - 1)
                grid.addWidget(lat_combo, row_idx + 1, 1)
                
                # Longitudinal combo
                lon_combo = QComboBox()
                lon_combo.addItems(LONGITUDINAL_BEHAVIORS)
                idx = lon_combo.findText(lon)
                if idx >= 0:
                    lon_combo.setCurrentIndex(idx)
                else:
                    lon_combo.addItem(lon)
                    lon_combo.setCurrentIndex(lon_combo.count() - 1)
                grid.addWidget(lon_combo, row_idx + 1, 2)
                
                self.combos.append((t, lat_combo, lon_combo))
            
            scroll.setWidget(scroll_widget)
            layout.addWidget(scroll)
            
            # Buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)
        
        def get_sequence(self) -> list:
            """Get the edited sequence."""
            result = []
            for t, lat_combo, lon_combo in self.combos:
                result.append({
                    't': t,
                    'lateral': lat_combo.currentText(),
                    'longitudinal': lon_combo.currentText()
                })
            return result
    
    class TrajectoryVisualizer(QMainWindow):
        """Main window for trajectory visualization."""
        
        def __init__(self, data_dir: Path):
            super().__init__()
            self.data_dir = data_dir
            
            # Find all video folders and frames
            self.video_folders: List[Path] = []
            self.frame_folders: Dict[Path, List[Path]] = {}
            self._scan_data_dir()
            
            if not self.video_folders:
                raise ValueError(f"No video folders found in {data_dir}")
            
            # Current state
            self.current_video_idx = 0
            self.current_frame_idx = 0
            self._remembered_future_values: Optional[Tuple[str, str]] = None
            self.show_future_trajectory_on_image = True
            
            # Initialize UI
            self.setWindowTitle("Trajectory Visualizer for Autonomous Driving")
            self.setMinimumSize(1600, 900)
            self._init_ui()
            self._setup_shortcuts()
            
            # Load initial frame
            self._load_current_frame()
        
        def _scan_data_dir(self):
            """Scan data directory to find all video folders and frames."""
            # Look for folder structure: data_dir/video_folder/frame_folder
            for video_folder in sorted(self.data_dir.iterdir()):
                if not video_folder.is_dir():
                    continue
                
                # Check if this is a video folder (contains frame folders with frame.json)
                frame_folders = []
                for item in sorted(video_folder.iterdir()):
                    if item.is_dir():
                        # Check if this frame folder has frame.json
                        frame_json = item / 'frame.json'
                        if frame_json.exists():
                            frame_folders.append(item)
                    else:
                        # Maybe the video folder IS the frame folder
                        if item.name == 'frame.json':
                            frame_folders.append(video_folder)
                            break
                
                if frame_folders:
                    self.video_folders.append(video_folder)
                    self.frame_folders[video_folder] = frame_folders
            
            # If no nested structure, treat data_dir itself as containing frame folders
            if not self.video_folders:
                frame_folders = []
                for item in sorted(self.data_dir.iterdir()):
                    if item.is_dir():
                        frame_json = item / 'frame.json'
                        if frame_json.exists():
                            frame_folders.append(item)
                
                if frame_folders:
                    self.video_folders.append(self.data_dir)
                    self.frame_folders[self.data_dir] = frame_folders
        
        def _init_ui(self):
            """Initialize the user interface."""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Main horizontal layout
            main_layout = QHBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            
            # Left panel (charts)
            left_panel = self._create_left_panel()
            
            # Right panel (image and controls)
            right_panel = self._create_right_panel()
            
            # Use splitter for resizable panels
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([400, 1200])
            
            main_layout.addWidget(splitter)
            
            # Status bar
            self.statusBar().showMessage("Ready")
        
        def _create_left_panel(self) -> QWidget:
            """Create left panel with charts."""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(10)
            
            # Title
            title = QLabel("Analysis Charts")
            title.setFont(QFont("Arial", 12, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Trajectory plot (X-Y view)
            trajectory_label = QLabel("Trajectory (X-Y View)")
            trajectory_label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(trajectory_label)
            
            self.trajectory_canvas = MplCanvas(self, width=4, height=3, dpi=100)
            layout.addWidget(self.trajectory_canvas)
            
            # Speed plot
            speed_label = QLabel("Speed (m/s)")
            speed_label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(speed_label)
            
            self.speed_canvas = MplCanvas(self, width=4, height=2, dpi=100)
            layout.addWidget(self.speed_canvas)
            
            # Acceleration plot
            accel_label = QLabel("Acceleration (m/s²)")
            accel_label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(accel_label)
            
            self.accel_canvas = MplCanvas(self, width=4, height=2, dpi=100)
            layout.addWidget(self.accel_canvas)
            
            # Ego Behavior display
            behavior_label = QLabel("Ego Behavior")
            behavior_label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(behavior_label)
            
            behavior_box = QWidget()
            behavior_box.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
            """)
            behavior_layout = QVBoxLayout(behavior_box)
            behavior_layout.setContentsMargins(10, 10, 10, 10)
            behavior_layout.setSpacing(8)
            
            # Lateral behavior (editable dropdown)
            lat_row = QHBoxLayout()
            lat_title = QLabel("Lateral:")
            lat_title.setFont(QFont("Arial", 9, QFont.Bold))
            lat_title.setFixedWidth(80)
            lat_row.addWidget(lat_title)
            self.lateral_behavior_combo = QComboBox()
            self.lateral_behavior_combo.setFont(QFont("Arial", 9))
            self.lateral_behavior_combo.addItems(LATERAL_BEHAVIORS)
            self.lateral_behavior_combo.setStyleSheet("QComboBox { color: #2196F3; font-weight: bold; }")
            lat_row.addWidget(self.lateral_behavior_combo)
            lat_row.addStretch()
            behavior_layout.addLayout(lat_row)
            
            # Longitudinal behavior (editable dropdown)
            lon_row = QHBoxLayout()
            lon_title = QLabel("Longitudinal:")
            lon_title.setFont(QFont("Arial", 9, QFont.Bold))
            lon_title.setFixedWidth(80)
            lon_row.addWidget(lon_title)
            self.longitudinal_behavior_combo = QComboBox()
            self.longitudinal_behavior_combo.setFont(QFont("Arial", 9))
            self.longitudinal_behavior_combo.addItems(LONGITUDINAL_BEHAVIORS)
            self.longitudinal_behavior_combo.setStyleSheet("QComboBox { color: #4CAF50; font-weight: bold; }")
            lon_row.addWidget(self.longitudinal_behavior_combo)
            lon_row.addStretch()
            behavior_layout.addLayout(lon_row)
            
            layout.addWidget(behavior_box)
            
            layout.addStretch()
            return panel
        
        def _create_right_panel(self) -> QWidget:
            """Create right panel with image and controls."""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(10)
            
            # Top controls: video selector and intent display
            top_controls = QHBoxLayout()
            
            # Video selector
            video_label = QLabel("Video:")
            video_label.setFont(QFont("Arial", 10, QFont.Bold))
            top_controls.addWidget(video_label)
            
            self.video_combo = QComboBox()
            for vf in self.video_folders:
                self.video_combo.addItem(vf.name)
            self.video_combo.currentIndexChanged.connect(self._on_video_changed)
            top_controls.addWidget(self.video_combo)
            
            top_controls.addSpacing(30)
            
            # Intent display
            intent_label = QLabel("Intent:")
            intent_label.setFont(QFont("Arial", 10, QFont.Bold))
            top_controls.addWidget(intent_label)
            
            self.intent_display = QLabel("N/A")
            self.intent_display.setFont(QFont("Arial", 12, QFont.Bold))
            self.intent_display.setStyleSheet("""
                QLabel {
                    background-color: #2196F3;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 5px;
                }
            """)
            top_controls.addWidget(self.intent_display)
            
            top_controls.addSpacing(20)
            
            # Future behavior controls (editable dropdowns, same style as current behavior)
            future_label = QLabel("Future:")
            future_label.setFont(QFont("Arial", 10, QFont.Bold))
            top_controls.addWidget(future_label)

            future_lat_label = QLabel("L:")
            future_lat_label.setFont(QFont("Arial", 9, QFont.Bold))
            top_controls.addWidget(future_lat_label)
            self.future_lateral_combo = QComboBox()
            self.future_lateral_combo.setFont(QFont("Arial", 9))
            self.future_lateral_combo.addItems(LATERAL_BEHAVIORS)
            top_controls.addWidget(self.future_lateral_combo)

            future_lon_label = QLabel("T:")
            future_lon_label.setFont(QFont("Arial", 9, QFont.Bold))
            top_controls.addWidget(future_lon_label)
            self.future_longitudinal_combo = QComboBox()
            self.future_longitudinal_combo.setFont(QFont("Arial", 9))
            self.future_longitudinal_combo.addItems(LONGITUDINAL_BEHAVIORS)
            top_controls.addWidget(self.future_longitudinal_combo)
            
            top_controls.addStretch()
            
            # Frame info
            self.frame_info_label = QLabel("Frame: 0/0")
            self.frame_info_label.setFont(QFont("Arial", 10))
            top_controls.addWidget(self.frame_info_label)
            
            layout.addLayout(top_controls)
            
            # Image display area
            self.image_scroll = QScrollArea()
            self.image_scroll.setWidgetResizable(False)
            self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # No horizontal scroll when fit to width
            self.image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_scroll.setWidget(self.image_label)
            
            # Panorama annotations area (height adjustable via splitter)
            panorama_group = QGroupBox("Panorama annotations (panorama_geo.json)")
            panorama_group.setStyleSheet("QGroupBox { font-weight: bold; }")
            panorama_layout = QVBoxLayout(panorama_group)
            self.panorama_geo_text = QTextEdit()
            self.panorama_geo_text.setReadOnly(True)
            self.panorama_geo_text.setMinimumHeight(80)
            self.panorama_geo_text.setFont(QFont("Consolas", 9))
            self.panorama_geo_text.setPlaceholderText("Empty when no panorama_geo.json for this frame")
            panorama_layout.addWidget(self.panorama_geo_text)
            
            # Vertical splitter: image on top, panorama below; drag divider to adjust height
            right_splitter = QSplitter(Qt.Vertical)
            right_splitter.addWidget(self.image_scroll)
            right_splitter.addWidget(panorama_group)
            right_splitter.setStretchFactor(0, 1)
            right_splitter.setStretchFactor(1, 0)
            right_splitter.setSizes([700, 200])  # initial heights in px
            layout.addWidget(right_splitter, stretch=1)
            
            # Image scale mode: 'fit_width' or manual scale factor
            self.image_scale_mode = 'fit_width'  # Auto fit to width by default
            self.image_scale = 1.0  # Manual scale factor when not in fit_width mode
            
            # Bottom controls: navigation
            nav_layout = QHBoxLayout()
            
            self.prev_video_btn = QPushButton("◀◀ Prev Video")
            self.prev_video_btn.clicked.connect(self._prev_video)
            nav_layout.addWidget(self.prev_video_btn)
            
            self.prev_frame_btn = QPushButton("◀ Prev Frame")
            self.prev_frame_btn.clicked.connect(self._prev_frame)
            nav_layout.addWidget(self.prev_frame_btn)
            
            nav_layout.addStretch()
            
            self.next_frame_btn = QPushButton("Next Frame ▶")
            self.next_frame_btn.clicked.connect(self._next_frame)
            nav_layout.addWidget(self.next_frame_btn)
            
            self.next_video_btn = QPushButton("Next Video ▶▶")
            self.next_video_btn.clicked.connect(self._next_video)
            nav_layout.addWidget(self.next_video_btn)
            
            layout.addLayout(nav_layout)
            
            # Legend and zoom controls
            legend_layout = QHBoxLayout()
            legend_layout.addWidget(self._create_legend_item("●", "#00FF00", "Future Trajectory"))
            legend_layout.addWidget(self._create_legend_item("●", "#FF6600", "Past Trajectory"))
            legend_layout.addWidget(self._create_legend_item("●", "#00FFFF", "Current Position"))
            
            legend_layout.addSpacing(20)
            
            # Checkbox to toggle future trajectory display on image
            self.show_future_traj_checkbox = QCheckBox("Show Future Traj")
            self.show_future_traj_checkbox.setFont(QFont("Arial", 9))
            self.show_future_traj_checkbox.setChecked(True)
            self.show_future_traj_checkbox.stateChanged.connect(self._on_show_future_traj_changed)
            legend_layout.addWidget(self.show_future_traj_checkbox)
            
            legend_layout.addStretch()
            
            # Zoom controls
            zoom_label = QLabel("Zoom:")
            zoom_label.setFont(QFont("Arial", 9))
            legend_layout.addWidget(zoom_label)
            
            self.zoom_out_btn = QPushButton("-")
            self.zoom_out_btn.setFixedWidth(30)
            self.zoom_out_btn.clicked.connect(self._zoom_out)
            legend_layout.addWidget(self.zoom_out_btn)
            
            self.zoom_label = QLabel("100%")
            self.zoom_label.setFixedWidth(50)
            self.zoom_label.setAlignment(Qt.AlignCenter)
            legend_layout.addWidget(self.zoom_label)
            
            self.zoom_in_btn = QPushButton("+")
            self.zoom_in_btn.setFixedWidth(30)
            self.zoom_in_btn.clicked.connect(self._zoom_in)
            legend_layout.addWidget(self.zoom_in_btn)
            
            self.zoom_fit_btn = QPushButton("Fit")
            self.zoom_fit_btn.setFixedWidth(40)
            self.zoom_fit_btn.clicked.connect(self._zoom_fit)
            legend_layout.addWidget(self.zoom_fit_btn)
            
            layout.addLayout(legend_layout)
            
            return panel
        
        def _create_legend_item(self, symbol: str, color: str, text: str) -> QLabel:
            """Create a legend item."""
            label = QLabel(f'<span style="color:{color}; font-size:16px;">{symbol}</span> {text}')
            label.setFont(QFont("Arial", 9))
            return label
        
        def _on_show_future_traj_changed(self, state: int):
            """Handle checkbox state change for showing future trajectory on image."""
            self.show_future_trajectory_on_image = (state == Qt.Checked)
            self._load_current_frame()
        
        def _load_current_frame(self):
            """Load and display the current frame."""
            if not self.video_folders:
                return
            
            video_folder = self.video_folders[self.current_video_idx]
            frame_folders = self.frame_folders[video_folder]
            
            if not frame_folders:
                return
            
            frame_folder = frame_folders[self.current_frame_idx]
            
            # Update frame info
            total_frames = len(frame_folders)
            self.frame_info_label.setText(f"Frame: {self.current_frame_idx + 1}/{total_frames}")
            
            # Load frame.json
            frame_json_path = frame_folder / 'frame.json'
            try:
                with open(frame_json_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
            except Exception as e:
                self.statusBar().showMessage(f"Error loading frame.json: {e}")
                return
            
            # Find image file
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidates = list(frame_folder.glob(f'*{ext}'))
                if candidates:
                    image_path = candidates[0]
                    break
            
            if image_path is None:
                self.statusBar().showMessage(f"No image found in {frame_folder}")
                return
            
            # Update intent display
            intent = frame_data.get('intent', 'N/A')
            self.intent_display.setText(intent)
            self._update_intent_color(intent)
            
            # Store current frame data for editing
            self.current_frame_data = frame_data
            self.current_frame_folder = frame_folder
            
            # Update future behavior combo boxes (from future_behavior_sequence or future_behavior)
            future_seq = self._get_future_behavior_sequence_from_frame(frame_data)
            future_lateral = "LANE_KEEPING"
            future_longitudinal = "CRUISING"
            if future_seq:
                future_seq_sorted = sorted(future_seq, key=lambda x: x.get('t', 0))
                first_item = future_seq_sorted[0]
                future_lateral = first_item.get('lateral', future_lateral)
                future_longitudinal = first_item.get('longitudinal', future_longitudinal)

            self.future_lateral_combo.blockSignals(True)
            self.future_longitudinal_combo.blockSignals(True)
            idx_f_lat = self.future_lateral_combo.findText(future_lateral)
            if idx_f_lat >= 0:
                self.future_lateral_combo.setCurrentIndex(idx_f_lat)
            else:
                self.future_lateral_combo.addItem(future_lateral)
                self.future_lateral_combo.setCurrentIndex(self.future_lateral_combo.count() - 1)
            idx_f_lon = self.future_longitudinal_combo.findText(future_longitudinal)
            if idx_f_lon >= 0:
                self.future_longitudinal_combo.setCurrentIndex(idx_f_lon)
            else:
                self.future_longitudinal_combo.addItem(future_longitudinal)
                self.future_longitudinal_combo.setCurrentIndex(self.future_longitudinal_combo.count() - 1)
            self.future_lateral_combo.blockSignals(False)
            self.future_longitudinal_combo.blockSignals(False)
            
            # Update ego behavior combo boxes
            ego_behavior = frame_data.get('ego_behavior', {})
            lateral = ego_behavior.get('lateral', 'LANE_KEEPING')
            longitudinal = ego_behavior.get('longitudinal', 'CRUISING')
            # Set combo box values (block signals to avoid triggering save)
            self.lateral_behavior_combo.blockSignals(True)
            self.longitudinal_behavior_combo.blockSignals(True)
            # Set lateral value (add if not in list)
            idx_lat = self.lateral_behavior_combo.findText(lateral)
            if idx_lat >= 0:
                self.lateral_behavior_combo.setCurrentIndex(idx_lat)
            else:
                # Add unknown value to the list and select it
                self.lateral_behavior_combo.addItem(lateral)
                self.lateral_behavior_combo.setCurrentIndex(self.lateral_behavior_combo.count() - 1)
            # Set longitudinal value (add if not in list)
            idx_lon = self.longitudinal_behavior_combo.findText(longitudinal)
            if idx_lon >= 0:
                self.longitudinal_behavior_combo.setCurrentIndex(idx_lon)
            else:
                # Add unknown value to the list and select it
                self.longitudinal_behavior_combo.addItem(longitudinal)
                self.longitudinal_behavior_combo.setCurrentIndex(self.longitudinal_behavior_combo.count() - 1)
            self.lateral_behavior_combo.blockSignals(False)
            self.longitudinal_behavior_combo.blockSignals(False)
            
            # Load and display image with trajectory
            self._display_image_with_trajectory(image_path, frame_data)
            
            panorama_geo_path = frame_folder / "panorama_geo.json"
            if panorama_geo_path.exists():
                try:
                    with open(panorama_geo_path, "r", encoding="utf-8") as f:
                        panorama_data = json.load(f)
                    self.panorama_geo_text.setPlainText(format_panorama_geo_for_display(panorama_data))
                except Exception as e:
                    self.panorama_geo_text.setPlainText(f"(Load error: {e})")
            else:
                self.panorama_geo_text.clear()
            
            # Update charts
            self._update_charts(frame_data)
            
            # Update status
            self.statusBar().showMessage(f"Loaded: {frame_folder.name}")
        
        def _update_intent_color(self, intent: str):
            """Update intent display color based on intent type."""
            intent_colors = {
                'GO_STRAIGHT': '#4CAF50',
                'TURN_LEFT': '#2196F3',
                'TURN_RIGHT': '#FF9800',
                'LANE_CHANGE_LEFT': '#9C27B0',
                'LANE_CHANGE_RIGHT': '#E91E63',
                'STOP': '#F44336',
                'SLOW_DOWN': '#FF5722',
                'ACCELERATE': '#00BCD4',
            }
            color = intent_colors.get(intent, '#607D8B')
            self.intent_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    padding: 5px 15px;
                    border-radius: 5px;
                }}
            """)
        
        def _save_current_frame_changes(self):
            """Save current/future behavior changes and propagate to subsequent frames."""
            if not hasattr(self, 'current_frame_data') or not hasattr(self, 'current_frame_folder'):
                return

            frame_json_path = self.current_frame_folder / "frame.json"
            if not frame_json_path.exists():
                return

            # Current behavior from combo boxes
            new_lateral = self.lateral_behavior_combo.currentText()
            new_longitudinal = self.longitudinal_behavior_combo.currentText()
            ego_behavior = self.current_frame_data.get('ego_behavior', {})
            old_lateral = ego_behavior.get('lateral', '')
            old_longitudinal = ego_behavior.get('longitudinal', '')
            lateral_changed = new_lateral != old_lateral
            longitudinal_changed = new_longitudinal != old_longitudinal

            # Future behavior from combo boxes
            new_future_lateral = self.future_lateral_combo.currentText()
            new_future_longitudinal = self.future_longitudinal_combo.currentText()
            future_seq = self._get_future_behavior_sequence_from_frame(self.current_frame_data)
            old_future_lateral = "LANE_KEEPING"
            old_future_longitudinal = "CRUISING"
            if future_seq:
                first_item = sorted(future_seq, key=lambda x: x.get('t', 0))[0]
                old_future_lateral = first_item.get('lateral', old_future_lateral)
                old_future_longitudinal = first_item.get('longitudinal', old_future_longitudinal)
            future_lateral_changed = new_future_lateral != old_future_lateral
            future_longitudinal_changed = new_future_longitudinal != old_future_longitudinal
            future_changed = future_lateral_changed or future_longitudinal_changed

            if not (lateral_changed or longitudinal_changed or future_changed):
                return

            # Update current frame data
            if 'ego_behavior' not in self.current_frame_data or not isinstance(self.current_frame_data.get('ego_behavior'), dict):
                self.current_frame_data['ego_behavior'] = {}
            self.current_frame_data['ego_behavior']['lateral'] = new_lateral
            self.current_frame_data['ego_behavior']['longitudinal'] = new_longitudinal

            use_sequence = self._uses_future_behavior_sequence(self.current_frame_data)
            if use_sequence:
                updated_future_seq = []
                src_future = self.current_frame_data.get('future_behavior_sequence', [])
                for idx, item in enumerate(src_future):
                    updated_future_seq.append({
                        't': item.get('t', idx),
                        'lateral': new_future_lateral,
                        'longitudinal': new_future_longitudinal,
                    })
                if not updated_future_seq:
                    updated_future_seq = [{
                        't': 0,
                        'lateral': new_future_lateral,
                        'longitudinal': new_future_longitudinal,
                    }]
                self.current_frame_data['future_behavior_sequence'] = updated_future_seq
            else:
                self.current_frame_data['future_behavior'] = {
                    'lateral': new_future_lateral,
                    'longitudinal': new_future_longitudinal
                }

            # Save current frame
            try:
                with open(frame_json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_frame_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.statusBar().showMessage(f"Error saving: {e}")
                return

            # Propagate current behavior changes
            propagated_ego = self._apply_ego_behavior_to_subsequent_frames(
                new_lateral if lateral_changed else None,
                new_longitudinal if longitudinal_changed else None
            )

            # Remember edited future behavior for manual apply with D key.
            if future_changed:
                self._remembered_future_values = (new_future_lateral, new_future_longitudinal)

            msg = f"Saved. Propagated current:{propagated_ego}"
            if future_changed:
                msg += " | Future behavior remembered (press D to apply on another frame)"
            self.statusBar().showMessage(msg)
        
        def _apply_ego_behavior_to_subsequent_frames(
            self,
            new_lateral: Optional[str] = None,
            new_longitudinal: Optional[str] = None
        ) -> int:
            """Apply ego behavior changes to all subsequent frames in current video."""
            if new_lateral is None and new_longitudinal is None:
                return 0
            video_folder = self.video_folders[self.current_video_idx]
            frame_folders = self.frame_folders[video_folder]
            
            updated_count = 0
            # Start from the frame after current
            for i in range(self.current_frame_idx + 1, len(frame_folders)):
                frame_folder = frame_folders[i]
                frame_json_path = frame_folder / "frame.json"
                
                if not frame_json_path.exists():
                    continue
                
                try:
                    with open(frame_json_path, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                    
                    if 'ego_behavior' not in frame_data:
                        frame_data['ego_behavior'] = {}
                    
                    changed = False
                    if new_lateral is not None and frame_data['ego_behavior'].get('lateral') != new_lateral:
                        frame_data['ego_behavior']['lateral'] = new_lateral
                        changed = True
                    if (
                        new_longitudinal is not None and
                        frame_data['ego_behavior'].get('longitudinal') != new_longitudinal
                    ):
                        frame_data['ego_behavior']['longitudinal'] = new_longitudinal
                        changed = True
                    
                    if changed:
                        with open(frame_json_path, 'w', encoding='utf-8') as f:
                            json.dump(frame_data, f, indent=2, ensure_ascii=False)
                        updated_count += 1
                except Exception:
                    continue
            
            return updated_count
        
        def _open_future_behavior_editor(self):
            """Open a dialog to edit future behavior sequence."""
            if not hasattr(self, 'current_frame_data') or not hasattr(self, 'current_frame_folder'):
                return
            
            original_use_sequence = self._uses_future_behavior_sequence(self.current_frame_data)
            original_seq = self._get_future_behavior_sequence_from_frame(self.current_frame_data)
            dialog = FutureBehaviorEditor(original_seq, self)
            if dialog.exec_() == QDialog.Accepted:
                # Update frame data with edited sequence
                new_sequence = dialog.get_sequence()
                if original_use_sequence:
                    self.current_frame_data['future_behavior_sequence'] = new_sequence
                else:
                    first = new_sequence[0] if new_sequence else {}
                    self.current_frame_data['future_behavior'] = {
                        'lateral': first.get('lateral', 'LANE_KEEPING'),
                        'longitudinal': first.get('longitudinal', 'CRUISING')
                    }
                
                # Save to file
                frame_json_path = self.current_frame_folder / "frame.json"
                try:
                    with open(frame_json_path, 'w', encoding='utf-8') as f:
                        json.dump(self.current_frame_data, f, indent=2, ensure_ascii=False)
                    propagated = 0
                    if new_sequence != original_seq:
                        propagated = self._apply_future_behavior_to_subsequent_frames(
                            new_sequence,
                            use_sequence=original_use_sequence
                        )
                    self.statusBar().showMessage(
                        f"Saved future behavior and propagated to {propagated} subsequent frames"
                    )
                    # Refresh the drop-downs so they match what was just saved
                    self._refresh_future_behavior_controls()
                except Exception as e:
                    self.statusBar().showMessage(f"Error saving: {e}")
        
        def _apply_future_behavior_to_subsequent_frames(
            self,
            new_sequence: List[Dict[str, Any]],
            use_sequence: bool
        ) -> int:
            """Apply edited future behavior to all subsequent frames in current video."""
            video_folder = self.video_folders[self.current_video_idx]
            frame_folders = self.frame_folders[video_folder]
            updated_count = 0
            
            for i in range(self.current_frame_idx + 1, len(frame_folders)):
                frame_folder = frame_folders[i]
                frame_json_path = frame_folder / "frame.json"
                if not frame_json_path.exists():
                    continue
                try:
                    with open(frame_json_path, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                    if use_sequence:
                        src_seq = frame_data.get('future_behavior_sequence', [])
                        template = new_sequence[0] if new_sequence else {
                            'lateral': 'LANE_KEEPING',
                            'longitudinal': 'CRUISING'
                        }
                        if isinstance(src_seq, list) and src_seq:
                            target_seq = []
                            for idx, item in enumerate(src_seq):
                                target_seq.append({
                                    't': item.get('t', idx),
                                    'lateral': template.get('lateral', 'LANE_KEEPING'),
                                    'longitudinal': template.get('longitudinal', 'CRUISING')
                                })
                        else:
                            target_seq = [{
                                't': 0,
                                'lateral': template.get('lateral', 'LANE_KEEPING'),
                                'longitudinal': template.get('longitudinal', 'CRUISING')
                            }]
                        changed = src_seq != target_seq
                        if changed:
                            frame_data['future_behavior_sequence'] = target_seq
                    else:
                        first = new_sequence[0] if new_sequence else {}
                        target_future = {
                            'lateral': first.get('lateral', 'LANE_KEEPING'),
                            'longitudinal': first.get('longitudinal', 'CRUISING')
                        }
                        changed = frame_data.get('future_behavior', {}) != target_future
                        if changed:
                            frame_data['future_behavior'] = target_future
                    
                    if changed:
                        with open(frame_json_path, 'w', encoding='utf-8') as f:
                            json.dump(frame_data, f, indent=2, ensure_ascii=False)
                        updated_count += 1
                except Exception:
                    continue
            
            return updated_count
        
        def _refresh_future_behavior_controls(self):
            """Sync the future behavior drop-downs with the current frame data."""
            future_seq = self._get_future_behavior_sequence_from_frame(self.current_frame_data)
            lateral = "LANE_KEEPING"
            longitudinal = "CRUISING"
            if future_seq:
                first_item = sorted(future_seq, key=lambda x: x.get('t', 0))[0]
                lateral = first_item.get('lateral', lateral)
                longitudinal = first_item.get('longitudinal', longitudinal)
            self._set_combo_value(self.future_lateral_combo, lateral)
            self._set_combo_value(self.future_longitudinal_combo, longitudinal)
        
        def _uses_future_behavior_sequence(self, frame_data: Dict[str, Any]) -> bool:
            """Return True if this frame stores future behavior in sequence format."""
            seq = frame_data.get('future_behavior_sequence')
            return isinstance(seq, list) and len(seq) > 0
        
        def _get_future_behavior_sequence_from_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Read future behavior from frame.json and normalize to sequence format for UI."""
            seq = frame_data.get('future_behavior_sequence')
            if isinstance(seq, list) and len(seq) > 0:
                return seq
            fb = frame_data.get('future_behavior')
            if isinstance(fb, dict) and (fb.get('lateral') is not None or fb.get('longitudinal') is not None):
                return [{
                    't': 0,
                    'lateral': fb.get('lateral', 'LANE_KEEPING'),
                    'longitudinal': fb.get('longitudinal', 'CRUISING')
                }]
            return []
        
        def _display_image_with_trajectory(self, image_path: Path, frame_data: Dict[str, Any]):
            """Display image with trajectory and object masks overlaid."""
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                self.statusBar().showMessage(f"Failed to load image: {image_path}")
                return
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Load and draw COCO annotations (bbox only, no masks) if available
            coco_json_path = image_path.parent / 'panorama_geo_sam2_coco.json'
            if coco_json_path.exists():
                annotations, categories = load_coco_annotations(coco_json_path)
                
                for i, ann in enumerate(annotations):
                    # Get color for this annotation
                    color = MASK_COLORS[i % len(MASK_COLORS)]
                    
                    # Draw bounding box
                    if 'bbox' in ann:
                        bbox = ann['bbox']  # [x, y, width, height]
                        x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 2)
                        
                        # Draw label
                        cat_id = ann.get('category_id', 0)
                        label = ann.get('label', categories.get(cat_id, f'Object {i+1}'))
                        cv2.putText(image, label, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Create projector
            calibrations = frame_data.get('camera_calibrations', {})
            if calibrations:
                projector = CameraProjector(calibrations, w, h)
                
                # Get trajectory data
                future_states = frame_data.get('future_states', {})
                past_states = frame_data.get('past_states', {})
                
                # Draw future trajectory (green) - controlled by checkbox
                future_points = []
                if self.show_future_trajectory_on_image and 'pos_x' in future_states and 'pos_y' in future_states:
                    pos_x = future_states['pos_x']
                    pos_y = future_states['pos_y']
                    pos_z = future_states.get('pos_z', [0] * len(pos_x))
                    
                    for x, y, z in zip(pos_x, pos_y, pos_z):
                        # Use camera projection (based on Waymo method)
                        u, v, _ = projector.project_point(np.array([x, y, z]))
                        
                        if u is not None and v is not None:
                            if 0 <= u < w and 0 <= v < h:
                                future_points.append((int(u), int(v)))
                    
                    # Draw future trajectory
                    for i, pt in enumerate(future_points):
                        # Gradient color from bright green to darker green
                        intensity = int(255 * (1 - i / max(len(future_points), 1)))
                        color = (0, 255, intensity)  # RGB
                        cv2.circle(image, pt, 10, color, -1)
                        cv2.circle(image, pt, 10, (0, 100, 0), 2)  # Dark green border
                        if i > 0:
                            cv2.line(image, future_points[i-1], pt, (0, 200, 0), 4)
                
                # Draw past trajectory (orange) - mostly behind the vehicle
                # Only points with positive x (in front) will be visible
                if 'pos_x' in past_states and 'pos_y' in past_states:
                    past_points = []
                    pos_x = past_states['pos_x']
                    pos_y = past_states['pos_y']
                    
                    for x, y in zip(pos_x, pos_y):
                        # Use camera projection
                        u, v, _ = projector.project_point(np.array([x, y, 0]))
                        if u is not None and v is not None:
                            if 0 <= u < w and 0 <= v < h:
                                past_points.append((int(u), int(v)))
                    
                    # Draw past trajectory (if any points are visible)
                    for i, pt in enumerate(past_points):
                        intensity = int(100 + 155 * i / max(len(past_points), 1))
                        color = (255, intensity, 0)  # RGB
                        cv2.circle(image, pt, 8, color, -1)
                        if i > 0:
                            cv2.line(image, past_points[i-1], pt, color, 3)
                
                # Debug: print projection info if no points were projected
                if not future_points and 'pos_x' in future_states:
                    print(f"[DEBUG] No future points projected. First few trajectory points:")
                    for i in range(min(3, len(future_states['pos_x']))):
                        x, y, z = future_states['pos_x'][i], future_states['pos_y'][i], future_states.get('pos_z', [0]*20)[i]
                        print(f"  Point {i}: vehicle coords ({x:.2f}, {y:.2f}, {z:.2f})")
                        for cam_name in ['FRONT']:
                            u, v, ok = projector.project_point_to_camera(np.array([x, y, z]), cam_name)
                            print(f"    {cam_name}: u={u}, v={v}, ok={ok}")
            
            # Convert to QPixmap and display
            qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            # Calculate scale based on mode
            if self.image_scale_mode == 'fit_width':
                # Fit image width to scroll area width
                scroll_width = self.image_scroll.viewport().width() - 10
                if scroll_width > 0:
                    scale = scroll_width / w
                    new_w = scroll_width
                    new_h = int(h * scale)
                    scaled_pixmap = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.zoom_label.setText(f"{int(scale * 100)}%")
                else:
                    scaled_pixmap = pixmap
            elif self.image_scale != 1.0:
                new_w = int(w * self.image_scale)
                new_h = int(h * self.image_scale)
                scaled_pixmap = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaled_pixmap = pixmap
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()  # Resize label to fit image
        
        def _update_charts(self, frame_data: Dict[str, Any]):
            """Update the charts with trajectory, speed, and acceleration data."""
            past_states = frame_data.get('past_states', {})
            future_states = frame_data.get('future_states', {})
            
            # Update trajectory chart
            self._update_trajectory_chart(past_states, future_states)
            
            # Update speed chart
            self._update_speed_chart(past_states)
            
            # Update acceleration chart
            self._update_accel_chart(past_states)
        
        def _update_trajectory_chart(self, past_states: Dict, future_states: Dict):
            """Update the trajectory X-Y plot."""
            self.trajectory_canvas.fig.clear()
            ax = self.trajectory_canvas.fig.add_subplot(111)
            
            # Plot past trajectory
            if 'pos_x' in past_states and 'pos_y' in past_states:
                past_x = past_states['pos_x']
                past_y = past_states['pos_y']
                ax.plot(past_x, past_y, 'o-', color='#FF6600', linewidth=2, 
                       markersize=4, label='Past Trajectory')
            
            # Plot future trajectory
            if 'pos_x' in future_states and 'pos_y' in future_states:
                future_x = future_states['pos_x']
                future_y = future_states['pos_y']
                ax.plot(future_x, future_y, 'o-', color='#00FF00', linewidth=2,
                       markersize=4, label='Future Trajectory')
            
            # Plot current position
            ax.plot(0, 0, 'o', color='#00FFFF', markersize=12, label='Current Position')
            
            # Add vehicle direction indicator
            ax.annotate('', xy=(2, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            
            ax.set_xlabel('X (forward, m)')
            ax.set_ylabel('Y (left, m)')
            ax.set_title('Trajectory (Vehicle Coordinate System)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Fixed axis range for consistent comparison across frames
            ax.set_xlim(-15, 30)  # Past ~-10m, Future ~+25m
            ax.set_ylim(-10, 10)  # Lateral range
            ax.set_aspect('equal')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            self.trajectory_canvas.fig.tight_layout()
            self.trajectory_canvas.draw()
        
        def _update_speed_chart(self, past_states: Dict):
            """Update the speed plot."""
            self.speed_canvas.fig.clear()
            ax = self.speed_canvas.fig.add_subplot(111)
            
            if 'vel_x' in past_states and 'vel_y' in past_states:
                vel_x = np.array(past_states['vel_x'])
                vel_y = np.array(past_states['vel_y'])
                speed = np.sqrt(vel_x**2 + vel_y**2)
                
                # Past 4 seconds at 4Hz sampling rate
                # Time ranges from -4s to 0s (current time)
                n_points = len(speed)
                t = np.linspace(-4, 0, n_points)  # -4s to 0s
                
                ax.plot(t, vel_x, 'b-', linewidth=2, label='Vel X (forward)')
                ax.plot(t, vel_y, 'g-', linewidth=2, label='Vel Y (left)')
                ax.plot(t, speed, 'r--', linewidth=2, label='Speed (magnitude)')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_xlim(-4, 0)
                ax.set_ylim(-5, 25)  # Fixed y-axis: -5 to 25 m/s
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Current')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.legend(loc='upper left', fontsize=7)
                ax.grid(True, alpha=0.3)
            
            self.speed_canvas.fig.tight_layout()
            self.speed_canvas.draw()
        
        def _update_accel_chart(self, past_states: Dict):
            """Update the acceleration plot."""
            self.accel_canvas.fig.clear()
            ax = self.accel_canvas.fig.add_subplot(111)
            
            if 'accel_x' in past_states and 'accel_y' in past_states:
                accel_x = np.array(past_states['accel_x'])
                accel_y = np.array(past_states['accel_y'])
                accel_mag = np.sqrt(accel_x**2 + accel_y**2)
                
                # Past 4 seconds at 4Hz sampling rate
                n_points = len(accel_x)
                t = np.linspace(-4, 0, n_points)  # -4s to 0s
                
                ax.plot(t, accel_x, 'b-', linewidth=2, label='Accel X (forward)')
                ax.plot(t, accel_y, 'g-', linewidth=2, label='Accel Y (left)')
                ax.plot(t, accel_mag, 'r--', linewidth=2, label='Magnitude')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Acceleration (m/s²)')
                ax.set_xlim(-4, 0)
                ax.set_ylim(-5, 5)  # Fixed y-axis: -5 to 5 m/s²
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.legend(loc='upper left', fontsize=7)
                ax.grid(True, alpha=0.3)
            
            self.accel_canvas.fig.tight_layout()
            self.accel_canvas.draw()
        
        def _on_video_changed(self, index: int):
            """Handle video selection change."""
            # Save any changes before switching
            self._save_current_frame_changes()
            
            self.current_video_idx = index
            self.current_frame_idx = 0
            self._load_current_frame()
        
        def _prev_frame(self):
            """Go to previous frame. If at first frame, wrap to last frame of this video."""
            # Save any changes before switching
            self._save_current_frame_changes()
            
            video_folder = self.video_folders[self.current_video_idx]
            frame_folders = self.frame_folders[video_folder]
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
            else:
                # Wrap to last frame of this video
                self.current_frame_idx = len(frame_folders) - 1
            self._load_current_frame()
        
        def _next_frame(self):
            """Go to next frame."""
            # Save any changes before switching
            self._save_current_frame_changes()
            
            video_folder = self.video_folders[self.current_video_idx]
            frame_folders = self.frame_folders[video_folder]
            if self.current_frame_idx < len(frame_folders) - 1:
                self.current_frame_idx += 1
                self._load_current_frame()
        
        def _apply_remembered_future_behavior(self):
            """Apply remembered future behavior to current frame UI (manual, D key)."""
            if self._remembered_future_values is None:
                self.statusBar().showMessage("No remembered future behavior yet")
                return
            lat, lon = self._remembered_future_values
            self._set_combo_value(self.future_lateral_combo, lat)
            self._set_combo_value(self.future_longitudinal_combo, lon)
            self.statusBar().showMessage("Applied remembered future behavior to current frame (D)")

        def _set_combo_value(self, combo: QComboBox, value: str):
            """Set combo box value, append if not exists."""
            combo.blockSignals(True)
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.addItem(value)
                combo.setCurrentIndex(combo.count() - 1)
            combo.blockSignals(False)
        
        def _prev_video(self):
            """Go to previous video."""
            # Save any changes before switching
            self._save_current_frame_changes()
            
            if self.current_video_idx > 0:
                self.current_video_idx -= 1
                self.current_frame_idx = 0
                self.video_combo.setCurrentIndex(self.current_video_idx)
                self._load_current_frame()
        
        def _next_video(self):
            """Go to next video."""
            # Save any changes before switching
            self._save_current_frame_changes()
            
            if self.current_video_idx < len(self.video_folders) - 1:
                self.current_video_idx += 1
                self.current_frame_idx = 0
                self.video_combo.setCurrentIndex(self.current_video_idx)
                self._load_current_frame()
        
        def _setup_shortcuts(self):
            """Setup keyboard shortcuts."""
            # Navigation shortcuts
            QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_frame)
            QShortcut(QKeySequence(Qt.Key_Right), self, self._next_frame)
            QShortcut(QKeySequence(Qt.Key_Up), self, self._prev_video)
            QShortcut(QKeySequence(Qt.Key_Down), self, self._next_video)
            
            # Also add A/D and W/S as alternative navigation keys
            QShortcut(QKeySequence(Qt.Key_A), self, self._prev_frame)
            QShortcut(QKeySequence(Qt.Key_D), self, self._apply_remembered_future_behavior)
            QShortcut(QKeySequence(Qt.Key_W), self, self._prev_video)
            QShortcut(QKeySequence(Qt.Key_S), self, self._next_video)
            
            # Zoom shortcuts
            QShortcut(QKeySequence(Qt.Key_Plus), self, self._zoom_in)
            QShortcut(QKeySequence(Qt.Key_Equal), self, self._zoom_in)  # = key (same as + without shift)
            QShortcut(QKeySequence(Qt.Key_Minus), self, self._zoom_out)
            QShortcut(QKeySequence(Qt.Key_F), self, self._zoom_fit)  # F for fit
        
        def _zoom_in(self):
            """Zoom in the image."""
            if self.image_scale_mode == 'fit_width':
                # Switch to manual mode, start from current fit scale
                self.image_scale_mode = 'manual'
                # Estimate current scale
                self.image_scale = 1.0
            self.image_scale = min(3.0, self.image_scale + 0.25)
            self.zoom_label.setText(f"{int(self.image_scale * 100)}%")
            self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._load_current_frame()
        
        def _zoom_out(self):
            """Zoom out the image."""
            if self.image_scale_mode == 'fit_width':
                self.image_scale_mode = 'manual'
                self.image_scale = 1.0
            self.image_scale = max(0.25, self.image_scale - 0.25)
            self.zoom_label.setText(f"{int(self.image_scale * 100)}%")
            self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._load_current_frame()
        
        def _zoom_fit(self):
            """Fit image width to scroll area."""
            self.image_scale_mode = 'fit_width'
            self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._load_current_frame()
        
        def resizeEvent(self, event):
            """Handle window resize - update image if in fit_width mode."""
            super().resizeEvent(event)
            if hasattr(self, 'image_scale_mode') and self.image_scale_mode == 'fit_width':
                # Use a timer to avoid too many redraws during resize
                from PyQt5.QtCore import QTimer
                if not hasattr(self, '_resize_timer'):
                    self._resize_timer = QTimer()
                    self._resize_timer.setSingleShot(True)
                    self._resize_timer.timeout.connect(self._load_current_frame)
                self._resize_timer.start(100)  # Delay 100ms
    
    # Start the GUI
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TrajectoryVisualizer(data_dir)
    window.show()
    return app.exec_()


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory Visualizer for Autonomous Driving Dataset"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory holding the decoded frames (a partition root, or a single scene folder)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze intent and trajectory distribution without opening GUI'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=1000,
        help='Maximum number of samples for trajectory analysis (default: 1000)'
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    if args.analyze:
        # Analyze intent distribution
        print(f"Analyzing dataset in {data_dir}...")
        print("\n" + "="*60)
        print("1. INTENT DISTRIBUTION ANALYSIS")
        print("="*60)
        
        intent_counts = analyze_intents(data_dir)
        
        total = sum(intent_counts.values())
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {intent:30s}: {count:6d} ({percentage:5.1f}%)")
        
        print("-"*60)
        print(f"  {'Total':30s}: {total:6d}")
        
        # Analyze trajectory characteristics
        print("\n" + "="*60)
        print("2. TRAJECTORY CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        stats = analyze_trajectory_characteristics(data_dir, args.max_samples)
        
        print(f"  Analyzed frames: {stats['total_frames']}")
        
        if stats['max_lateral_displacement']:
            lat_disp = np.array(stats['max_lateral_displacement'])
            print(f"\n  Lateral Displacement (max |y|):")
            print(f"    Mean: {np.mean(lat_disp):.2f} m")
            print(f"    Max:  {np.max(lat_disp):.2f} m")
            print(f"    Std:  {np.std(lat_disp):.2f} m")
        
        if stats['max_forward_displacement']:
            fwd_disp = np.array(stats['max_forward_displacement'])
            print(f"\n  Forward Displacement (max x):")
            print(f"    Mean: {np.mean(fwd_disp):.2f} m")
            print(f"    Max:  {np.max(fwd_disp):.2f} m")
        
        if stats['avg_speed']:
            avg_speed = np.array(stats['avg_speed'])
            print(f"\n  Average Speed:")
            print(f"    Mean: {np.mean(avg_speed):.2f} m/s ({np.mean(avg_speed) * 3.6:.1f} km/h)")
            print(f"    Max:  {np.max(avg_speed):.2f} m/s ({np.max(avg_speed) * 3.6:.1f} km/h)")
        
        print(f"\n  Lateral Direction Distribution:")
        total_dir = sum(stats['lateral_direction'].values())
        for direction, count in stats['lateral_direction'].items():
            pct = count / total_dir * 100 if total_dir > 0 else 0
            print(f"    {direction:10s}: {count:5d} ({pct:.1f}%)")
        
        # Suggestions for behavior description
        print("\n" + "="*60)
        print("3. SUGGESTED BEHAVIOR CANDIDATE SET FOR VLM TRAINING")
        print("="*60)
        print("""
Based on typical autonomous driving behaviors, consider these unified categories
for both historical and future trajectory descriptions:

LONGITUDINAL BEHAVIORS (Speed-related):
  - ACCELERATING      : Vehicle is increasing speed
  - DECELERATING      : Vehicle is decreasing speed (gentle braking)
  - MAINTAINING_SPEED : Vehicle is at constant velocity
  - STOPPED           : Vehicle is stationary (v ≈ 0)
  - BRAKING_HARD      : Vehicle is applying emergency or hard braking

LATERAL BEHAVIORS (Direction-related):
  - GOING_STRAIGHT    : Vehicle moving forward with minimal lateral movement
  - TURNING_LEFT      : Vehicle executing a left turn (intersection/junction)
  - TURNING_RIGHT     : Vehicle executing a right turn (intersection/junction)
  - SLIGHT_LEFT       : Gentle leftward movement (curve following)
  - SLIGHT_RIGHT      : Gentle rightward movement (curve following)
  - LANE_CHANGE_LEFT  : Vehicle changing to the left lane
  - LANE_CHANGE_RIGHT : Vehicle changing to the right lane

COMBINED DESCRIPTION FORMAT (recommended):
  "[LATERAL_BEHAVIOR] while [LONGITUDINAL_BEHAVIOR]"
  
  Examples:
  - "GOING_STRAIGHT while ACCELERATING"
  - "TURNING_LEFT while DECELERATING"
  - "LANE_CHANGE_RIGHT while MAINTAINING_SPEED"
  - "GOING_STRAIGHT while STOPPED"

SIMPLE FORMAT (if combined format is too complex):
  Use the most significant behavior:
  - If turning/lane changing: use lateral behavior
  - If going straight: use longitudinal behavior if notable (accel/decel/stopped)
  - Default: "GOING_STRAIGHT" or "CRUISING"
""")
        return 0
    
    # Start GUI
    try:
        return run_gui(data_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except ImportError as e:
        print(f"Error: Could not import GUI libraries: {e}")
        print("Try running with --analyze for headless analysis mode.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
