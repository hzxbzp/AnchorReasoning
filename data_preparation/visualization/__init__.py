"""Visualization of decoded and annotated driving frames.

Modules:
  - ``annotation_showcase``: render one frame into a single poster PNG (panorama
    with masks, boxes and projected trajectory, per-object driving implications,
    reasoning, final plan and the ego motion charts).
  - ``trajectory_visualizer``: interactive PyQt5 viewer for stepping through
    scenes and frames, and a headless ``--analyze`` statistics mode; it also
    provides the camera projection and annotation helpers reused below.
  - ``generate_annotated_video``: one composite review video per scene
    (annotated panorama, annotation text panel and motion charts).
  - ``generate_trajectory_videos``: one lightweight video per scene showing only
    the projected future trajectory and the intent banner.

Every entry point takes the frame directory to read as a command-line argument;
run any of them with ``python -m data_preparation.visualization.<module> --help``.
"""
