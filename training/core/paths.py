"""Filesystem layout and dataset constants.

Every path is overridable through an environment variable so the code runs
unchanged on any machine. Import paths from this module; do not hard-code them
anywhere else.

    ANCHOR_ROOT          repository root (default: this checkout)
    WAYMO_TRAIN_ROOT     decoded training frames  <root>/<partition>/<scene>/<scene>-<frame>/
    WAYMO_VAL_ROOT       decoded validation frames, same layout
    ANCHOR_WEIGHTS       base backbone checkpoints
    ANCHOR_RUNS          training run outputs
    ANCHOR_EVAL          evaluation run outputs
    ANCHOR_DATA          generated index / cache JSON
    ANCHOR_CACHE         scratch cache
    ANCHOR_LOGS          job logs
    OPENAI_API_KEY_FILE  file holding the OpenAI key used by the LLM judges
"""

import os

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT = os.environ.get("ANCHOR_ROOT") or os.path.dirname(os.path.dirname(_CORE_DIR))
PKG = os.path.join(ROOT, "training")

DATA_ROOT = os.environ.get("WAYMO_TRAIN_ROOT", os.path.join(ROOT, "datasets", "waymo_e2e_processed"))
VAL_ROOT = os.environ.get("WAYMO_VAL_ROOT", os.path.join(ROOT, "datasets", "waymo_e2e_processed_val"))
RFS_DIR = os.path.join(VAL_ROOT, "rater_feedback_frames")
CLUSTER_JSON = os.path.join(VAL_ROOT, "val_sequence_name_to_scenario_cluster.json")

WEIGHTS = os.environ.get("ANCHOR_WEIGHTS", os.path.join(ROOT, "weights", "base"))
RUNS = os.environ.get("ANCHOR_RUNS", os.path.join(ROOT, "runs"))
EVAL = os.environ.get("ANCHOR_EVAL", os.path.join(ROOT, "eval_results"))
DATA = os.environ.get("ANCHOR_DATA", os.path.join(ROOT, "data"))
CACHE = os.environ.get("ANCHOR_CACHE", os.path.join(ROOT, "cache"))
LOGS = os.environ.get("ANCHOR_LOGS", os.path.join(ROOT, "logs"))
API_KEY_FILE = os.environ.get("OPENAI_API_KEY_FILE", os.path.join(ROOT, "api_key.txt"))

PANO_W, PANO_H = 2916, 1079     # stitched FRONT_LEFT | FRONT | FRONT_RIGHT panorama
DT = 0.25                       # trajectory time step in seconds (4 Hz)
HIST_STEPS = 16                 # past ego states fed to the model
FUT_STEPS = 20                  # future waypoints predicted (5 s horizon)
S2_PARTITIONS = [f"p{i}" for i in range(7, 22)]   # annotation partitions used by Stage-2


def partition_of(fdir: str) -> str:
    """'<root>/p19--xx/<scene>/<scene>-<frame>' -> 'p19'."""
    for part in fdir.split("/"):
        if part.startswith("p") and "--" in part and part.split("--")[0][1:].isdigit():
            return part.split("--")[0]
    return "?"
