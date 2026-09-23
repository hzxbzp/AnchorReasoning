# `data_preparation`

Everything that turns the raw **Waymo End-to-End Driving** tfrecord shards into the on-disk
frame folders the training and evaluation code reads, plus the index / cache JSON files that
select and describe the training rows, and the scripts that render frames and scenes for
inspection.

Every script is a module: run it with `python -m data_preparation.<module>`, from the repository
root or with `PYTHONPATH=<repo root>` set. All of them accept `--help`.

## Pipeline order

```
decode_waymo_e2e     tfrecord shards      -> frame folders (panorama_geo.png + frame.json)
extract_rfs_frames   val frame folders    -> rater_feedback_frames/ + rfs_index.json

(annotation step, outside this folder: panorama_geo.json + panorama_geo_sam2_coco.json per frame)

annotation_hygiene   frame folders        -> normalised context enums + intent_corrected, in place
make_dev_split       p19-p21 scenes       -> dev_scenes.json, dev_eval_frames.json, dev_online_128.json
build_index          annotated scenes     -> index_train.json (+ index_stats.json)
build_caches         index_train.json  -> row-aligned caches + attr_class_freq.json
plan_repair          index_train.json  -> plan_repair.json overlay (+ stats)
```

`extract_rfs_frames` only touches the validation root and can run any time after decoding.
`make_dev_split` runs before `build_index` because the index is built with the dev scenes excluded;
it scans the scene folders directly and does not need the index. `build_caches` and `plan_repair`
both consume the finished index and emit files that are aligned with it or keyed by its frame
directories.

## Files

| File | What it does |
| --- | --- |
| `decode_waymo_e2e.py` | Decodes `E2EDFrame` records into one folder per frame: a stitched forward panorama plus `frame.json`. Handles both splits. Reads the shards with a small pure-python TFRecord reader (no TensorFlow); the `E2EDFrame` message comes from `wod_proto.py`. |
| `wod_proto.py` | Resolves the `E2EDFrame` message class: the installed `waymo_open_dataset` package if there is one, otherwise message classes built at runtime from a protobuf descriptor set (`$WOD_DESC`, default `third_party/wod_e2e.desc`). |
| `make_wod_desc.sh` | Generates that descriptor set: fetches the six `.proto` sources and compiles them with `--include_imports`, using a `protoc` installed into a throwaway prefix so the active environment is untouched. |
| `extract_rfs_frames.py` | Copies the rater-feedback frames — those whose `preference_trajectories` carry real positions rather than a `preference_score: -1.0` placeholder — into one flat folder, with an `rfs_index.json` listing them. |
| `annotation_hygiene.py` | Two in-place fixes, **dry-run by default**: context enum normalisation of `panorama_geo.json` (scenario spellings, empty weather, traffic-event types) and ego-intent de-flicker, written as `intent_corrected` into `frame.json`. `--apply` appends an auditable run record to a changelog and runs an acceptance check. |
| `make_dev_split.py` | Carves an internal dev split of 30 scenes out of p19-p21, stratified by `<majority intent>|<majority sample_class>` over the evaluation frames (frame id 140-160), with greedy repair until hard minima on start / stop / left / right frames hold. |
| `build_index.py` | Builds the training index: content-aware per-scene strides (all turn frames, every second object-bearing straight frame, sparse empty negatives), negatives then capped as a fraction of the kept set. Also hosts the directory-discovery helpers (`list_partitions`, `list_scenes`, `list_frame_dirs`, `frame_id`, `load_scene_list`) the other scripts import. |
| `build_caches.py` | Per-frame caches aligned **line-by-line** with the index: intent/object counts, weather/visibility/event flags, motion and sampling labels, and attribute-value frequencies. Label rules come from `training.core.labels`, the single implementation shared with training and evaluation. |
| `plan_repair.py` | Compares the annotated `final_plan` with the longitudinal trend of the ground-truth future trajectory and rewrites the plan's longitudinal verb phrase where the two contradict each other. Emits an overlay keyed by frame directory; the `panorama_geo.json` files are not modified. |
| `make_scene_videos.py` | One MP4 preview per scene, straight from that scene's `panorama_geo.png` frames in frame order. |
| `visualization/annotation_showcase.py` | Renders one frame into a single poster PNG: panorama with masks, boxes and projected trajectory, one card per annotated object, the reasoning and final plan, and the ego motion charts. |
| `visualization/trajectory_visualizer.py` | Interactive PyQt5 + matplotlib viewer for stepping through scenes and frames; `--analyze` is a headless statistics mode. Also provides the camera projection and annotation helpers the other visualisation scripts import. |
| `visualization/generate_annotated_video.py` | One composite review video per scene: annotated panorama, annotation text panel, and bird's-eye / speed charts. |
| `visualization/generate_trajectory_videos.py` | One lightweight video per scene showing only the projected future trajectory and the intent banner. |

## On-disk layout

```
<root>/
  p<N>--<suffix>/                              # partition folder (train); "p<N>" as written by the decoder
    <scene_id>/
      <scene_id>-<frame_id>/
        panorama_geo.png              # 2916x1079 stitched FRONT_LEFT | FRONT | FRONT_RIGHT
        frame.json                    # intent, ego states, preference trajectories
        panorama_geo.json             # frame annotation (added by the annotation step)
        panorama_geo_sam2_coco.json   # COCO-format instance masks for those annotations
      ...
```

The decoder groups scenes 100 per `p<N>` folder, in order of first appearance. Everything after
the decoding step — `build_index`, `build_caches`, `plan_repair`, `annotation_hygiene`,
`make_dev_split` — discovers partitions by the pattern `p<N>--<suffix>`, so a decoded root has its
partition folders renamed to that form before the index is built. `build_index` only keeps frame
folders that hold **both** `frame.json` and `panorama_geo.json`, i.e. annotated frames.

`extract_rfs_frames` writes a flat folder next to the validation partitions:

```
<val root>/rater_feedback_frames/
  <scene_id>-<frame_id>/       # the frame folder, copied verbatim
  rfs_index.json               # {"count": N, "frames": [{scene_id, frame_id, intent, preference_scores, ...}]}
```

### `frame.json`

| Field | Meaning |
| --- | --- |
| `scene_id`, `frame_id` | Split of the record's context name; the folder is named `<scene_id>-<frame_id>`. |
| `past_states` | Ego history, 16 samples at 4 Hz (0.25 s step), oldest first, the last one at the current pose. Only the populated fields out of `pos_x`, `pos_y`, `pos_z`, `vel_x`, `vel_y`, `accel_x`, `accel_y` are written. Metres / m·s⁻¹ / m·s⁻² in the vehicle frame (x forward, y left, z up), origin at the current pose. |
| `future_states` | Ego future, 20 samples at 4 Hz (5 s horizon), first sample one step ahead, same fields and frame. This is the trajectory target. |
| `intent` | The record's intent enum name: `GO_STRAIGHT`, `GO_LEFT`, `GO_RIGHT`. |
| `intent_corrected` | Written by `annotation_hygiene --apply`. Every consumer reads `intent_corrected or intent`. |
| `preference_trajectories` | List of rater trajectories, each a state dict plus `preference_score`. Frames without rater feedback carry placeholder entries that hold only `preference_score: -1.0`. |
| `ego_behavior` | `{"longitudinal": ..., "lateral": ...}`. |
| `camera_calibrations` | Per-camera `intrinsic` / `extrinsic.transform`, when the frame carries it; this is what the `visualization/` scripts use to project trajectories onto the panorama. |

### `panorama_geo.json`

Read by `build_index` (object presence), `build_caches` (context, events, attributes, chain
completeness) and `plan_repair` (`final_plan`); written by `annotation_hygiene`'s clean step.

| Field | Meaning |
| --- | --- |
| `context` | `{weather, daytime, visibility, scenario, road}`. |
| `traffic_events` | List of `{types: [...]}`; anything other than `Roadside parking` counts as an interesting event. |
| `annotations` | Per object: `class`, `bbox` `{x, y, w, h}` in panorama pixels, `attributes` (`type`, `impact_rank`, and type-dependent fields such as `intention` / `state` / `content` / `location` / `description`), and `driving_implication`. |
| `reason` | Frame-level reasoning rationale. |
| `final_plan` | Frame-level plan sentence. |

## Environment variables

Read through `training/core/paths.py`; set them, or pass the corresponding flag on every command.

| Variable | Used for |
| --- | --- |
| `ANCHOR_ROOT` | Repository root; the default of every path below is derived from it. |
| `WAYMO_TRAIN_ROOT` | Decoded training frames. Default of `--out` for `--split train`, and of `--roots` in `annotation_hygiene` / `make_dev_split` / `build_index`. |
| `WAYMO_VAL_ROOT` | Decoded validation frames. Default of `--out` for `--split val`, and of `--root` in `extract_rfs_frames` / `make_scene_videos`. |
| `ANCHOR_DATA` | Where the index, caches, dev split, overlay, changelog and reports are written, and where `--index` is read from (`--out`, `--out-dir`, `--stats-out`, `--index`, `--changelog`, `--report`). |

The `visualization/` scripts take the directory to read as a command-line argument and read no
environment variables of their own.

Python dependencies: `numpy`, `opencv-python`, `Pillow` and `protobuf` for decoding;
`matplotlib` for the posters and videos; `PyQt5` for the interactive viewer.

Before the first decode, generate the Waymo E2E descriptor set that `wod_proto.py` reads
(skip it if the `waymo_open_dataset` package is already importable — `wod_proto.py` prefers it):

```bash
bash data_preparation/make_wod_desc.sh        # writes third_party/wod_e2e.desc
```

Pass a path as the first argument, or set `$WOD_DESC`, to keep it elsewhere. The script needs
`curl`, network access and either `pip` or `uv`; it installs `protoc` into a throwaway prefix and
leaves the active environment untouched.

## Commands

### 1. Decode the tfrecord shards

```bash
# validation split -> $WAYMO_VAL_ROOT
python -m data_preparation.decode_waymo_e2e --split val --src /path/to/wod_e2e/val

# training split, explicit output root
python -m data_preparation.decode_waymo_e2e --split train \
    --src /path/to/wod_e2e/train --out /path/to/waymo_e2e_processed

# smoke run: first two shards, stop after 50 frames
python -m data_preparation.decode_waymo_e2e --split val --src /path/to/wod_e2e/val \
    --shards 2 --max-frames 50 --out /tmp/decode_smoke
```

`--input-glob` overrides the shard pattern (default `<split>*.tfrecord*`); `--max-scenes` stops
after N scenes.

### 2. Extract the rater-feedback frames

```bash
python -m data_preparation.extract_rfs_frames --dry-run     # count and list only
python -m data_preparation.extract_rfs_frames               # copy into $WAYMO_VAL_ROOT/rater_feedback_frames
python -m data_preparation.extract_rfs_frames --root /path/to/val --dest rater_feedback_frames
```

### 3. Annotation hygiene

```bash
# dry run (default): nothing is written, statistics go to the report file
PYTHONPATH=$PWD python -m data_preparation.annotation_hygiene --dry-run

# apply to selected partitions, writing intent_corrected only where the label changed
PYTHONPATH=$PWD python -m data_preparation.annotation_hygiene \
    --partitions p19 p20 p21 --apply --changed-only --workers 16

# acceptance check on its own
PYTHONPATH=$PWD python -m data_preparation.annotation_hygiene --verify-only --verify-n 500
```

`--limit N` caps the clean step to the first N panorama files and `--limit-scenes N` the intent
step to the first N scenes (smoke runs); `--skip-clean` / `--skip-intent` run one step only.

### 4. Dev split

```bash
PYTHONPATH=$PWD python -m data_preparation.make_dev_split --seed 0 --workers 16

# smoke run (implies --relax)
PYTHONPATH=$PWD python -m data_preparation.make_dev_split \
    --limit 5 --n-scenes 2 --n-online 8 --out-dir /tmp/devsplit
```

Constraints are adjustable with `--min-start / --min-stop / --min-left / --min-right`, the frame
window with `--fid-lo / --fid-hi`, and the search with `--restarts`. Without `--relax` the script
exits non-zero when the constraints cannot be met.

### 5. Training index

```bash
PYTHONPATH=$PWD python -m data_preparation.build_index \
    --exclude-scenes $ANCHOR_DATA/dev_scenes.json --workers 32

# restrict to some partitions, write elsewhere
PYTHONPATH=$PWD python -m data_preparation.build_index \
    --partitions p19 p20 p21 --limit 20 --out /tmp/index_smoke.json
```

Writes `index_train.json` (a flat list of frame directories) and `index_stats.json` next to it.

### 6. Row-aligned caches

```bash
PYTHONPATH=$PWD python -m data_preparation.build_caches \
    --index $ANCHOR_DATA/index_train.json --workers 32
```

Writes `index_meta.json`, `index_ctx_meta.json`, `index_motion_meta.json`,
`attr_class_freq.json` and `caches_stats.json` into `--out-dir`. The three `index_*.json`
files have exactly one row per index row, in the same order — unreadable frames keep default values
so the alignment never breaks.

### 7. Final-plan repair

```bash
PYTHONPATH=$PWD python -m data_preparation.plan_repair \
    --index $ANCHOR_DATA/index_train.json --workers 32

# all partitions instead of the default p7..p21
PYTHONPATH=$PWD python -m data_preparation.plan_repair --partitions all
```

Writes `plan_repair.json`, an overlay `{frame_dir: {final_plan, reason_weight, orig_plan, ...}}`,
and `plan_repair_v2_stats.json`.

### 8. Scene preview videos

```bash
python -m data_preparation.make_scene_videos                          # all scenes under $WAYMO_VAL_ROOT, 10 fps
python -m data_preparation.make_scene_videos --root /path/to/train_root --jobs 8 --overwrite
python -m data_preparation.make_scene_videos --fps 5 --codec avc1
```

Each video is written as `<scene_id>/<scene_id>.mp4` inside the scene folder.

### 9. Visualisation

```bash
# poster PNG of one frame, no window
python -m data_preparation.visualization.annotation_showcase \
    --frame_dir /path/to/root/p19--part/<scene>/<scene>-142 --no-show

# or let it pick the most richly annotated frame under a directory
python -m data_preparation.visualization.annotation_showcase \
    --data_dir /path/to/root/p19--part --out /tmp/poster.png --no-show

# interactive viewer
python -m data_preparation.visualization.trajectory_visualizer --data_dir /path/to/root/p19--part

# headless statistics over the frames below --data_dir
python -m data_preparation.visualization.trajectory_visualizer \
    --data_dir /path/to/root/p19--part --analyze --max_samples 2000

# composite review video per scene
python -m data_preparation.visualization.generate_annotated_video \
    --data_dir /path/to/root/p19--part --fps 10
python -m data_preparation.visualization.generate_annotated_video \
    --data_dir /path/to/root/p19--part/<scene> --output /tmp/scene.mp4 \
    --main_height 720 --panel_width 380 --chart_height 320

# trajectory-only video per scene
python -m data_preparation.visualization.generate_trajectory_videos \
    --data_dir /path/to/root/p19--part --output_dir /tmp/traj_videos --fps 10
```

`annotation_showcase` needs `panorama_geo.json` in the frame folder and writes
`annotation_showcase.png` there unless `--out` says otherwise. The mask overlays come from
`panorama_geo_sam2_coco.json` when it is present.
