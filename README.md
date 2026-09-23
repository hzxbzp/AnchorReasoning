<div align="center">
<img src="assets/banner.png" width="880" alt="AnchorReasoning">
<p><b>AnchorReasoning</b> — visual grounding and causal reasoning for end-to-end autonomous driving.<br>
<sub>Waymo End-to-End Driving &nbsp;·&nbsp; point-anchored objects &nbsp;·&nbsp; a reasoning chain that ends in a 5 s trajectory</sub></p>
<p><a href="data_preparation/README.md"><b>Data preparation</b></a> &nbsp;·&nbsp; <a href="training/README.md"><b>Training</b></a> &nbsp;·&nbsp; <a href="evaluation/README.md"><b>Evaluation</b></a></p>
</div>

## Samples

<p align="center"><a href="assets/samples/stop_line_red_light.mp4"><img src="assets/samples/stop_line_red_light.webp" width="880" alt="Stopping at a red light"></a><br>
<sub><b>Stop line, red light</b> — a sign, a red light and a stop line, ranked in that order → decelerate to a full stop at the stop line and wait.</sub></p>
<p align="center"><a href="assets/samples/cyclists_intersection.mp4"><img src="assets/samples/cyclists_intersection.webp" width="880" alt="Cyclists at an intersection"></a><br>
<sub><b>Cyclists at an intersection</b> — a green light ranked first, then four cyclists in the adjacent lane → keep lane and creep forward.</sub></p>
<p align="center"><a href="assets/samples/rainy_narrow_street.mp4"><img src="assets/samples/rainy_narrow_street.webp" width="880" alt="Narrow street in the rain"></a><br>
<sub><b>Narrow street in the rain</b> — a truck stopping ahead, two roadside obstacles closing the corridor → keep lane and decelerate.</sub></p>
<p align="center"><a href="assets/samples/school_bus_overtake.mp4"><img src="assets/samples/school_bus_overtake.webp" width="880" alt="Passing a parked car beside a school bus"></a><br>
<sub><b>Past a parked car, bus alongside</b> — a car parked at the curb, a school bus moving in the adjacent lane → wait for the bus to clear, then nudge left and creep past.</sub></p>
<p align="center"><a href="assets/samples/pull_over_curbside.mp4"><img src="assets/samples/pull_over_curbside.webp" width="880" alt="Pulling over to the curb"></a><br>
<sub><b>Pulling over to the curb</b> — an oncoming car and a bus parked at the roadside → decelerate and pull over to the right curb to stop.</sub></p>
<p align="center"><a href="assets/samples/animal_in_lane.mp4"><img src="assets/samples/animal_in_lane.webp" width="880" alt="Animal in the ego lane"></a><br>
<sub><b>Animal in the ego lane</b> — a single object decides the frame → decelerate and keep lane.</sub></p>

## Layout

```
data_preparation/          tfrecord shards -> frame folders, index and caches
  decode_waymo_e2e.py        shards -> panorama_geo.png + frame.json per frame
  wod_proto.py               E2EDFrame from a descriptor set, no TensorFlow
  make_wod_desc.sh           generates that descriptor set
  extract_rfs_frames.py      the rated validation frames, flattened
  annotation_hygiene.py      context enum normalisation, intent de-flicker
  make_dev_split.py          the internal dev split
  build_index.py             the training index
  build_caches.py            caches aligned line-by-line with the index
  plan_repair.py             final-plan overlay
  make_scene_videos.py       one MP4 per scene
  visualization/             frame posters, annotated videos, viewer

training/                  Stage-1 and Stage-2 supervised fine-tuning
  train_stage.py             entry point for both stages
  dev_eval_callback.py       in-training dev eval, best-checkpoint tracking
  configs/                   common.yaml + 8 backbones x 2 stages, deepspeed/
  core/                      shared library: paths, config, dataset, prompts,
                             targets, trainer, trajectory codec, metrics/
  adapters/                  the 8 backbone adapters
  sbatch/                    SLURM launchers, universal-checkpoint conversion

evaluation/                generation-based evaluation
  eval_dev.py                shared library + dev-set CLI
  eval_val456.py             rated-validation run + official score
  run_rfs.py                 official rater-feedback score and ADE/FDE, CPU only
  ci_eval.py                 chain causal-intervention experiments
  rejudge.py                 re-score saved chains without a GPU
  render_rfs_viz.py          per-frame renders
  baselines/                 released-model scoring, incl. native runners
  sbatch/                    SLURM launchers
```

`training/core/` and `training/adapters/` are the shared library; `data_preparation/` and
`evaluation/` import from them, so the repository root must be on `PYTHONPATH`.

## Setup

Linux, Python 3.11+. Training and generation need an NVIDIA GPU; decoding, index building, scoring
and re-judging run on CPU.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip setuptools wheel

# torch first and alone, from the index matching your driver (cpu instead of cu128 on a CPU host)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0

pip install -r requirements.txt          # GPU: training, generation, evaluation
# pip install -r requirements-cpu.txt    # CPU-only subset
# pip install -r requirements-viz.txt    # figures, videos, renders

# optional, GPU only; pass model.attn=sdpa or --attn sdpa to run without it
MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS=90 pip install flash-attn==2.8.3 --no-build-isolation

# Waymo protos for the decoder -> third_party/wod_e2e.desc
bash data_preparation/make_wod_desc.sh

# official rater-feedback scorer
mkdir -p third_party/waymo_metrics
curl -sfL -o third_party/waymo_metrics/rater_feedback_utils.py \
  https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py

export ANCHOR_ROOT="$PWD" ANCHOR_WEIGHTS="$PWD/weights/base" ANCHOR_DATA="$PWD/data"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Base backbone weights go under `$ANCHOR_WEIGHTS`; the "Base weights" column of
[`training/README.md`](training/README.md) says which layout each adapter expects.

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$ANCHOR_WEIGHTS/Qwen2.5-VL-7B-Instruct"
HF_HUB_CACHE="$ANCHOR_WEIGHTS" hf download nvidia/Cosmos-Reason2-8B
```

<details>
<summary>Environment variables</summary>

Every path is resolved through `training/core/paths.py` and defaults to a location under
`$ANCHOR_ROOT`.

| Variable | Default | Holds |
| --- | --- | --- |
| `ANCHOR_ROOT` | this checkout | repository root; base of every default below |
| `WAYMO_TRAIN_ROOT` | `datasets/waymo_e2e_processed` | decoded training frame folders |
| `WAYMO_VAL_ROOT` | `datasets/waymo_e2e_processed_val` | decoded validation frames, `rater_feedback_frames/`, the cluster JSON |
| `ANCHOR_DATA` | `data` | index, caches, dev split, plan-repair overlay |
| `ANCHOR_WEIGHTS` | `weights/base` | base backbone checkpoints (`ANCHOR_BASE_PATH` overrides with a node-local copy) |
| `ANCHOR_RUNS`, `ANCHOR_EVAL`, `ANCHOR_CACHE`, `ANCHOR_LOGS` | `runs`, `eval_results`, `cache`, `logs` | outputs, scratch cache, job logs |
| `WOD_DESC` | `third_party/wod_e2e.desc` | protobuf descriptor set for the decoder |
| `WAYMO_METRICS_SRC` | `third_party/waymo_metrics` | directory holding `rater_feedback_utils.py` |
| `WAYMO_CALIB_JSON` | `$ANCHOR_DATA/calib_fixed.json` | camera calibration used by the renderer |
| `ANCHOR_RATED_INDEX` | `waymo_e2e_index/val_rated_only.jsonl` | locates a rated frame outside the flattened folder |
| `OPENAI_API_KEY_FILE` | `api_key.txt` | LLM-judge key file; `OPENAI_API_KEY` and `JUDGE_MODEL` are read too |

The evaluation launchers read `ANCHOR_PYTHON` for the interpreter, the training ones `PYTHON_BIN`.

</details>

## Data preparation

```bash
python -m data_preparation.decode_waymo_e2e --split train --src /path/to/wod_e2e/train
python -m data_preparation.decode_waymo_e2e --split val   --src /path/to/wod_e2e/val
python -m data_preparation.extract_rfs_frames

python -m data_preparation.annotation_hygiene --dry-run
python -m data_preparation.annotation_hygiene --apply --changed-only --workers 16

python -m data_preparation.make_dev_split --seed 0 --workers 16
python -m data_preparation.build_index --exclude-scenes "$ANCHOR_DATA/dev_scenes.json" --workers 32
python -m data_preparation.build_caches --workers 32
python -m data_preparation.plan_repair  --workers 32
```

Frame schema and the visualisation scripts: [`data_preparation/README.md`](data_preparation/README.md)

## Training

```bash
torchrun --standalone --nproc_per_node=4 training/train_stage.py \
    --config training/configs/qwen25_7b_s1.yaml trainer.gradient_accumulation_steps=16 wandb.enabled=false
torchrun --standalone --nproc_per_node=4 training/train_stage.py \
    --config training/configs/qwen25_7b_s2.yaml trainer.gradient_accumulation_steps=16 wandb.enabled=false
```

Stage-2 starts from the Stage-1 best checkpoint named by `init_from`. Any trailing `key.sub=value`
overrides the YAML; the configs ship `1 x 2 GPUs x 32 = 64` frames per optimizer step, so set
accumulation to `64 / GPUs`.

The 16 configs and the SLURM launchers: [`training/README.md`](training/README.md)

## Evaluation

```bash
python -m evaluation.eval_val456 --ckpt runs/qwen25_7b_s2/best/best_stage_s2 --run qwen25_7b_s2 --judge

python -m evaluation.eval_dev --ckpt runs/qwen25_7b_s2/best/best_stage_s2 --stage s2 \
    --out eval_results/qwen25_7b_s2/dev_s2_metrics.json

python -m evaluation.ci_eval --ckpt runs/qwen25_7b_s2/best/best_stage_s2 --run qwen25_7b_s2 \
    --base-dir eval_results/qwen25_7b_s2

python -m evaluation.run_rfs --preds eval_results/qwen25_7b_s2/rfs_preds.json \
    --out-dir eval_results/qwen25_7b_s2/rescored
```

Baselines, GPU-free re-judging and renders: [`evaluation/README.md`](evaluation/README.md)
