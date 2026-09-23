# Training

Two-stage supervised fine-tuning of a vision-language backbone on driving frames. Each sample is
one stitched front panorama plus the ego vehicle's past 4 s of motion (16 waypoints, as text) and a
high-level intent; the model answers in a tag grammar.

* **Stage-1 (`s1`) — structured scene understanding.** `<context>` (weather, daytime, visibility,
  scenario, road) → `<events>` → `<has_objects>` → one `<obj …>` block per key object
  (`<point>`, `<rank>`, `<location>`, `<intention>`, `<state>`, `<content>`) → `<n_objects>`.
  A fraction of the Stage-1 samples are single-object Attr-QA samples (the object is named by its
  point in the question). Stage-1 runs a two-phase curriculum: phase `a` supervises the scene-level
  fields only, phase `b` additionally unlocks the per-object fields.
* **Stage-2 (`s2`) — the reasoning chain ending in a trajectory.** `<ego_state>` → the same
  understanding block with an extra `<implication>` per object → `<reason>` → `<final_plan>` →
  `<motion>` → `<traj>`: 5 future waypoints at t = 1…5 s in the ego frame. Stage-2 starts from the
  Stage-1 best checkpoint of the same backbone and has no curriculum.

One entry point (`train_stage.py`) and one YAML per (backbone, stage) drive both stages, for all
eight backbones.

`training/core/` and `training/adapters/` are the shared library of this repository:
`data_preparation/` and `evaluation/` import from them (paths, labels, prompts, trajectory codec,
metrics, adapters), so **the repository root must be on `PYTHONPATH`** for anything here to run.

## Layout

| Path | What it is |
| --- | --- |
| `train_stage.py` | Entry point for both stages: config → adapter → dataset → sampler → trainer, resume, signal handling, `DONE` marker |
| `dev_eval_callback.py` | In-training dev evaluation and best-checkpoint tracking |
| `run_in_alloc.sh` | Run one or more stages sequentially inside an existing interactive allocation |
| `configs/common.yaml` | The documented defaults every model YAML is derived from (not runnable itself) |
| `configs/<backbone>_<s1\|s2>.yaml` | The 16 runnable configs (8 backbones × 2 stages) |
| `configs/deepspeed/*.json` | ZeRO-2 configs: plain, optimizer offload (pinned / unpinned), universal-checkpoint load |
| `sbatch/h100_train.sbatch` | SLURM launcher, defaults sized for 4 GPUs |
| `sbatch/b200_train.sbatch` | SLURM launcher, defaults sized for 2 large-memory GPUs |
| `sbatch/smoke_train_h100.sbatch` | 3-step GPU smoke test of one or more configs |
| `sbatch/ds_to_universal.sbatch` | CPU job converting a ZeRO checkpoint to a universal checkpoint |
| `core/paths.py` | Filesystem layout and dataset constants, all environment-overridable |
| `core/config.py` | YAML loader: defaults, path tokens, dot-list overrides, validation |
| `core/dataset.py` | `WaymoDataset`: sample construction, task switching, curriculum state |
| `core/prompts.py` | System/user prompts, chat scaffold, `prompt_hash` |
| `core/target_builder.py` | Assistant target as `(text, field, weight_mult)` segments + the field weight table `W` |
| `core/labels.py` | Deterministic label derivation from the raw frame records |
| `core/normalize.py` | Canonical label vocabulary and type → field-group rules |
| `core/traj_codec.py` | Trajectory text codec and PCHIP upsampling to the 4 Hz evaluation grid |
| `core/sampler.py` | Stage-aware Stage-1 sampler and static Stage-2 sample weights |
| `core/curriculum.py` | Stage-1 field-unlock sets and the switching callback |
| `core/collator.py` | `TrainCollator`: pads text tensors, concatenates vision tensors |
| `core/trainer.py` | `WeightedTrainer`: two-term weighted CE, per-field logging, LR multipliers, save hooks |
| `core/parse_output.py` | Parses a generated answer back into a structured dict |
| `core/metrics/` | Understanding, object, chain, behaviour, trajectory, judge metrics and the proxy composites |
| `adapters/registry.py` | `get_adapter(name)` / `REGISTRY` |
| `adapters/base.py` | `BackboneAdapter` and the point codecs |
| `adapters/{qwen25,qwen3vl,impromptu,autovla,cosmos,alpamayo}.py` | The concrete backbones |

## Backbones

An adapter isolates everything backbone-specific — image patching, the `<point>` coordinate
convention, model construction, inherited vocabulary that must load but never be generated, and the
vision-tower parameter prefix — so dataset, target builder, trainer and evaluation stay
backbone-agnostic. Every adapter also carries a golden check that the hand-rendered prompt is
token-identical to the processor's own chat template.

| `adapter` | Family | Patch | `<point>` codec | Base weights (under `WEIGHTS`) |
| --- | --- | --- | --- | --- |
| `qwen25_7b` | qwen25 | 14 | absolute processed-image pixels | `Qwen2.5-VL-7B-Instruct` |
| `impromptu_7b` | qwen25 | 14 | absolute processed-image pixels | `ImpromptuVLA-7B_AD` |
| `autovla_3b` | qwen25 | 14 | absolute processed-image pixels | `models--Zewei-Zhou--AutoVLA/snapshots/<hash>` (Lightning `.ckpt`; processor from the Qwen2.5-VL-3B snapshot) |
| `qwen3vl_8b` | qwen3vl | 16 | normalized to 0–1000 | `Qwen3-VL-8B-Instruct` |
| `cosmos_2b` | qwen3vl | 16 | normalized to 0–1000 | `models--nvidia--Cosmos-Reason2-2B/snapshots/<hash>` |
| `cosmos_8b` | qwen3vl | 16 | normalized to 0–1000 | `models--nvidia--Cosmos-Reason2-8B/snapshots/<hash>` |
| `alpamayo_r1_10b` | qwen3vl | 16 | normalized to 0–1000 | `models--nvidia--Alpamayo-R1-10B/snapshots/<hash>` (processor from the Cosmos-Reason2-8B snapshot) |
| `alpamayo_15_10b` | qwen3vl | 16 | normalized to 0–1000 | `models--nvidia--Alpamayo-1.5-10B/snapshots/<hash>` (processor from the Cosmos-Reason2-8B snapshot) |

The AutoVLA action tokens and the Alpamayo `<iN>` / trajectory special tokens stay in the vocabulary
so the released embedding rows load unchanged; they are masked at generation time.

## Environment

```bash
cd <repository root>
export PYTHONPATH=$PWD          # required: training.* is imported as a package
export ANCHOR_ROOT=$PWD         # repository root used by training/core/paths.py
```

`core/paths.py` reads these, each with a default under `ANCHOR_ROOT`:

| Variable | Default | Holds |
| --- | --- | --- |
| `ANCHOR_ROOT` | this checkout | repository root |
| `WAYMO_TRAIN_ROOT` | `datasets/waymo_e2e_processed` | decoded training frame directories |
| `WAYMO_VAL_ROOT` | `datasets/waymo_e2e_processed_val` | decoded validation frame directories |
| `ANCHOR_WEIGHTS` | `weights/base` | base backbone checkpoints |
| `ANCHOR_RUNS` | `runs` | training run outputs |
| `ANCHOR_DATA` | `data` | index and cache JSON built by `data_preparation/` |
| `ANCHOR_EVAL` | `eval_results` | evaluation outputs |
| `ANCHOR_CACHE` | `cache` | scratch cache |
| `ANCHOR_LOGS` | `logs` | job logs |
| `OPENAI_API_KEY_FILE` | `api_key.txt` | key file used by the LLM judges |

Before the first run, `data_preparation/` must have produced the frame index and the row-aligned
caches referenced by the `data:` block, and the backbone weights must exist under `ANCHOR_WEIGHTS`.

## Configuration

`train_stage.py` loads exactly **one** YAML; `common.yaml` is not merged in, every model YAML repeats
all keys. Missing keys fall back to the stage-aware defaults in `core/config.py`.

| Block | Keys that matter |
| --- | --- |
| top level | `adapter` (registry name), `stage` (`s1`/`s2`), `run_name` (run id, also the default output dir), `init_from` (checkpoint directory to start from; `null` = the adapter's base weights) |
| `data` | `index`, `meta`, `ctx_meta`, `motion_meta`, `attr_freq`, `plan_repair`, `dev_online` (index and caches), `long_edge` (image long edge), `attrqa_ratio` (Stage-1 Attr-QA share), `s2_partitions` (Stage-2 pool), `point_mode_train` |
| `sampling` | inverse-frequency powers and caps for intent / context / objects, `combined_cap`, `motion_mult`, `attr_mult`, `event_boost` |
| `loss` | `lambda_traj` (weight of the trajectory term in `L = L_text + lambda_traj * L_traj`), `field_weights` (per-field CE weights; unknown field names are rejected at startup) |
| `curriculum` | `schedule` and `boundaries` (fractions of total steps). Stage-1 only — Stage-2 requires `curriculum: null` |
| `model` | `attn` (`flash_attention_2` / `sdpa`), `freeze_vision`, `gradient_checkpointing`, `visual_lr_mult` |
| `eval` | `enabled`, `every_steps`, `subset`, `max_new_tokens` for the in-training dev eval |
| `trainer` | passed straight to `transformers.TrainingArguments`: `output_dir`, `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `save_steps`, `deepspeed`, `report_to`, `ddp_timeout`, … |
| `wandb` | `project`, `enabled` (advisory; `enabled: false` also forces `trainer.report_to: none`) |

Stage defaults differ: Stage-1 uses lr `1.0e-5`, a trainable vision tower and the `[a, b]` curriculum;
Stage-2 uses lr `5.0e-6`, a frozen vision tower and no curriculum.

### Path tokens

Path-valued keys (`init_from`, everything under `data:` that names a file, `trainer.output_dir`,
`trainer.deepspeed`, `trainer.logging_dir`) may start with a token that `core/config.py` expands
against `core/paths.py`, written either as `DATA/x` or as `${DATA}/x`:

| Token | Expands to |
| --- | --- |
| `ROOT` | repository root |
| `PKG` | `<root>/training` |
| `DATA` | index / cache directory |
| `WEIGHTS` | base checkpoints |
| `RUNS` | run outputs |
| `EVAL` | evaluation outputs |
| `CACHE` | scratch cache |
| `LOGS` | job logs |

### Overrides

Any trailing `key.sub=value` argument overrides the YAML (values parsed as YAML scalars) and is
accepted by `train_stage.py` and by every launcher:

```bash
# resolve a config to its final form (no torch import, instant)
python -m training.core.config training/configs/qwen25_7b_s1.yaml \
    trainer.gradient_accumulation_steps=16 eval.every_steps=250
```

## Running

### Review a config and build the dataset index (no model, no training)

```bash
python training/train_stage.py --config training/configs/qwen25_7b_s1.yaml --print-config
```

Prints the resolved config, builds the dataset, prints the frame count, the sampling summary and the
dataset stats, then exits — exit code `2` if the index or caches are missing.

### Smoke test

3 optimizer steps, DeepSpeed and the dev eval switched off, a handful of frames, a reduced image
long edge, writing to `RUNS/<run_name>_smoke`:

```bash
python training/train_stage.py --config training/configs/qwen25_7b_s1.yaml --smoke
```

### Local / interactive run

```bash
torchrun --standalone --nproc_per_node=4 training/train_stage.py \
    --config training/configs/qwen25_7b_s1.yaml \
    trainer.gradient_accumulation_steps=16
```

The effective batch is `per_device_train_batch_size × GPUs × gradient_accumulation_steps`; the
configs ship the 2-GPU value (1 × 2 × 32 = 64 frames per optimizer step), so set the accumulation to
`64 / GPUs` when you change the GPU count. The launchers below do that themselves.

Inside an existing interactive allocation, run whole stage chains instead:

```bash
bash training/run_in_alloc.sh --dry-run qwen25_7b_s1 qwen25_7b_s2   # print the commands only
bash training/run_in_alloc.sh qwen25_7b_s1 qwen25_7b_s2             # run them sequentially
bash training/run_in_alloc.sh --gpus 2 qwen25_7b_s1 loss.lambda_traj=0.5
```

Stages are idempotent: a stage whose run directory holds a `DONE` marker is skipped, one with
checkpoints resumes, and a failing stage stops the chain.

### Batch runs

The `#SBATCH` header and the `CUDA_MODULE` / `PYTHON_BIN` / `HF_CACHE_DIR` block at the top of each
sbatch file are site placeholders — fill them in before submitting. Submit from the repository root
(or export `ANCHOR_ROOT`), and create the log directory first.

```bash
mkdir -p logs
CONFIG=training/configs/qwen25_7b_s1.yaml sbatch -J qwen25_s1 training/sbatch/h100_train.sbatch
CONFIG=training/configs/qwen25_7b_s2.yaml sbatch -J qwen25_s2 training/sbatch/h100_train.sbatch

# 2-GPU large-memory node
CONFIG=training/configs/alpamayo_15_10b_s1.yaml sbatch -J alpa15_s1 training/sbatch/b200_train.sbatch

# 3-step smoke of one or more configs (config basenames, no .yaml)
CONFIGS="qwen25_7b_s1 qwen25_7b_s2" sbatch training/sbatch/smoke_train_h100.sbatch

# overrides are forwarded to the trainer
CONFIG=training/configs/cosmos_8b_s1.yaml sbatch -J cosmos_s1 training/sbatch/h100_train.sbatch \
    eval.every_steps=250 trainer.save_steps=200
```

Environment knobs understood by the two training launchers:

| Variable | Effect |
| --- | --- |
| `GPUS` | processes to launch; must match the `--gres` GPU count (default 4 for H100, 2 for B200) |
| `LOCAL_COPY_SRC=<dir>` | copy that directory to node-local scratch and use it as `init_from` |
| `RESUME_SRC=<dir>` | resume from that checkpoint instead of the newest one in the run dir (`h100_train.sbatch`) |
| `STAGE_BASE=0` / `STAGE_RESUME=0` | do not stage base weights / the resumed checkpoint on node-local scratch |
| `LOCAL_SCRATCH` | node-local disk the staged copies are written under (default `/lscratch`) |
| `STAGE_TIMEOUT` | seconds a staging copy may take before the job requeues itself |
| `NO_REQUEUE=1` | exit instead of requeueing when training stops without `DONE` |
| `MIN_RUN_SECS` | a run that dies inside this window without saving a checkpoint is treated as a deterministic failure and is not requeued |

Both launchers take a single-writer lock on the run directory, probe CUDA readiness before starting,
forward the walltime `SIGUSR1` to the workers so the trainer saves and stops gracefully, and requeue
themselves until the run directory holds a `DONE` marker. `run_in_alloc.sh` takes the same lock and
performs the same node-local staging.

### Resume

`train_stage.py` always resumes from the newest `checkpoint-<step>` directory in `trainer.output_dir`;
setting `RESUME_FROM=<dir>` points it at a node-local copy of that checkpoint instead (the launchers
export it after staging). Nothing else needs to be passed on a restart — resubmit the same command.

### Run directory

```
RUNS/<run_name>/
  checkpoint-<step>/          trainer checkpoints (model + DeepSpeed optimizer state)
  best/best_stage_{a,b,s2}/   best dev checkpoint per curriculum stage (+ dev_metrics.json)
  best/best.json              the best-composite tracker, survives a requeue
  eval_online/history.jsonl   one line per dev eval, plus one JSON per eval step
  config_resolved.yaml        the fully resolved config of this run
  prompt_hash.json            prompt fingerprint, also written into every save directory
  adapter_name.json           adapter, family, base path, init_from
  final/                      final weights, written only when the full run completed
  DONE                        completion marker
  RUNNING.lock                single-writer lock held while a job trains here
```

## Stage-2 from the Stage-1 best checkpoint

Stage-2 is a continued-training run: `init_from` names the Stage-1 best directory of the *same*
backbone, which the dev-eval callback wrote during Stage-1:

```yaml
init_from: RUNS/qwen25_7b_s1/best/best_stage_b
```

`best_stage_b` is the best checkpoint of the last Stage-1 curriculum phase. `train_stage.py` fails
immediately if `init_from` does not exist (`--print-config` downgrades that to a warning), and warns
if a Stage-2 run has no `init_from` at all, in which case it would train from the base backbone. To
point a Stage-2 run at another checkpoint, override the key:

```bash
CONFIG=training/configs/qwen25_7b_s2.yaml sbatch -J qwen25_s2 training/sbatch/h100_train.sbatch \
    init_from=/path/to/some/checkpoint
```

Because the Stage-1 best directory is large, the launchers can stage it on node-local scratch first:

```bash
LOCAL_COPY_SRC=runs/qwen25_7b_s1/best/best_stage_b \
CONFIG=training/configs/qwen25_7b_s2.yaml sbatch -J qwen25_s2 training/sbatch/h100_train.sbatch
```

## DeepSpeed

`trainer.deepspeed` selects one of the ZeRO-2 configs; batch, accumulation and clipping fields are
`"auto"` and filled in from `TrainingArguments`.

| Config | Use |
| --- | --- |
| `configs/deepspeed/zero2.json` | default, full fine-tuning |
| `configs/deepspeed/zero2_offload.json` | optimizer states on CPU (pinned memory), for fewer or smaller GPUs |
| `configs/deepspeed/zero2_offload_nopin.json` | the same without pinned memory |
| `configs/deepspeed/zero2_universal.json` | loads a universal checkpoint (`"checkpoint": {"load_universal": true}`) |

```bash
CONFIG=training/configs/qwen25_7b_s1.yaml GPUS=2 sbatch -J qwen25_s1 training/sbatch/h100_train.sbatch \
    trainer.deepspeed=training/configs/deepspeed/zero2_offload.json
```

### Resuming on a different GPU count

DeepSpeed stores ZeRO optimizer state as per-rank shards and refuses to load them when the
data-parallel world size changed. Convert the checkpoint once, then resume with any GPU count:

```bash
CKPT=runs/qwen25_7b_s1/checkpoint-3000 sbatch -J univ_qwen25_s1 training/sbatch/ds_to_universal.sbatch
```

This CPU job writes `<CKPT>/global_step<N>_universal/` and the marker file `<CKPT>/latest_universal`,
leaving the per-rank shards untouched, and verifies the result before exiting. On the next launch the
training scripts notice `latest_universal` in the checkpoint they are about to read and add
`trainer.deepspeed=training/configs/deepspeed/zero2_universal.json` themselves (they skip that if you
passed `trainer.deepspeed=` on the command line). Checkpoints saved afterwards are ordinary per-rank
ones again, so later requeues fall back to the config's `zero2.json`.

After a universal load, a callback in `train_stage.py` moves the restored optimizer `step` tensors
onto the parameter device before the first step, which fused AdamW requires.

## In-training dev evaluation

Every `eval.every_steps` optimizer steps — and once at a real end of training — rank 0 generates on
the fixed dev frame list named by `data.dev_online` (`eval.subset` frames, greedy, `eval.max_new_tokens`,
with the backbone's inherited tokens masked), scores the generations with
`evaluation.eval_dev.run_dev_eval` and reduces them to the proxy composite for the stage
(`training.core.metrics.composite`, the same functions the offline evaluation uses). The other ranks
wait at a barrier, which is why `trainer.ddp_timeout` is raised well above the default.

Each evaluation appends to `eval_online/history.jsonl`, logs to the tracker backend if one is
configured, and — when the composite improves for the current curriculum stage — saves model and
processor to `best/best_stage_<a|b|s2>/` together with `dev_metrics.json`, updating `best/best.json`
so a requeued job does not reset the tracker. Stage-2 reads `best/best_stage_b` from the Stage-1 run.

Turn it off per run:

```bash
python training/train_stage.py --config training/configs/qwen25_7b_s1.yaml eval.every_steps=0
```

If the dev frame list is missing or the evaluation package cannot be imported, the callback prints a
warning and training continues without it.
