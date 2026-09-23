# `evaluation/`

Generation-based evaluation for the driving chain: one greedy decode per frame, the chain is parsed,
the five text waypoints are PCHIP-upsampled to the official 20-point / 5 s grid, and the result is
scored for understanding, trajectory and chain consistency. On top of that sit the official Waymo
rater-feedback score, the causal-intervention experiments, and a capability-aware scoring path for
released models that were never trained on this output format.

Every entry point is a module of the `evaluation` package and is run from the repository root:

```bash
cd <repo root>
export PYTHONPATH=$PWD
python -m evaluation.eval_dev --help
```

## What a frame is

All scripts consume *frame directories*, produced by `data_preparation/`:

```
<scene_id>-<frame_id>/
  frame.json                     ego past/future states, intent, rater preference trajectories
  panorama_geo.png               FRONT_LEFT | FRONT | FRONT_RIGHT stitched panorama
  panorama_geo.json              object annotations (the understanding ground truth)
  panorama_geo_sam2_coco.json    SAM2 masks, used for stable interior points (optional)
```

`--frames` accepts `rfs` (all rated validation frames under `$WAYMO_VAL_ROOT/rater_feedback_frames`),
a directory that is walked recursively, or a JSON file holding a list of frame dirs (or a dict with a
`frames` / `fdirs` key) — for example `data/dev_eval_frames.json` from
`data_preparation/make_dev_split.py`.

## Files

| Path | What it is |
| --- | --- |
| `eval_dev.py` | Shared library **and** dev-set CLI: prompt building, the frozen decoding protocol, record assembly, scoring, paired bootstrap CIs. Everything else imports it. |
| `eval_val456.py` | Rated-validation evaluation: Stage-2 chain pass, official rater-feedback score, reference rows, acceptance gates. |
| `run_rfs.py` | Official Waymo rater-feedback score plus ADE/FDE on a predictions file. Pure numpy — runs on a login node. |
| `ci_eval.py` | Chain causal-intervention experiments CI-1 / CI-2 / CI-3 (teacher-forced assistant prefixes). |
| `rejudge.py` | CPU re-scoring of a finished run from its saved chains; refreshes the LLM-judge columns without touching a GPU. |
| `render_rfs_viz.py` | Per-frame PNG renderer for a finished run (panorama + trajectories + chain text box). |
| `viz_common.py` | Figure construction and panorama projection used by the renderer. |
| `baselines/profiles.py` | Static capability profile per released model (coordinate system, display name, notes). |
| `baselines/capabilities.py` | Decides from a model's own output which fields are actually scorable (coverage and distinctness thresholds). |
| `baselines/text_extract.py` | Tolerant free-text → `parse_output`-shaped dict, for models that do not honour the tag grammar. |
| `baselines/typeonly.py` | Geometry-free object scoring (class-multiset matching) for models that emit types but no image points. |
| `baselines/rescore.py` | CLI tying the four above together: `chains.json` → `metrics_baseline.json`. |
| `baselines/native/common.py` | Environment-agnostic helpers (frame index, resampling, resumable row sink) loaded by path, so each runner works inside its model's own virtualenv. |
| `baselines/native/run_alpamayo.py` | Native runner for Alpamayo-R1-10B / Alpamayo-1.5-10B. |
| `baselines/native/run_autovla.py` | Native runner for AutoVLA. |
| `baselines/native/run_impromptu.py` | Native runner for Impromptu-VLA-7B. |
| `sbatch/*.sbatch` | SLURM launchers for all of the above (see the last section). |

## Paths and environment

Locations come from `training/core/paths.py` and are all overridable:

| Variable | Used for |
| --- | --- |
| `ANCHOR_ROOT` | repository root |
| `WAYMO_VAL_ROOT` | decoded validation frames; `rater_feedback_frames/` and `val_sequence_name_to_scenario_cluster.json` live under it |
| `ANCHOR_DATA` | generated index / cache JSON (`dev_eval_frames.json`, `calib_fixed.json`) |
| `ANCHOR_EVAL` | default parent of evaluation output dirs |
| `ANCHOR_WEIGHTS` | downloaded base-model snapshots (native runners, zero-shot runs) |
| `ANCHOR_RATED_INDEX` | JSONL index of rated validation frames, used to locate a frame outside the flattened folder |
| `WAYMO_METRICS_SRC` | directory holding `rater_feedback_utils.py` (default `<root>/third_party/waymo_metrics`); `run_rfs.py` imports the official `get_rater_feedback_score` from the import path first, then from here |
| `WAYMO_CALIB_JSON` | camera calibration used by the renderer (default `<ANCHOR_DATA>/calib_fixed.json`) |
| `OPENAI_API_KEY` / `OPENAI_API_KEY_FILE` | credentials for the LLM judges |
| `JUDGE_MODEL` | default judge model name |

## Dev-set evaluation

`eval_dev.py` generates on a frame list with either the Stage-1 (understanding) or the Stage-2
(full chain) prompt and prints a one-screen summary. It also verifies the `prompt_hash.json` stored
with the checkpoint against the prompt the code would build now, and exits with status 2 on a
mismatch unless `--no-hash-check` is passed.

```bash
# Stage-2 chain on the internal dev split
python -m evaluation.eval_dev \
  --ckpt runs/<run>/best/best_stage_s2 \
  --stage s2 \
  --frames data/dev_eval_frames.json \
  --out eval_results/<run>/dev_s2_metrics.json

# Stage-1 understanding on the rated frames, bucketed by scenario cluster, with the LLM judge
python -m evaluation.eval_dev \
  --ckpt runs/<run>/best/best_stage_s1 \
  --stage s1 --frames rfs --by-cluster --judge \
  --out eval_results/<run>/dev_s1_metrics.json

# Zero-shot: load a backbone's base weights through its adapter
python -m evaluation.eval_dev --ckpt base --adapter qwen25_7b --stage s2 \
  --frames rfs --subset 2 --max-new-tokens 512 --no-hash-check \
  --out /tmp/smoke_qwen25.json
```

Produces `--out` (the metrics JSON) and, next to it, `<out>_chains.json` — one compact row per frame
with the raw generated text, the parsed chain, the 20-point prediction and its displacement errors.
Without `--out` the metrics are printed only.

Other options: `--adapter` (defaults to `<ckpt>/adapter_name.json`), `--subset N`, `--long-edge`,
`--max-new-tokens`, `--no-repeat-ngram`, `--attn`, `--device`, `--judge-model`, `--key-file`.

## Rated-validation evaluation

`eval_val456.py` is the main result-producing entry point. One Stage-2 pass over the rated
validation frames, then the official rater-feedback score for that pass and for four reference rows
(`gt`, `cv`, `static`, `ceiling`), a paired per-frame bootstrap CI of the model against each
reference, and the acceptance gates.

```bash
python -m evaluation.eval_val456 \
  --ckpt runs/<run>/best/best_stage_s2 \
  --run <run>_best_stage_s2 \
  --judge

# faster loop while iterating: no reference rows, no judges
python -m evaluation.eval_val456 --ckpt runs/<run>/best/best_stage_s2 \
  --subset 32 --no-reference --out-dir /tmp/val456_smoke
```

Output directory (`--out-dir`, default `$ANCHOR_EVAL/<run>`):

```
chains.json        per-frame rows of the generation pass, including the raw text
rfs_preds.json     [{scene_id, frame_id, pred_xy(20)}]; frames without a trajectory get zeros
metrics.json       {s2, rfs, reference, gates, ...}
official/          run_rfs.py file set for the model pass
reference/<kind>/  run_rfs.py file set per reference row
```

Other options: `--adapter`, `--frames`, `--subset`, `--max-new-tokens`, `--no-repeat-ngram`,
`--long-edge`, `--attn`, `--device`, `--no-rfs`, `--n-boot`, `--cluster-json`, `--judge-model`,
`--key-file`, `--no-hash-check`.

## Official rater-feedback score and displacement errors

`run_rfs.py` is the scoring half on its own: it builds the official metric inputs from a predictions
file and scores them with the vendored Waymo function. Use it to re-score an existing
`rfs_preds.json`, to score predictions produced elsewhere, or to score a reference row.

Input is either `[{"scene_id", "frame_id", "pred_xy"}]` or the official `{"results": [...]}` form.
`pred_xy` may be `null`/empty (stationary fallback, counted as a missing prediction) or shorter than
20 points, in which case it is linearly resampled onto the 0.25 s grid.

```bash
# score a prediction file
python -m evaluation.run_rfs \
  --preds eval_results/<run>/rfs_preds.json \
  --out-dir eval_results/<run>/official

# predictions on a coarser grid whose first point is the current position
python -m evaluation.run_rfs --preds my_preds.json --out-dir /tmp/rfs_mine \
  --src-dt 0.5 --has-origin

# reference rows: gt | cv | static | ceiling
python -m evaluation.run_rfs --reference ceiling --out-dir /tmp/rfs_ceiling
```

Writes into `--out-dir`:

```
results_for_official.json    the predictions in the official results format
rater_feedback_inputs.json   20-point predictions + rater trajectories/scores + initial speed
per_sample_results.jsonl     per-frame ADE@1/3/5 s, FDE@5 s, intent, speed, RFS, cluster
metrics_summary.json         aggregate ADE/FDE plus the rater_feedback block
rater_feedback_results.json  per-frame RFS and the per-cluster table
```

It prints both aggregations: the frame mean and the leaderboard cluster-mean.
`--cluster-json`, `--subset` and `--tag` are also accepted.

## Chain causal-intervention experiments

`ci_eval.py` reuses the decoding protocol of `eval_dev.py` with a teacher-forced assistant prefix.

* **CI-1** — take the model's own chain, keep everything up to `<final_plan>`, substitute a
  counterfactual plan and motion, force `<traj>`, and measure how often the emitted trajectory
  follows the injected class. A control re-forces the *original* plan and motion.
* **CI-2** — full chain versus a prompt with `<traj>` forced immediately, compared by displacement
  error, start-frame statistics and rater-feedback score with paired CIs.
* **CI-3** — ground-truth prefix upper bound, at the `understanding` level (GT ego state, scene and
  objects forced) and the `chain` level (the whole GT chain through `<motion>` forced).

```bash
# all three, reusing the Stage-2 pass that eval_val456 already wrote
python -m evaluation.ci_eval \
  --ckpt runs/<run>/best/best_stage_s2 \
  --base-dir eval_results/<run> \
  --out-dir eval_results/<run>/ci

# CI-1 only, every alternative trend class per frame
python -m evaluation.ci_eval --ckpt runs/<run>/best/best_stage_s2 \
  --which 1 --ci1-targets all --base-dir eval_results/<run>

# CI-3 restricted to one level, no rater-feedback scoring
python -m evaluation.ci_eval --ckpt runs/<run>/best/best_stage_s2 \
  --which 3 --ci3-levels chain --no-rfs --subset 64
```

Without `--base-dir` the base Stage-2 pass is regenerated. Outputs in `--out-dir` (default
`$ANCHOR_EVAL/<run>/ci`): `ci_metrics.json`, `base_rows.json`, `ci1_rows.json`,
`ci2_<variant>_rows.json`, `ci3_<level>_rows.json`, and one `rfs_<variant>/` file set per variant
unless `--no-rfs`.

Other options: `--adapter`, `--run`, `--frames`, `--subset`, `--max-new-tokens`, `--long-edge`,
`--attn`, `--device`, `--n-boot`, `--cluster-json`, `--no-hash-check`.

## CPU re-scoring / re-judging

Everything except the judge columns is recomputed deterministically from the stored text, so a
finished run can be re-scored without a GPU and without regenerating anything. `rejudge.py` rebuilds
records from the saved chains, re-scores them, and writes the metrics file back in place, keeping the
previous copy as `*.pre_rejudge.json`. Judge results that the old file had and the fresh pass lost
(API failures) are carried over.

```bash
# Stage-2 run directory written by eval_val456: metrics.json + chains.json
python -m evaluation.rejudge --run-dir eval_results/<run> --judge

# Stage-1 dev run: dev_s1_metrics.json + dev_s1_metrics_chains.json
python -m evaluation.rejudge --run-dir eval_results/<run> --stage s1 --judge
```

Other options: `--judge-model`, `--key-file`, `--subset`, `--cluster-json`.

## Released-model baselines

Released models are scored on exactly the capabilities their own output shows. There are two ways to
get their generations, both ending in a `chains.json` and both scored by the same `rescore` step.

**Adapter path** — the model's base weights are loaded through its adapter and prompted with this
repository's Stage-2 prompt, so the run is directly comparable with a fine-tuned checkpoint:

```bash
python -m evaluation.eval_val456 --ckpt base --adapter qwen25_7b \
  --run zs_qwen25_7b --out-dir eval_results/zs_qwen25_7b
```

**Native path** — the model runs through its own published inference code (its own prompt, sampling
and action head) and writes rows in the same `chains.json` schema. Each runner needs its model's own
virtualenv and released source checkout, and is invoked as a script (not as a module), because it
must not import this repository's `training` package:

```bash
export ANCHOR_ROOT=$PWD ANCHOR_WEIGHTS=$PWD/weights/base

# Alpamayo-R1-10B / Alpamayo-1.5-10B
$ALPAMAYO_PYTHON evaluation/baselines/native/run_alpamayo.py \
  --model alpamayo_15_10b --repo $ALPAMAYO_REPO \
  --out eval_results/zs_alpamayo_15_10b/chains.json

# AutoVLA
$AUTOVLA_PYTHON evaluation/baselines/native/run_autovla.py \
  --repo $AUTOVLA_REPO --out eval_results/zs_autovla_3b/chains.json

# Impromptu-VLA-7B (front third of the panorama by default)
$IMPROMPTU_PYTHON evaluation/baselines/native/run_impromptu.py \
  --out eval_results/zs_impromptu_7b/chains.json
```

Common runner options: `--index` (JSONL index of the rated frames), `--subset N`,
`--max-new-tokens`, `--seed`, `--no-resume`. Rows are appended to `<out>.partial.jsonl` as they are
produced, so an interrupted run resumes by default. Per-runner extras: `--model-dir`, `--repo`,
`--device`, `--top-p`, `--top-k`, `--temperature` (alpamayo); `--ckpt`, `--repo`, `--crop-dir`
(autovla); `--model-dir`, `--full-panorama` (impromptu).

**Scoring step** — `baselines.rescore` runs each row's text through the tolerant extractor, rebuilds
records, scores them with the shared code, and then lets `capabilities.py` decide from the output
itself what is reported: a field that is absent or constant across frames becomes `None` rather than
zero, and object types without image points are scored geometry-free through `typeonly`. It is
CPU-only:

```bash
python -m evaluation.baselines.rescore \
  --chains eval_results/zs_qwen25_7b/chains.json \
  --profile qwen25_7b \
  --out eval_results/zs_qwen25_7b/baseline \
  --judge

# re-mask without spending API credits: take the judge verdicts from the previous run
cp eval_results/zs_qwen25_7b/baseline/metrics_baseline.json /tmp/prev_qwen25.json
python -m evaluation.baselines.rescore --chains eval_results/zs_qwen25_7b/chains.json \
  --profile qwen25_7b --out eval_results/zs_qwen25_7b/baseline \
  --reuse-judges /tmp/prev_qwen25.json
```

Writes `metrics_baseline.json` (metrics plus the `capability` block recording every decision and its
evidence), `chains_baseline_parsed.json`, and an `official/` rater-feedback file set when the model
produces trajectories. Profile names: `qwen25_7b`, `qwen3vl_8b`, `cosmos_2b`, `cosmos_8b`,
`alpamayo_r1_10b`, `alpamayo_15_10b`, `autovla_3b`, `impromptu_7b`. Other options: `--subset`,
`--judge-model`, `--key-file`, `--no-rfs`, `--cluster-json`.

## Per-frame visualization

`render_rfs_viz.py` renders one PNG per frame from a finished `eval_val456` run — the panorama with
the ground-truth, predicted and rater trajectories, an intent banner, a BEV inset and a text box
carrying the plan, the reason and the emitted objects. No GPU is needed.

It reads `<run>/official/rater_feedback_inputs.json`, `<run>/official/per_sample_results.jsonl` and
`<run>/chains.json`, and needs the camera calibration at `$WAYMO_CALIB_JSON`.

```bash
# everything, into <run>/viz/
python -m evaluation.render_rfs_viz --eval-dir eval_results/<run>

# a handful of named frames, with an explicit label and output directory
python -m evaluation.render_rfs_viz --eval-dir eval_results/<run> \
  --clips <scene>_<frame>,<scene>_<frame> \
  --model-name "<run>" --out-dir /tmp/viz
```

`--limit N` renders the first N frames instead.

## LLM judges

Two free-text fields are scored by an LLM judge: `implication` (per object) and `reason` (per frame).
Both need an OpenAI key, resolved as `OPENAI_API_KEY`, else the file named by `--key-file` /
`OPENAI_API_KEY_FILE`, else `<repo root>/api_key.txt`. `--judge-model` (or `JUDGE_MODEL`) selects the
model. API errors drop the item rather than scoring it zero.

Judges are used only where explicitly requested:

| Entry point | Flag | Judged |
| --- | --- | --- |
| `eval_dev` | `--judge` | implication |
| `eval_val456` | `--judge` | implication and reason |
| `rejudge` | `--judge` | implication; reason as well with `--stage s2` |
| `baselines.rescore` | `--judge` | implication and reason |

Everything else — `run_rfs`, `ci_eval`, `render_rfs_viz` — runs without a key.

## Batch scripts

The launchers in `sbatch/` are SLURM wrappers around the entry points above. Everything up to
`set -uo pipefail` in each file is site-specific: replace the placeholder partition / QoS / account,
size the resource requests for your node, and `mkdir -p logs` before submitting. They read `ROOT`
from `ANCHOR_ROOT` or the submit directory, and the interpreter from `ANCHOR_PYTHON`.

| Script | Purpose |
| --- | --- |
| `eval_h100.sbatch`, `eval_b200.sbatch` | Evaluate a trained checkpoint. `MODE=val456\|dev\|ci\|all` (`eval_h100` also has `s1full` = `eval_dev --stage s1 --frames rfs --judge --by-cluster`). The B200 variant additionally gates on CUDA readiness and requeues instead of crashing on a freshly rebooted node. |
| `eval_l40s.sbatch` | Zero-shot run of a released backbone through its adapter: `eval_val456 --ckpt base`, then `baselines.rescore`. `MODE=all\|val456\|rescore`. Stages the base weights on node-local disk first. |
| `native_l40s.sbatch`, `native_b200.sbatch` | Native-pipeline zero-shot run: `baselines/native/run_<model>.py`, then `baselines.rescore`. `MODE=all\|real\|rescore`. Each model uses its own interpreter (`ALPAMAYO_PYTHON` / `AUTOVLA_PYTHON` / `IMPROMPTU_PYTHON`); the rescore step always uses the project interpreter. |
| `rescore_cpu.sbatch` | CPU-only rerun of `baselines.rescore` on an existing `chains.json`. `REUSE_JUDGES=1` reuses the stored judge verdicts. |
| `rejudge_cpu.sbatch` | CPU-only rerun of `evaluation.rejudge` on a finished run directory. |
| `smoke_all_adapters.sbatch` | Loads every backbone's base weights through its adapter and generates two frames each, as a wiring check. |

```bash
mkdir -p logs

# trained checkpoint: rated-validation evaluation, then the intervention experiments on its chains
CKPT=runs/<run>/best/best_stage_s2 MODE=all \
  sbatch -J eval_s2 evaluation/sbatch/eval_h100.sbatch --judge

# dev-set Stage-1 pass
CKPT=runs/<run>/best/best_stage_s1 MODE=dev STAGE=s1 FRAMES=data/dev_eval_frames.json \
  sbatch -J dev_s1 evaluation/sbatch/eval_h100.sbatch

# zero-shot released backbone through its adapter
ADAPTER=qwen25_7b sbatch -J zs_qwen25_7b evaluation/sbatch/eval_l40s.sbatch

# zero-shot released model through its own pipeline
MODEL=alpamayo_15_10b sbatch -J zs_alpamayo_15 evaluation/sbatch/native_l40s.sbatch

# scoring-only reruns, no GPU
RUN=zs_qwen25_7b PROFILE=qwen25_7b REUSE_JUDGES=1 \
  sbatch -J rescore_qwen25 evaluation/sbatch/rescore_cpu.sbatch
RUN=eval_results/<run> STAGE=s2 sbatch -J rejudge_s2 evaluation/sbatch/rejudge_cpu.sbatch
```

Extra command-line arguments after the script name are forwarded to the python entry point
(`sbatch ... evaluation/sbatch/eval_h100.sbatch --subset 32 --no-reference`). `NOHASH=1` downgrades a
`prompt_hash.json` mismatch to a warning; `OUT=<dir>` overrides the derived output directory; `JUDGE=0`
turns the judges off in the launchers that enable them by default.
