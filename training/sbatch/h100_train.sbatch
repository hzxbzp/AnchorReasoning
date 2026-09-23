#!/bin/bash -l
# =============================================================================
# Stage-1 / Stage-2 supervised fine-tuning launcher for one multi-GPU node.
# The defaults below are sized for 4 x 80 GB GPUs. For an INTERACTIVE allocation use
# training/run_in_alloc.sh instead -- this file is only for `sbatch`.
#
#   CONFIG=<yaml> sbatch -J <name> training/sbatch/h100_train.sbatch [key=value ...]
#   e.g. CONFIG=training/configs/qwen25_7b_s1.yaml sbatch -J qwen25_s1 training/sbatch/h100_train.sbatch
#        CONFIG=training/configs/qwen25_7b_s2.yaml sbatch -J qwen25_s2 training/sbatch/h100_train.sbatch
#   Submit from the repository root, or export ANCHOR_ROOT=<repo root> beforehand.
#   Any trailing `key=value` argument is forwarded to the trainer as a config override.
#
# Optional environment knobs:
#   GPUS (default 4)      processes to launch; must match the --gres GPU count below.
#   RESUME_SRC=<dir>      resume from THAT checkpoint dir instead of the newest checkpoint-N in
#                         the run dir (e.g. to re-test a checkpoint while writing the outputs
#                         to a throwaway trainer.output_dir).
#   LOCAL_COPY_SRC=<dir>  copy that dir to node-local scratch and use it as `init_from`.
#   STAGE_BASE=0          do not stage the backbone weights on node-local scratch.
#   STAGE_RESUME=0        do not stage the checkpoint being resumed on node-local scratch.
#   STAGE_TIMEOUT (2400)  seconds a staging copy may take before the job requeues itself.
#   NO_REQUEUE=1          exit 1 instead of requeueing when training stops without DONE.
#   MIN_RUN_SECS (2400)   a run that dies inside this window without saving a new checkpoint is
#                         treated as a deterministic failure and is NOT requeued (loop guard).
#
# Rails: SIGUSR1 is forwarded to the python workers 15 min before the walltime so the trainer
# saves and stops gracefully, CUDA readiness is probed before launching, and the job requeues
# itself until the run dir holds a DONE marker.
# Universal checkpoints: if the checkpoint being resumed carries `latest_universal` (written by
# training/sbatch/ds_to_universal.sbatch) the launcher switches this launch to
# training/configs/deepspeed/zero2_universal.json automatically -- that is how a ZeRO-2 run saved
# with N GPUs continues on M GPUs (DeepSpeed refuses per-rank shards across a data-parallel
# world-size change). Checkpoints saved afterwards are ordinary ones, so later requeues fall back
# to the config's zero2.json.
#
# -----------------------------------------------------------------------------
# CLUSTER SETTINGS -- EDIT THIS BLOCK FOR YOUR SITE
#   * #SBATCH lines: account, partition, GPU type and count, CPUs, memory and walltime.
#   * --qos: some partitions require an explicit QoS, others reject it and apply their default.
#     Uncomment and fill it in only if your site needs it.
#   * --output/--error are relative to the directory you submit from; create it first
#     (`mkdir -p logs`) or point them somewhere else.
#   * Mail options are left out on purpose; add e.g. `#SBATCH --mail-type=END,FAIL` if wanted.
#   * CUDA_MODULE / PYTHON_BIN / HF_CACHE_DIR below: how the CUDA toolchain and the python
#     environment holding the dependencies are made available on a compute node.
#   * LOCAL_SCRATCH (default /lscratch): the node-local disk weights are staged on. Set it to
#     whatever your site calls that directory; staging is skipped when it cannot be created.
# -----------------------------------------------------------------------------
#SBATCH --account=<slurm-account>
#SBATCH --partition=<gpu-partition>
##SBATCH --qos=<qos>
#SBATCH --gres=gpu:<gpu-type>:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT="${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
PKG="$ROOT/training"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"          # module providing nvcc; set empty to skip
PYTHON_BIN="${PYTHON_BIN:-python}"                 # interpreter of the environment with the deps
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/hf_cache}"     # Hugging Face / triton / extension caches
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-anchor-sft}"   # used if the config sets no project
GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-9.0}"            # 9.0 = H100, 8.9 = L40S, 10.0 = B200
# =============================================================================

PY="$PYTHON_BIN"
GPUS=${GPUS:-4}
CONFIG=${CONFIG:?set CONFIG=<path to training/configs/<model>_<stage>.yaml>}
case "$CONFIG" in /*) ;; *) CONFIG=$ROOT/$CONFIG ;; esac
cd "$ROOT" || exit 2
LOGDIR="${ANCHOR_LOGS:-$ROOT/logs}"
mkdir -p "$LOGDIR"

[ -n "$CUDA_MODULE" ] && module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export ANCHOR_ROOT="$ROOT"
export HF_HOME=$HF_CACHE_DIR
export HF_HUB_CACHE=$HF_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # compute nodes without outbound network
export TRITON_CACHE_DIR=$HF_CACHE_DIR/.triton_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=8
# the interpreter's bin dir on PATH: torch.cpp_extension needs the `ninja` executable
# (DeepSpeed builds CPU-Adam just in time for the offload config)
case "$PY" in */*) export PATH="$(dirname "$PY"):$PATH" ;; esac
export TORCH_EXTENSIONS_DIR=$HF_CACHE_DIR/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at the ZeRO-2 optimizer step

# resolve run dir / name + wandb project from the config + the key=value overrides passed after
# the script name (config.py imports no torch -> instant). --smoke changes the run dir to <run>_smoke
# (train_stage.apply_smoke). Line 1 = trainer.output_dir, line 2 = wandb.project.
OV=(); SMOKE=0
for a in "$@"; do case "$a" in --smoke) SMOKE=1 ;; --*) ;; *=*) OV+=("$a") ;; esac; done
CFGINFO=$("$PY" -c "import sys; from training.core.config import load_config; c = load_config(sys.argv[1], sys.argv[2:]); print(c['trainer']['output_dir']); print((c.get('wandb') or {}).get('project') or '$WANDB_PROJECT_DEFAULT')" "$CONFIG" "${OV[@]}") || { echo "[sbatch] config failed to load: $CONFIG"; exit 1; }
{ read -r OUT; read -r WBP; } <<< "$CFGINFO"
[ "$SMOKE" -eq 1 ] && OUT="${OUT}_smoke"
RUN=$(basename "$OUT")
# WANDB_PROJECT is exported EXPLICITLY: an already exported value wins, then the yaml
# `wandb.project`, then WANDB_PROJECT_DEFAULT. train_stage only setdefault()s it.
export WANDB_PROJECT=${WANDB_PROJECT:-$WBP}
export WANDB_DIR=$LOGDIR/wandb
export WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN}_$USER}   # fixed id -> ONE wandb run across requeues; _$USER so accounts sharing the wandb login never resume each other's run
export WANDB_RESUME=allow
mkdir -p "$WANDB_DIR"

echo "===== $(hostname) $(date) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} GPUS=$GPUS CONFIG=$CONFIG OUT=$OUT ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -f "$OUT/DONE" ]; then echo "training already DONE ($OUT), nothing to do"; exit 0; fi

# Single-writer guard: the same run dir may have several queued jobs (e.g. one per partition,
# whichever gets a node first). Two trainers writing $OUT would corrupt checkpoints, so the first
# one to start takes a lock and the others exit cleanly (no requeue -- resubmit if needed).
LOCK="$OUT/RUNNING.lock"
if [ -f "$LOCK" ]; then
  OTHER=$(head -1 "$LOCK" 2>/dev/null)
  if [ -n "$OTHER" ] && [ "$OTHER" != "${SLURM_JOB_ID:-}" ] && [ -n "$(squeue -h -j "$OTHER" -t R 2>/dev/null)" ]; then
    echo "[sbatch] job $OTHER is already training in $OUT -> exiting (no requeue) $(date)"
    exit 0
  fi
  echo "[sbatch] stale lock (job ${OTHER:-?} not running) -> taking over $(date)"
fi
mkdir -p "$OUT"; echo "${SLURM_JOB_ID:-manual}" > "$LOCK"
trap 'rm -f "$LOCK"; [ -n "${LSCRATCH:-}" ] && rm -rf "$LSCRATCH"' EXIT

EXTRA=()
# Stage the BASE weights on node-local NVMe when this run starts from them (init_from empty).
# safetensors are loaded through mmap, and on some nodes mmap page faults against the shared
# filesystem collapse to a few MB/s (a 17 GB backbone then takes half an hour to load), while a
# sequential `cp` streams at tens of MB/s and local NVMe mmap is fast.
# STAGE_BASE=0 disables. The adapter picks the copy up via ANCHOR_BASE_PATH.
LSCRATCH=${LOCAL_SCRATCH:-/lscratch}/$SLURM_JOB_ID          # SLURM does not create this -- we do (and clean it up on exit)
mkdir -p "$LSCRATCH" 2>/dev/null || LSCRATCH=""
if [ "${STAGE_BASE:-1}" = "1" ] && [ -n "$LSCRATCH" ]; then
  BASE_SRC=$("$PY" -c "import sys; from training.adapters import get_adapter; from training.core.config import load_config; c=load_config(sys.argv[1], sys.argv[2:]); print('' if c.get('init_from') else get_adapter(c['adapter']).base_path)" "$CONFIG" "${OV[@]}" 2>/dev/null)
  if [ -n "$BASE_SRC" ] && [ -d "$BASE_SRC" ]; then
    BASE_DST=$LSCRATCH/base-$(basename "$BASE_SRC")
    echo "[sbatch] staging base weights $BASE_SRC ($(du -shL "$BASE_SRC" 2>/dev/null | cut -f1)) -> $BASE_DST $(date)"
    # Storage gate: when the shared filesystem is unhealthy this copy stalls at 0 B/s and the run
    # would hang for hours holding the GPUs. Bound it and requeue instead of starting doomed.
    # Integrity gate: compare the total bytes of regular files, source dereferenced (`find -L`) so a
    # hub-cache snapshot of symlinks is measured at its blob sizes. Adapter-agnostic on purpose:
    # checking for a specific file (e.g. config.json) requeue-loops backbones whose snapshot holds
    # only a checkpoint file and takes its processor config from another directory.
    stage_bytes() { find ${2:+-L} "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
    if mkdir -p "$BASE_DST" && timeout "${STAGE_TIMEOUT:-2400}" cp -rL "$BASE_SRC"/. "$BASE_DST"/ \
       && SRC_B=$(stage_bytes "$BASE_SRC" deref) && DST_B=$(stage_bytes "$BASE_DST") \
       && [ "${SRC_B:-0}" -gt 0 ] && [ "$SRC_B" = "$DST_B" ] ; then
      export ANCHOR_BASE_PATH="$BASE_DST"; echo "[sbatch] base stage done ($SRC_B bytes) $(date)"
    else
      echo "[sbatch] base staging FAILED/TIMED OUT (shared filesystem unhealthy?) -> requeue, retry in 30 min $(date)"
      rm -rf "$BASE_DST"
      scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true
      scontrol update JobId="$SLURM_JOB_ID" StartTime=now+30minutes 2>/dev/null || true
      exit 0
    fi
  fi
fi

# Resume staging: reading a checkpoint's DeepSpeed optimizer states (~100 GB for an 8B model) off
# the shared filesystem can crawl at a few MB/s, while a sequential `cp` runs at GB/s. Copy the
# latest checkpoint to node-local scratch and resume from there.
# STAGE_RESUME=0 disables. The trainer still SAVES to $OUT on the shared filesystem.
if [ "${STAGE_RESUME:-1}" = "1" ] && [ -n "$LSCRATCH" ] && [ -d "$OUT" ]; then
  LAST_CK=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1)
  SRC=""
  if [ -n "${RESUME_SRC:-}" ]; then SRC="${RESUME_SRC%/}"; echo "[sbatch] RESUME_SRC given -> resuming from $SRC (not the run dir's newest checkpoint)"
  elif [ -n "$LAST_CK" ]; then SRC="$OUT/checkpoint-$LAST_CK"; fi
  if [ -n "$SRC" ] && [ -d "$SRC" ] && [ -d "$LSCRATCH" ]; then
    DST=$LSCRATCH/resume-$(basename "$SRC")
    echo "[sbatch] staging $SRC ($(du -shL "$SRC" 2>/dev/null | cut -f1)) -> $DST $(date)"
    if mkdir -p "$DST" && timeout "${STAGE_TIMEOUT:-2400}" cp -rL "$SRC"/. "$DST"/ ; then
      export RESUME_FROM="$DST"; echo "[sbatch] stage done $(date)"
    else
      echo "[sbatch] resume staging FAILED/TIMED OUT (shared filesystem unhealthy?) -> requeue, retry in 30 min $(date)"
      rm -rf "$DST"
      scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true
      scontrol update JobId="$SLURM_JOB_ID" StartTime=now+30minutes 2>/dev/null || true
      exit 0
    fi
  fi
fi
# RESUME_SRC with staging disabled/unavailable: hand the shared-filesystem dir to the trainer directly.
if [ -n "${RESUME_SRC:-}" ] && [ -z "${RESUME_FROM:-}" ] && [ -d "${RESUME_SRC%/}" ]; then export RESUME_FROM="${RESUME_SRC%/}"; fi

# Universal-checkpoint auto-switch (see header). Decide from the directory the trainer will actually
# read: the staged copy if there is one, else the newest checkpoint-N in the run dir.
RES_DIR="${RESUME_FROM:-}"
[ -z "$RES_DIR" ] && [ -n "${LAST_CK:-}" ] && RES_DIR="$OUT/checkpoint-$LAST_CK"
if [ -n "$RES_DIR" ] && [ -f "$RES_DIR/latest_universal" ]; then
  case " $* " in *" trainer.deepspeed="*) echo "[sbatch] universal checkpoint in $RES_DIR but trainer.deepspeed given on the command line -> not overriding" ;;
    *) EXTRA+=("trainer.deepspeed=$PKG/configs/deepspeed/zero2_universal.json")
       echo "[sbatch] universal checkpoint ($(cat "$RES_DIR/latest_universal")) in $RES_DIR -> loading with zero2_universal.json (saved with $(ls "$RES_DIR"/global_step*/ 2>/dev/null | grep -c optim_states) per-rank shards, resuming on GPUS=$GPUS)" ;;
  esac
fi

# keep the effective batch at 64 frames/step regardless of GPU count (1 x GPUS x accum = 64).
# ZeRO-2 full fine-tuning of an 8B backbone needs ~46 GB/GPU with 4 GPUs (fits an 80 GB card) and
# ~75 GB with 2. Use GPUS=4, or GPUS=2 with configs/deepspeed/zero2_offload.json (optimizer states
# on CPU).
case " $* " in *" trainer.gradient_accumulation_steps="*) ;; *) EXTRA+=("trainer.gradient_accumulation_steps=$((64 / GPUS))") ;; esac
if [ -n "${LOCAL_COPY_SRC:-}" ]; then
  LDST=${LOCAL_SCRATCH:-/lscratch}/$SLURM_JOB_ID/init_ckpt
  mkdir -p "$LDST" && echo "[sbatch] copying $LOCAL_COPY_SRC -> $LDST $(date)"
  cp -rL "$LOCAL_COPY_SRC"/. "$LDST"/ && EXTRA+=("init_from=$LDST") && echo "[sbatch] copy done $(date)"
fi

# Forward walltime SIGUSR1 (B: -> batch shell only) to THIS job's python workers so
# SignalSaveStop checkpoints gracefully (cgroup-scoped; the torchrun agent is skipped --
# it would die with rc=138 on USR1).
forward_usr1() {
  echo "[sbatch] SIGUSR1 -> forwarding to job ${SLURM_JOB_ID} python workers $(date)"
  for p in $(pgrep -u "$USER" -f "train_stage.py"); do
    grep -q "torch.distributed.run" /proc/$p/cmdline 2>/dev/null && continue
    grep -q "job_${SLURM_JOB_ID}" /proc/$p/cgroup 2>/dev/null || continue
    kill -USR1 "$p" 2>/dev/null || true
  done
}
trap forward_usr1 USR1

# CUDA readiness gate: after a node reboot the driver stack can stay un-initialised for minutes
# while nvidia-smi already looks fine. Starting torch in that window crash-loops the requeue chain.
# Probe from the batch shell (same cgroup/GPUs) and only launch once CUDA really works; otherwise
# requeue with a 30 min delay.
cuda_ready() { "$PY" - <<'PYEOF' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PYEOF
}
for i in $(seq 1 30); do
  if cuda_ready; then echo "[sbatch] CUDA ready (probe $i) $(date)"; break; fi
  if [ "$i" -eq 30 ]; then
    echo "[sbatch] CUDA NOT ready after 15 min -> requeue, retry in 30 min $(date)"
    scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true
    scontrol update JobId="$SLURM_JOB_ID" StartTime=now+30minutes 2>/dev/null || true
    exit 0
  fi
  sleep 30
done

newest_ck() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1; }
CK_BEFORE=$(newest_ck); T0=$(date +%s)
srun --kill-on-bad-exit=1 "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  "$PKG/train_stage.py" --config "$CONFIG" "${EXTRA[@]}" "$@" &
TRAIN_PID=$!
RC=1
while kill -0 $TRAIN_PID 2>/dev/null; do
  wait $TRAIN_PID; RC=$?   # >128 = interrupted by trapped USR1 -> keep waiting
done
echo "===== training exited rc=$RC $(date) ====="

if [ -f "$OUT/DONE" ]; then
  echo "DONE marker present -> finished."; exit 0
elif [ "${NO_REQUEUE:-0}" = "1" ]; then
  echo "not finished and NO_REQUEUE=1 -> exit 1 (no requeue)."; exit 1
elif [ "$(newest_ck)" = "$CK_BEFORE" ] && [ $(( $(date +%s) - T0 )) -lt "${MIN_RUN_SECS:-2400}" ]; then
  # Died within MIN_RUN_SECS without saving anything: a deterministic failure (bad resume, OOM at
  # step 1, config error), not walltime or preemption. Requeueing would loop forever.
  echo "not finished, NO new checkpoint, exited after $(( $(date +%s) - T0 ))s -> deterministic failure, NOT requeueing (fix, then resubmit)."
  exit 1
else
  echo "not finished -> requeue for another walltime window."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
