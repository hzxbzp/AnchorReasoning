#!/bin/bash
# =============================================================================
#  Run one or more training stages INSIDE an existing allocation (salloc / srun --pty, or
#  whatever interactive wrapper your site provides). `sbatch` users want
#  training/sbatch/h100_train.sbatch; this script is the equivalent for a long-lived
#  interactive node.
#
#   1) on the LOGIN node, always start a tmux session first -- an interactive allocation
#      dies with its terminal, and a dropped ssh would kill a multi-day node:
#        tmux new -s train
#   2) grab the node, e.g.:
#        salloc -A <slurm-account> -p <gpu-partition> --gres=gpu:<gpu-type>:4 \
#               -c 56 --mem 480G -t 48:00:00
#   3) inside the allocation:
#        bash training/run_in_alloc.sh qwen25_7b_s1 qwen25_7b_s2
#      (detach with ctrl-b d; re-attach later with `tmux attach -t train`)
#
# Stages run sequentially and are idempotent: a stage whose run dir already has a DONE
# marker is skipped, and a stage with checkpoints resumes from the latest one.  A stage
# that fails stops the chain (the next stage would start from a wrong/missing init).
#
# Usage: run_in_alloc.sh [--dry-run] [--gpus N] <config-name> [<config-name> ...]
#   <config-name> = basename in training/configs, e.g. qwen25_7b_s1  (".yaml" optional)
#   extra `key=value` args are passed through to train_stage.py for EVERY stage.
#   STAGE_BASE=0 / STAGE_INIT=0 / STAGE_RESUME=0 switch off the node-local staging copies.
#
# -----------------------------------------------------------------------------
# CLUSTER SETTINGS -- EDIT THIS BLOCK FOR YOUR SITE
#   CUDA_MODULE / PYTHON_BIN / HF_CACHE_DIR below: how the CUDA toolchain and the python
#   environment holding the dependencies are made available on a compute node.
#   LOCAL_SCRATCH (default /lscratch): the node-local disk weights are staged on. Set it to
#   whatever your site calls that directory; staging is skipped when it cannot be created.
# -----------------------------------------------------------------------------
set -uo pipefail
ROOT="${ANCHOR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repository root
PKG="$ROOT/training"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"          # module providing nvcc; set empty to skip
PYTHON_BIN="${PYTHON_BIN:-python}"                 # interpreter of the environment with the deps
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/hf_cache}"     # Hugging Face / triton caches
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-anchor-sft}"   # used if the config sets no project
GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-9.0}"            # 9.0 = H100, 8.9 = L40S, 10.0 = B200
# =============================================================================
PY="$PYTHON_BIN"

DRY=0; GPUS_OVERRIDE=""; STAGES=(); OVERRIDES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --gpus) shift; GPUS_OVERRIDE="$1" ;;
    *=*) OVERRIDES+=("$1") ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) STAGES+=("${1%.yaml}") ;;
  esac
  shift
done
[ ${#STAGES[@]} -eq 0 ] && { echo "no stage given; see --help"; exit 2; }

# ---- allocation sanity -------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: not inside a SLURM allocation. Grab a node first (see the header)."; exit 2
fi
GPUS=${GPUS_OVERRIDE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}
[ -z "$GPUS" ] || [ "$GPUS" -lt 1 ] 2>/dev/null && { echo "ERROR: no GPU visible"; exit 2; }
ACCUM=$(( 64 / GPUS ))                       # keep the effective batch at 64 frames/step
[ $(( ACCUM * GPUS )) -ne 64 ] && echo "WARNING: 64 is not divisible by GPUS=$GPUS -> effective batch $(( ACCUM * GPUS ))"

# ---- environment (same as the sbatch rails) ---------------------------------
[ -n "$CUDA_MODULE" ] && { module load "$CUDA_MODULE" 2>/dev/null || true; }
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export ANCHOR_ROOT="$ROOT"
export HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$HF_CACHE_DIR/.triton_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at the ZeRO-2 optimizer step (same as h100_train.sbatch)
LOGDIR="${ANCHOR_LOGS:-$ROOT/logs}"
export WANDB_DIR=$LOGDIR/wandb; mkdir -p "$WANDB_DIR" "$LOGDIR"
cd "$ROOT" || exit 2

echo "===== run_in_alloc  $(hostname)  job=$SLURM_JOB_ID  GPUS=$GPUS  accum=$ACCUM  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "stages: ${STAGES[*]}    extra overrides: ${OVERRIDES[*]:-<none>}"

run_stage () {
  local NAME="$1" CFG="$PKG/configs/$1.yaml"
  [ -f "$CFG" ] || { echo "[$NAME] ERROR: config not found: $CFG"; return 2; }

  # resolve run dir + wandb project (config.py imports no torch -> instant)
  local INFO OUT WBP
  INFO=$("$PY" -c "import sys; from training.core.config import load_config; c=load_config(sys.argv[1], sys.argv[2:]); print(c['trainer']['output_dir']); print((c.get('wandb') or {}).get('project') or '$WANDB_PROJECT_DEFAULT'); print(c.get('init_from') or '')" \
        "$CFG" "trainer.gradient_accumulation_steps=$ACCUM" "${OVERRIDES[@]}") || { echo "[$NAME] config failed to load"; return 1; }
  { read -r OUT; read -r WBP; read -r INIT; } <<< "$INFO"
  # _$USER: accounts sharing the wandb login must never resume each other's run (same rule as h100_train.sbatch)
  export WANDB_PROJECT="${WANDB_PROJECT:-$WBP}" WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "$OUT")_$USER}" WANDB_RESUME=allow

  echo; echo "########## [$NAME] out=$OUT  $(date)"
  if [ -f "$OUT/DONE" ]; then echo "[$NAME] DONE marker present -> skip"; return 0; fi
  if [ -n "$INIT" ] && [ ! -d "$INIT" ]; then
    echo "[$NAME] ERROR: init_from does not exist: $INIT"
    echo "         (Stage-2 needs the Stage-1 best dir -- did the previous stage finish?)"; return 1
  fi

  # single-writer lock: never let two trainers share a run dir
  local LOCK="$OUT/RUNNING.lock"
  if [ -f "$LOCK" ]; then
    local OTHER; OTHER=$(head -1 "$LOCK" 2>/dev/null)
    if [ -n "$OTHER" ] && [ "$OTHER" != "$SLURM_JOB_ID" ] && [ -n "$(squeue -h -j "$OTHER" -t R 2>/dev/null)" ]; then
      echo "[$NAME] job $OTHER is already training in $OUT -> refusing to start"; return 1
    fi
    echo "[$NAME] stale lock (job ${OTHER:-?}) -> taking over"
  fi
  mkdir -p "$OUT"; echo "$SLURM_JOB_ID" > "$LOCK"

  # base-weight staging (Stage-1 = init_from empty): on some nodes mmap page faults against the
  # shared filesystem collapse to a few MB/s (a 17 GB backbone then takes half an hour to load),
  # a sequential cp streams fine and local NVMe mmap is fast. STAGE_BASE=0 disables.
  unset ANCHOR_BASE_PATH
  local INITOV=()
  local LS=${LOCAL_SCRATCH:-/lscratch}/$SLURM_JOB_ID   # SLURM does not create this -- we do
  mkdir -p "$LS" 2>/dev/null || LS=""
  if [ "${STAGE_BASE:-1}" = "1" ] && [ -z "$INIT" ] && [ -n "$LS" ]; then
    local BSRC BDST
    BSRC=$("$PY" -c "import sys; from training.adapters import get_adapter; from training.core.config import load_config; c=load_config(sys.argv[1]); print(get_adapter(c['adapter']).base_path)" "$CFG" 2>/dev/null)
    if [ -n "$BSRC" ] && [ -d "$BSRC" ]; then
      BDST="$LS/base-$(basename "$BSRC")"
      if [ -d "$BDST" ]; then export ANCHOR_BASE_PATH="$BDST"; echo "[$NAME] reusing staged base $BDST"
      else
        echo "[$NAME] staging base weights $BSRC ($(du -shL "$BSRC" 2>/dev/null | cut -f1)) -> $BDST $(date)"
        # cp -rL: a hub-cache snapshot is a tree of symlinks into ../../blobs/ -- `cp -r` would
        # copy the links and leave them dangling on node-local scratch.  The byte-count gate
        # (source dereferenced) catches a truncated/stalled copy before the GPUs are wasted.
        if mkdir -p "$BDST" && cp -rL "$BSRC"/. "$BDST"/ \
           && BSRC_B=$(find -L "$BSRC" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') \
           && BDST_B=$(find "$BDST" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') \
           && [ "${BSRC_B:-0}" -gt 0 ] && [ "$BSRC_B" = "$BDST_B" ]; then
          export ANCHOR_BASE_PATH="$BDST"; echo "[$NAME] base stage done ($BSRC_B bytes) $(date)"
        else echo "[$NAME] base staging failed/incomplete -> loading from the shared filesystem"; rm -rf "$BDST"; fi
      fi
    fi
  fi

  # init_from staging (Stage-2): the Stage-1 best dir is ~18 GB on the shared filesystem and mmap
  # page faults against it can collapse to a few MB/s -- the sbatch launchers solve this with
  # LOCAL_COPY_SRC; this is the equivalent for an interactive allocation. STAGE_INIT=0 disables.
  case " ${OVERRIDES[*]:-} " in *" init_from="*) ;; *)
    if [ "${STAGE_INIT:-1}" = "1" ] && [ -n "$INIT" ] && [ -d "$INIT" ] && [ -n "$LS" ]; then
      local IDST="$LS/init-$NAME"
      if [ -d "$IDST" ]; then INITOV=("init_from=$IDST"); echo "[$NAME] reusing staged init $IDST"
      else
        echo "[$NAME] staging init_from $INIT ($(du -shL "$INIT" 2>/dev/null | cut -f1)) -> $IDST $(date)"
        if mkdir -p "$IDST" && cp -rL "$INIT"/. "$IDST"/ \
           && ISRC_B=$(find -L "$INIT" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') \
           && IDST_B=$(find "$IDST" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}') \
           && [ "${ISRC_B:-0}" -gt 0 ] && [ "$ISRC_B" = "$IDST_B" ]; then
          INITOV=("init_from=$IDST"); echo "[$NAME] init stage done ($ISRC_B bytes) $(date)"
        else echo "[$NAME] init staging failed/incomplete -> loading from the shared filesystem"; rm -rf "$IDST"; fi
      fi
    fi ;;
  esac

  # resume staging: reading ~100 GB of DeepSpeed optimizer state off the shared filesystem can
  # crawl at a few MB/s; a sequential cp to node-local scratch runs at GB/s.
  unset RESUME_FROM
  if [ "${STAGE_RESUME:-1}" = "1" ] && [ -n "$LS" ]; then
    local LAST DST SRC
    LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1)
    if [ -n "$LAST" ]; then
      SRC="$OUT/checkpoint-$LAST"; DST="$LS/resume-$NAME-$LAST"
      if [ -d "$DST" ]; then export RESUME_FROM="$DST"; echo "[$NAME] reusing staged $DST"
      else
        echo "[$NAME] staging $SRC ($(du -shL "$SRC" 2>/dev/null | cut -f1)) -> $DST $(date)"
        if mkdir -p "$DST" && cp -rL "$SRC"/. "$DST"/; then export RESUME_FROM="$DST"; echo "[$NAME] stage done $(date)"
        else echo "[$NAME] staging failed -> resuming from the shared filesystem"; rm -rf "$DST"; fi
      fi
    fi
  fi

  # Universal-checkpoint auto-switch (same rule as h100_train.sbatch): a checkpoint carrying
  # `latest_universal` (written by training/sbatch/ds_to_universal.sbatch) was saved with a
  # different GPU count and can only be loaded through configs/deepspeed/zero2_universal.json;
  # checkpoints this run saves are ordinary again, so a later resume falls back to the config's
  # zero2.json automatically.
  local RES_DIR="${RESUME_FROM:-}"
  [ -z "$RES_DIR" ] && [ -n "${LAST:-}" ] && RES_DIR="$OUT/checkpoint-$LAST"
  local UNIV=()
  if [ -n "$RES_DIR" ] && [ -f "$RES_DIR/latest_universal" ]; then
    case " ${OVERRIDES[*]:-} " in *" trainer.deepspeed="*) ;;
      *) UNIV=("trainer.deepspeed=$PKG/configs/deepspeed/zero2_universal.json")
         echo "[$NAME] universal checkpoint ($(cat "$RES_DIR/latest_universal")) -> loading with zero2_universal.json on GPUS=$GPUS" ;;
    esac
  fi

  local LOG="$LOGDIR/${NAME}_alloc_${SLURM_JOB_ID}.log"
  echo "[$NAME] log -> $LOG"
  if [ "$DRY" = "1" ]; then
    echo "[$NAME] DRY-RUN: $PY -m torch.distributed.run --standalone --nproc_per_node=$GPUS $PKG/train_stage.py --config $CFG trainer.gradient_accumulation_steps=$ACCUM ${UNIV[*]:-} ${INITOV[*]:-} ${OVERRIDES[*]:-}"
    rm -f "$LOCK"; return 0
  fi

  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
      "$PKG/train_stage.py" --config "$CFG" \
      "trainer.gradient_accumulation_steps=$ACCUM" "${UNIV[@]}" "${INITOV[@]}" "${OVERRIDES[@]}" 2>&1 | tee -a "$LOG"
  local RC=${PIPESTATUS[0]}
  rm -f "$LOCK"
  if [ -f "$OUT/DONE" ]; then echo "[$NAME] finished (DONE written) $(date)"; return 0; fi
  echo "[$NAME] exited rc=$RC without DONE $(date)"
  return "${RC:-1}"
}

FAILED=""
for S in "${STAGES[@]}"; do
  run_stage "$S" || { FAILED="$S"; break; }
done
echo
if [ -n "$FAILED" ]; then
  echo "===== STOPPED at stage '$FAILED' $(date).  Fix it, then re-run the same command --"
  echo "      finished stages are skipped (DONE) and the failed one resumes from its last checkpoint."
  exit 1
fi
echo "===== all stages finished: ${STAGES[*]}  $(date)"
echo "next: evaluate a finished run, e.g."
echo "  CKPT=<run dir>/best/best_stage_s2 sbatch -J eval evaluation/sbatch/eval_h100.sbatch"
