#!/bin/bash -l
# =============================================================================
# Quick GPU check of the training path on a single GPU: `train_stage.py --smoke` runs 3 optimizer
# steps per config with DeepSpeed and the online dev eval switched off and only a handful of
# frames, writing to RUNS/<run_name>_smoke. Use it to verify the environment, the adapter and the
# frame index before launching a real run.
#
#   sbatch training/sbatch/smoke_train_h100.sbatch
#   CONFIGS="qwen25_7b_s1 qwen25_7b_s2" sbatch training/sbatch/smoke_train_h100.sbatch
#   Submit from the repository root, or export ANCHOR_ROOT=<repo root> beforehand.
#
# -----------------------------------------------------------------------------
# CLUSTER SETTINGS -- EDIT THIS BLOCK FOR YOUR SITE
#   * #SBATCH lines: account, partition, GPU type, CPUs, memory and walltime.
#   * --output/--error are relative to the directory you submit from; create it first
#     (`mkdir -p logs`) or point them somewhere else.
#   * Mail options are left out on purpose; add e.g. `#SBATCH --mail-type=END,FAIL` if wanted.
#   * CUDA_MODULE / PYTHON_BIN / HF_CACHE_DIR below: how the CUDA toolchain and the python
#     environment holding the dependencies are made available on a compute node.
# -----------------------------------------------------------------------------
#SBATCH --account=<slurm-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:<gpu-type>:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=smoke_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT="${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"          # module providing nvcc; set empty to skip
PYTHON_BIN="${PYTHON_BIN:-python}"                 # interpreter of the environment with the deps
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/hf_cache}"     # Hugging Face / triton caches
GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-9.0}"            # 9.0 = H100, 8.9 = L40S, 10.0 = B200
# =============================================================================

PY="$PYTHON_BIN"
LOGDIR="${ANCHOR_LOGS:-$ROOT/logs}"
mkdir -p "$LOGDIR"
[ -n "$CUDA_MODULE" ] && module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH" TRITON_CACHE_DIR=$HF_CACHE_DIR/.triton_cache TOKENIZERS_PARALLELISM=false
export ANCHOR_ROOT=$ROOT PYTHONPATH=$ROOT HF_HOME=$HF_CACHE_DIR HF_HUB_CACHE=$HF_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR=$LOGDIR
cd "$ROOT" || exit 2
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi -L

for C in ${CONFIGS:-qwen25_7b_s1 qwen25_7b_s2}; do
  echo "================= smoke $C  $(date)"
  CFG=$ROOT/training/configs/$C.yaml
  EXTRA=()
  # Stage-2 normally starts from the Stage-1 best directory; for a smoke run start from the
  # adapter's backbone instead so the check does not depend on a finished Stage-1.
  if [[ $C == *_s2* ]]; then
    BASE=$("$PY" -c "import sys; from training.adapters import get_adapter; from training.core.config import load_config; print(get_adapter(load_config(sys.argv[1])['adapter']).base_path)" "$CFG" 2>/dev/null)
    [ -n "$BASE" ] && EXTRA+=("init_from=$BASE")
  fi
  timeout "${SMOKE_TIMEOUT:-3000}" "$PY" -m training.train_stage --config "$CFG" --smoke "${EXTRA[@]}"
  echo "----- $C exit=$?  $(date)"
done
echo "SMOKE TRAIN DONE $(date)"
