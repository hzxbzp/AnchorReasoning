#!/bin/bash -l
# =============================================================================
# Smoke test: load every backbone's BASE weights through its adapter and generate two rated
# validation frames with the Stage-2 prompt on one GPU. Set ADAPTERS=<space separated names> to
# restrict the list (names come from training.adapters.REGISTRY).
#
# CLUSTER SETTINGS -- everything down to `set -uo pipefail` is scheduler specific: replace the
# placeholders, drop what your site does not use, and create the log directory before submitting.
# =============================================================================
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=smoke_adapters
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT=${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}    # repository root (submit from it, or set ANCHOR_ROOT)
PY=${ANCHOR_PYTHON:-python}                       # interpreter of the project environment
EVAL_DIR=${ANCHOR_EVAL:-$ROOT/eval_results}
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
OUT=$EVAL_DIR/smoke_adapters; mkdir -p "$OUT"
nvidia-smi -L
for A in ${ADAPTERS:-qwen25_7b qwen3vl_8b impromptu_7b autovla_3b cosmos_2b alpamayo_r1_10b alpamayo_15_10b cosmos_8b}; do
  echo "================= $A  $(date)"
  timeout 2400 "$PY" -m evaluation.eval_dev --ckpt base --adapter "$A" --stage s2 --frames rfs --subset 2 \
    --max-new-tokens 512 --out "$OUT/$A.json" --no-hash-check
  echo "----- $A exit=$?  $(date)"
done
echo "SMOKE DONE $(date)"
