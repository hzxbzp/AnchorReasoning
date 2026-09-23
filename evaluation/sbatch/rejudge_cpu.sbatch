#!/bin/bash -l
# =============================================================================
# Re-score the saved generations of a finished evaluation run with the LLM judges -- CPU only, no
# GPU: it reads the chains written by evaluation.eval_dev / evaluation.eval_val456 and refreshes the
# judge-based columns of the metrics file.
#
#   RUN=<eval output dir> [STAGE=s1|s2] [JUDGE=1] \
#     sbatch -J rejudge_<name> evaluation/sbatch/rejudge_cpu.sbatch [extra args]
#
#   The judges read their API key from the file named by OPENAI_API_KEY_FILE (see training/core/paths.py).
#
# CLUSTER SETTINGS -- everything down to `set -uo pipefail` is scheduler specific: replace the
# placeholders, drop what your site does not use, and create the log directory before submitting.
# =============================================================================
#SBATCH --partition=<cpu-partition>
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT=${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}    # repository root (submit from it, or set ANCHOR_ROOT)
PY=${ANCHOR_PYTHON:-python}                       # interpreter of the project environment
RUN=${RUN:?set RUN=<evaluation output dir>}; STAGE=${STAGE:-s2}; JUDGE=${JUDGE:-1}
case "$RUN" in /*) ;; *) RUN=$ROOT/$RUN ;; esac
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
A=(--run-dir "$RUN" --stage "$STAGE"); [ "$JUDGE" = 1 ] && A+=(--judge)
echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} rejudge ${A[*]} $* ====="
"$PY" -m evaluation.rejudge "${A[@]}" "$@"; RC=$?
echo "===== rejudge exited rc=$RC $(date) ====="; exit $RC
