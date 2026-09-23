#!/bin/bash -l
# =============================================================================
# Re-score a finished zero-shot run -- CPU only: tolerant text extraction, capability profile and
# the LLM judges, starting from the chains.json the generation job wrote. Use it to re-run the
# scoring side without regenerating anything on a GPU.
#
#   RUN=zs_<model> PROFILE=<adapter / model name> [JUDGE=1] [REUSE_JUDGES=1] \
#     sbatch -J rescore_<name> evaluation/sbatch/rescore_cpu.sbatch [extra args]
#
#   REUSE_JUDGES=1 re-uses the judge verdicts stored in the previous metrics_baseline.json instead of
#   calling the judge again (the previous file is kept as metrics_baseline.prev.json).
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
EVAL_DIR=${ANCHOR_EVAL:-$ROOT/eval_results}
RUN=${RUN:?set RUN=zs_<model>}
PROFILE=${PROFILE:?set PROFILE=<adapter / model name>}
JUDGE=${JUDGE:-1}
REUSE_JUDGES=${REUSE_JUDGES:-0}
OUT=${OUT:-$EVAL_DIR/$RUN}
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} rescore RUN=$RUN PROFILE=$PROFILE REUSE_JUDGES=$REUSE_JUDGES args=$* ====="
A=(--chains "$OUT/chains.json" --profile "$PROFILE" --out "$OUT/baseline")
[ "$JUDGE" = "1" ] && A+=(--judge)
if [ "$REUSE_JUDGES" = "1" ] && [ -f "$OUT/baseline/metrics_baseline.json" ]; then
  cp "$OUT/baseline/metrics_baseline.json" "$OUT/baseline/metrics_baseline.prev.json"
  A+=(--reuse-judges "$OUT/baseline/metrics_baseline.prev.json")
fi
"$PY" -m evaluation.baselines.rescore "${A[@]}" "$@"; RC=$?
echo "===== rescore exited rc=$RC $(date) ====="
exit $RC
