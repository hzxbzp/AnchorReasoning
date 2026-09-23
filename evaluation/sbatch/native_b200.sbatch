#!/bin/bash -l
# =============================================================================
# Native-pipeline zero-shot launcher for the released baseline models -- 1 GPU (B200 class).
# Each model runs through its OWN published inference code (prompt, sampling, action head) and
# writes rows in the chains.json schema that the shared rescore step then scores.
#
#   MODEL=<alpamayo_r1_10b|alpamayo_15_10b|autovla_3b|impromptu_7b> [OUT=<dir>] [JUDGE=1] \
#     [MODE=all|real|rescore] sbatch -J zs_<model> evaluation/sbatch/native_b200.sbatch [--subset N ...]
#
#   MODE all (default; extra CLI args are forwarded to every step):
#     1. evaluation/baselines/native/run_<model>.py -> $OUT/chains.json
#     2. evaluation.baselines.rescore              -> $OUT/baseline/metrics_baseline.json
#        (tolerant text extraction, capability detection, LLM judges, official rater-feedback score)
#
# Every baseline needs its OWN python environment, because the released models pin different torch /
# transformers versions. Point these variables at your installs; each falls back to the project
# interpreter:
#     ALPAMAYO_PYTHON / AUTOVLA_PYTHON / IMPROMPTU_PYTHON   interpreters
#     ALPAMAYO_REPO / ALPAMAYO_R1_REPO / AUTOVLA_REPO       released source checkouts
#     ANCHOR_WEIGHTS                                        downloaded model snapshots
# The rescore step always runs in the project environment (CPU only). The weights are copied to
# node-local disk first, because memory-mapping safetensors off a shared parallel filesystem can
# collapse to a few MB/s.
#
# CLUSTER SETTINGS -- everything down to `set -uo pipefail` is scheduler specific. Replace the
# placeholders with the names your site uses, delete the directives it does not need (many sites
# have no QoS or accounting group), size the resource requests for your node, and create the log
# directory before submitting (`mkdir -p logs`).
# =============================================================================
#SBATCH --partition=<gpu-partition>
#SBATCH --qos=<qos>
#SBATCH --account=<slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT=${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}    # repository root (submit from it, or set ANCHOR_ROOT)
PY=${ANCHOR_PYTHON:-python}                       # interpreter of the project environment
WEIGHTS=${ANCHOR_WEIGHTS:-$ROOT/weights/base}
EVAL_DIR=${ANCHOR_EVAL:-$ROOT/eval_results}
LOG_DIR=${ANCHOR_LOGS:-$ROOT/logs}
NATIVE=$ROOT/evaluation/baselines/native
MODEL=${MODEL:?set MODEL=<alpamayo_r1_10b|alpamayo_15_10b|autovla_3b|impromptu_7b>}
MODE=${MODE:-all}
JUDGE=${JUDGE:-1}
RUN=zs_$MODEL
OUT=${OUT:-$EVAL_DIR/$RUN}
cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUT"

# Optional environment-module system: CUDA_MODULE names the toolkit the environments were built
# against. Skipped silently when the site has no `module` command.
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE:-CUDA/12.8.0}" 2>/dev/null
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"         # 10.0 = B200
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}   # compute nodes without internet
export TRITON_CACHE_DIR=$HF_HOME/.triton_cache
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ANCHOR_ROOT="$ROOT" ANCHOR_WEIGHTS="$WEIGHTS"                # the runners read their locations from these

# ---- per-model interpreter + weights + runner ----
case "$MODEL" in
  alpamayo_r1_10b|alpamayo_15_10b)
    MODEL_PY=${ALPAMAYO_PYTHON:-$PY}
    REPO_DIR=$([ "$MODEL" = alpamayo_r1_10b ] && echo models--nvidia--Alpamayo-R1-10B || echo models--nvidia--Alpamayo-1.5-10B)
    WDIR=$(ls -d "$WEIGHTS/$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1); WDIR=${WDIR%/}; KEY=config.json
    RUNNER=("$MODEL_PY" "$NATIVE/run_alpamayo.py" --model "$MODEL") ;;
  autovla_3b)
    MODEL_PY=${AUTOVLA_PYTHON:-$PY}
    WDIR=$(ls -d "$WEIGHTS"/*AutoVLA*/snapshots/*/ 2>/dev/null | head -1); WDIR=${WDIR%/}; KEY=AutoVLA_PDMS_89.ckpt
    RUNNER=("$MODEL_PY" "$NATIVE/run_autovla.py") ;;
  impromptu_7b)
    MODEL_PY=${IMPROMPTU_PYTHON:-$PY}
    WDIR=$WEIGHTS/ImpromptuVLA-7B_AD; KEY=config.json
    RUNNER=("$MODEL_PY" "$NATIVE/run_impromptu.py") ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} MODE=$MODE MODEL=$MODEL OUT=$OUT JUDGE=$JUDGE python=$MODEL_PY ====="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
"$MODEL_PY" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'|',torch.cuda.get_device_name(0))"

# ---- stage the weights on node-local disk (set LOCAL_SCRATCH to your site's local disk) ----
STAGE=${LOCAL_SCRATCH:-/lscratch}/anchor_${SLURM_JOB_ID:-$$}   # own subdirectory: it is deleted on exit
if [ -d "$WDIR" ] && mkdir -p "$STAGE" 2>/dev/null; then
  trap 'rm -rf "$STAGE"' EXIT
  DST=$STAGE/$(basename "$WDIR"); t0=$(date +%s)
  # -L: a hub snapshot directory is a farm of relative symlinks into ../../blobs -> dereference
  if cp -rL "$WDIR" "$DST" && [ -f "$DST/$KEY" ]; then
    export ANCHOR_BASE_PATH=$DST
    echo "[stage] $WDIR -> $DST ($(du -sh "$DST" | cut -f1), $(( $(date +%s) - t0 ))s)"
  else
    echo "[stage] copy failed or incomplete, loading from the shared filesystem"; rm -rf "$DST"
  fi
else
  echo "[stage] no node-local scratch; loading from the shared filesystem"
fi

J=(); [ "$JUDGE" = "1" ] && J+=(--judge)
run_real()    { "${RUNNER[@]}" --out "$OUT/chains.json" "$@"; }
run_rescore() { CUDA_VISIBLE_DEVICES="" "$PY" -m evaluation.baselines.rescore --chains "$OUT/chains.json" \
                  --profile "$MODEL" --out "$OUT/baseline" ${J[@]+"${J[@]}"} "$@"; }

RC=0
case "$MODE" in
  real)    run_real "$@"; RC=$? ;;
  rescore) run_rescore "$@"; RC=$? ;;
  all)     run_real "$@"; RC=$?
           if [ -f "$OUT/chains.json" ]; then run_rescore "$@"; r=$?; [ $r -ne 0 ] && RC=$(( RC > r ? RC : r )); \
           else echo "[all] no chains.json -> rescore skipped"; fi ;;
  *)       echo "unknown MODE=$MODE"; RC=2 ;;
esac
echo "===== native eval exited rc=$RC $(date) -> $OUT ====="
exit $RC
