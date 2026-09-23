#!/bin/bash -l
# =============================================================================
# Zero-shot launcher for a released backbone -- 1 GPU (L40S class). The model is loaded through its
# adapter and prompted with this repository's own Stage-2 prompt, so the output is directly
# comparable with a fine-tuned checkpoint. Checkpoints trained by training/ use eval_h100.sbatch or
# eval_b200.sbatch instead.
#
#   ADAPTER=<name> [OUT=<dir>] [JUDGE=1] [MODE=all|val456|rescore] \
#     sbatch -J zs_<name> evaluation/sbatch/eval_l40s.sbatch [extra args, e.g. --subset 5]
#
#   MODE all (default; extra CLI args are forwarded to every step):
#     1. evaluation.eval_val456      --ckpt base: Stage-2 chain generation on the rated validation
#                                    frames, official rater-feedback score, reference rows
#     2. evaluation.baselines.rescore tolerant text extraction + capability profile + LLM judges (CPU)
#   OUT defaults to <eval dir>/zs_<ADAPTER> (not the shared 'base' directory).
#   The base weights are copied to node-local disk first: memory-mapping safetensors off a shared
#   parallel filesystem can collapse to a few MB/s (see training/adapters/base.py, _staged_or).
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
EVAL_DIR=${ANCHOR_EVAL:-$ROOT/eval_results}
LOG_DIR=${ANCHOR_LOGS:-$ROOT/logs}
ADAPTER=${ADAPTER:?set ADAPTER=<adapter name, see training.adapters.REGISTRY>}
MODE=${MODE:-all}
JUDGE=${JUDGE:-1}
RUN=zs_$ADAPTER
OUT=${OUT:-$EVAL_DIR/$RUN}
cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUT"

# Optional environment-module system: CUDA_MODULE names the toolkit the project environment was
# built against. Skipped silently when the site has no `module` command.
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE:-CUDA/12.8.0}" 2>/dev/null
command -v nvcc >/dev/null 2>&1 && export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"          # 8.9 = L40S (Ada)
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}   # compute nodes without internet
export TRITON_CACHE_DIR=$HF_HOME/.triton_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} MODE=$MODE ADAPTER=$ADAPTER OUT=$OUT JUDGE=$JUDGE ====="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

# ---- stage the base weights on node-local disk (set LOCAL_SCRATCH to your site's local disk) ----
STAGE=${LOCAL_SCRATCH:-/lscratch}/anchor_${SLURM_JOB_ID:-$$}   # own subdirectory: it is deleted on exit
BASE=$("$PY" - "$ADAPTER" <<'PYEOF'
import sys
from training.adapters import get_adapter
print(get_adapter(sys.argv[1]).base_path)
PYEOF
)
if [ -d "$BASE" ] && mkdir -p "$STAGE" 2>/dev/null; then
  trap 'rm -rf "$STAGE"' EXIT                # installed BEFORE the copy: a killed copy leaves nothing behind
  DST=$STAGE/$(basename "$BASE")
  t0=$(date +%s)
  # -L: a hub snapshot directory is a farm of relative symlinks into ../../blobs -> dereference,
  #     or the copy is dangling
  if cp -rL "$BASE" "$DST" && [ -f "$DST/config.json" ]; then
    export ANCHOR_BASE_PATH=$DST
    echo "[stage] $BASE -> $DST ($(du -sh "$DST" | cut -f1), $(( $(date +%s) - t0 ))s)"
  else
    echo "[stage] copy failed or incomplete, loading from the shared filesystem"; rm -rf "$DST"
  fi
else
  echo "[stage] no node-local scratch; loading from the shared filesystem"
fi

# The judges run ONLY in the rescore step, which scores the tolerant extraction; judging the strict
# parse inside eval_val456 as well would pay for every recovered item twice.
J=(); [ "$JUDGE" = "1" ] && J+=(--judge)
run_val456()  { "$PY" -m evaluation.eval_val456 --ckpt base --adapter "$ADAPTER" --run "$RUN" --out-dir "$OUT" "$@"; }
run_rescore() { "$PY" -m evaluation.baselines.rescore --chains "$OUT/chains.json" --profile "$ADAPTER" \
                  --out "$OUT/baseline" ${J[@]+"${J[@]}"} "$@"; }

RC=0
case "$MODE" in
  val456)  run_val456 "$@"; RC=$? ;;
  rescore) run_rescore "$@"; RC=$? ;;
  all)     # rescore is CPU-only and needs nothing but chains.json, so it still runs when the
           # generation pass reported a non-zero status; the worst return code is reported.
           run_val456 "$@"; RC=$?
           if [ -f "$OUT/chains.json" ]; then run_rescore "$@"; r=$?; [ $r -ne 0 ] && RC=$(( RC > r ? RC : r )); \
           else echo "[all] no chains.json -> rescore skipped"; fi ;;
  *)       echo "unknown MODE=$MODE"; RC=2 ;;
esac
echo "===== eval exited rc=$RC $(date) -> $OUT ====="
exit $RC
