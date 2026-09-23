#!/bin/bash -l
# =============================================================================
# Evaluation launcher for a checkpoint produced by training/ -- 1 GPU (H100 class).
#
#   CKPT=<save_pretrained dir | base> [ADAPTER=<name>] [OUT=<dir>] [MODE=val456|dev|ci|all|s1full] \
#     sbatch -J <name> evaluation/sbatch/eval_h100.sbatch [extra args for the python entry point]
#
#   e.g. CKPT=runs/<run-name>/best/best_stage_s2 sbatch -J eval_s2 evaluation/sbatch/eval_h100.sbatch
#        CKPT=runs/<run-name>/best/best_stage_s1 MODE=dev STAGE=s1 FRAMES=data/dev_eval_frames.json \
#          sbatch -J dev_s1 evaluation/sbatch/eval_h100.sbatch
#
#   MODE (default val456):
#     val456 -> evaluation.eval_val456  rated validation frames: Stage-2 chain generation, the
#               official rater-feedback score, the reference rows and the sanity gates
#     dev    -> evaluation.eval_dev     dev-set proxy metrics (STAGE=s1|s2, FRAMES=<json>|rfs)
#     ci     -> evaluation.ci_eval      chain causal-intervention experiments; re-uses OUT/chains.json
#               when val456 ran first in the same output dir
#     all    -> val456, then ci on its chains
#     s1full -> eval_dev --stage s1 --frames rfs with the LLM judge and the per-cluster breakdown
#   ADAPTER defaults to <CKPT>/adapter_name.json; OUT defaults to <eval dir>/<run name>.
#   The prompt_hash.json stored with the checkpoint is verified (mismatch -> exit 2); NOHASH=1
#   downgrades the mismatch to a warning.
#
# CLUSTER SETTINGS -- everything down to `set -uo pipefail` is scheduler specific. Replace the
# placeholders with the names your site uses, delete the directives it does not need (many sites
# have no QoS or accounting group), size the resource requests for your node, and create the log
# directory before submitting (`mkdir -p logs`).
# =============================================================================
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT=${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}    # repository root (submit from it, or set ANCHOR_ROOT)
PY=${ANCHOR_PYTHON:-python}                       # interpreter of the project environment
EVAL_DIR=${ANCHOR_EVAL:-$ROOT/eval_results}
LOG_DIR=${ANCHOR_LOGS:-$ROOT/logs}
CKPT=${CKPT:?set CKPT=<checkpoint dir> (or CKPT=base)}
MODE=${MODE:-val456}
ADAPTER=${ADAPTER:-}
OUT=${OUT:-}
FRAMES=${FRAMES:-}
STAGE=${STAGE:-s2}
NOHASH=${NOHASH:-0}
[ "$CKPT" != "base" ] && case "$CKPT" in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
cd "$ROOT"
mkdir -p "$LOG_DIR" "$EVAL_DIR"

# Optional environment-module system: CUDA_MODULE names the toolkit the project environment was
# built against. Skipped silently when the site has no `module` command.
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE:-CUDA/12.8.0}" 2>/dev/null
command -v nvcc >/dev/null 2>&1 && export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"          # 9.0 = H100
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}   # compute nodes without internet
export TRITON_CACHE_DIR=$HF_HOME/.triton_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

# run name / output dir (same derivation as eval_val456.run_name_of; no torch import)
if [ -z "$OUT" ]; then
  RUN=$("$PY" - "$CKPT" <<'PYEOF'
import os, sys
ckpt = None if sys.argv[1] == "base" else sys.argv[1]
if not ckpt: print("base"); sys.exit()
parts = [p for p in os.path.abspath(ckpt).split(os.sep) if p]
tail = parts[-3:] if len(parts) >= 3 and parts[-2] == "best" else parts[-2:]
print("_".join(tail).replace("checkpoint-", "ckpt"))
PYEOF
)
  OUT=$EVAL_DIR/$RUN
fi
mkdir -p "$OUT"
COMMON=()
[ -n "$ADAPTER" ] && COMMON+=(--adapter "$ADAPTER")
[ "$NOHASH" = "1" ] && COMMON+=(--no-hash-check)

echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} MODE=$MODE CKPT=$CKPT ADAPTER=${ADAPTER:-auto} OUT=$OUT ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_val456() { "$PY" -m evaluation.eval_val456 --ckpt "$CKPT" --out-dir "$OUT" "${COMMON[@]}" "$@"; }
run_dev()    { "$PY" -m evaluation.eval_dev --ckpt "$CKPT" --stage "$STAGE" ${FRAMES:+--frames "$FRAMES"} \
                 --out "$OUT/dev_${STAGE}_metrics.json" "${COMMON[@]}" "$@"; }
run_ci()     { "$PY" -m evaluation.ci_eval --ckpt "$CKPT" --out-dir "$OUT/ci" \
                 $([ -f "$OUT/chains.json" ] && echo --base-dir "$OUT") "${COMMON[@]}" "$@"; }

RC=0
case "$MODE" in
  val456) run_val456 "$@"; RC=$? ;;
  dev)    run_dev "$@";    RC=$? ;;
  ci)     run_ci "$@";     RC=$? ;;
  all)    run_val456 "$@" && run_ci; RC=$? ;;
  s1full) STAGE=s1; FRAMES=${FRAMES:-rfs}
          run_dev --judge --by-cluster "$@"; RC=$? ;;
  *)      echo "unknown MODE=$MODE"; RC=2 ;;
esac
echo "===== eval exited rc=$RC $(date) -> $OUT ====="
exit $RC
