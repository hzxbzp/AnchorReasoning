#!/bin/bash -l
# =============================================================================
# Evaluation launcher for a checkpoint produced by training/ -- 1 GPU (B200 class), with a CUDA
# readiness gate for nodes that have just rebooted.
#
#   CKPT=<save_pretrained dir | base> [ADAPTER=<name>] [OUT=<dir>] [MODE=val456|dev|ci|all] \
#     sbatch -J <name> evaluation/sbatch/eval_b200.sbatch [extra args for the python entry point]
#
#   e.g. CKPT=runs/<run-name>/best/best_stage_s2 sbatch -J eval_s2 evaluation/sbatch/eval_b200.sbatch
#
#   MODE (default val456):
#     val456 -> evaluation.eval_val456  rated validation frames: Stage-2 chain generation, the
#               official rater-feedback score, the reference rows and the sanity gates
#     dev    -> evaluation.eval_dev     dev-set proxy metrics (STAGE=s1|s2, FRAMES=<json>|rfs)
#     ci     -> evaluation.ci_eval      chain causal-intervention experiments; re-uses OUT/chains.json
#               when val456 ran first in the same output dir
#     all    -> val456, then ci on its chains
#   ADAPTER defaults to <CKPT>/adapter_name.json; OUT defaults to <eval dir>/<run name>.
#   NOHASH=1 downgrades a prompt_hash.json mismatch from an error to a warning.
#   After a node reboot CUDA initialisation can fail for minutes while nvidia-smi already looks
#   healthy, which would crash the job on start-up; the gate below probes CUDA for up to 15 minutes
#   and requeues the job with a 30 minute delay instead.
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
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --requeue
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
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"         # 10.0 = B200
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

echo "===== $(hostname) $(date) job=${SLURM_JOB_ID:-manual} restarts=${SLURM_RESTART_COUNT:-0} MODE=$MODE CKPT=$CKPT ADAPTER=${ADAPTER:-auto} OUT=$OUT ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# CUDA readiness gate: probe from the batch shell (same cgroup and GPUs as the job) and only launch
# once CUDA really works; otherwise requeue with a 30 minute delay.
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
  *)      echo "unknown MODE=$MODE"; RC=2 ;;
esac
echo "===== eval exited rc=$RC $(date) -> $OUT ====="
exit $RC
