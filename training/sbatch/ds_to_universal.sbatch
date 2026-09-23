#!/bin/bash -l
# =============================================================================
# Convert a DeepSpeed ZeRO-1/2 checkpoint saved with N GPUs into a "universal" checkpoint that can
# be resumed with any GPU count (DeepSpeed refuses to load per-rank ZeRO shards across a data-parallel
# world-size change: engine._load_zero_checkpoint raises ZeRORuntimeException). CPU only.
#
#   CKPT=runs/<run>/checkpoint-<N> sbatch -J univ_<run> training/sbatch/ds_to_universal.sbatch
#   Submit from the repository root, or export ANCHOR_ROOT=<repo root> beforehand.
#
# Output: <CKPT>/global_step<N>_universal/{zero/..., mp_rank_00_model_states.pt} and
#         <CKPT>/latest_universal  (= "global_step<N>_universal")
# Resume with configs/deepspeed/zero2_universal.json ("checkpoint": {"load_universal": true});
# the training launchers pick that up automatically. The per-rank shards (<CKPT>/global_step<N>)
# are left untouched.
#
# -----------------------------------------------------------------------------
# CLUSTER SETTINGS -- EDIT THIS BLOCK FOR YOUR SITE
#   * #SBATCH lines: account, CPU-only partition, CPUs, memory and walltime.
#   * --output/--error are relative to the directory you submit from; create it first
#     (`mkdir -p logs`) or point them somewhere else.
#   * Mail options are left out on purpose; add e.g. `#SBATCH --mail-type=END,FAIL` if wanted.
#   * PYTHON_BIN below: the interpreter of the environment that has DeepSpeed installed.
# -----------------------------------------------------------------------------
#SBATCH --account=<slurm-account>
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
ROOT="${ANCHOR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
PYTHON_BIN="${PYTHON_BIN:-python}"                 # interpreter of the environment with DeepSpeed
# =============================================================================

PY="$PYTHON_BIN"
CKPT=${CKPT:?set CKPT=<checkpoint-N dir>}
case "$CKPT" in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
export OMP_NUM_THREADS=4 PYTHONPATH=$ROOT ANCHOR_ROOT=$ROOT
cd "$ROOT" || exit 2
DS=$("$PY" -c "import os, deepspeed; print(os.path.dirname(deepspeed.__file__))") || { echo "DeepSpeed not importable with $PY"; exit 2; }
TAG=$(cat "$CKPT/latest")                       # e.g. global_step4200
IN="$CKPT/$TAG"; OUT="$CKPT/${TAG}_universal"

echo "===== $(hostname) job=$SLURM_JOB_ID $(date)"
echo "  in : $IN  ($(du -sh "$IN" | cut -f1), $(ls "$IN" | grep -c optim_states) optimizer shards)"
echo "  out: $OUT"
[ -d "$IN" ] || { echo "no $IN"; exit 2; }
if [ -f "$CKPT/latest_universal" ] && [ -f "$OUT/zero/optimizer_state.pt" ]; then
  echo "already converted ($(cat "$CKPT/latest_universal")) -> nothing to do"; exit 0
fi
rm -rf "$OUT"

# --inject_missing_state: checkpoints written through the HF Trainer carry no
# universal_checkpoint_info (only Megatron-style clients write it); the tool then fills in the
# version key itself.
"$PY" "$DS/checkpoint/ds_to_universal.py" --input_folder "$IN" --output_folder "$OUT" \
      --inject_missing_state --num_extract_workers 2 --num_merge_workers 4
RC=$?
echo "--- converter rc=$RC $(date)"
[ "$RC" -eq 0 ] || { rm -rf "$OUT"; echo "CONVERSION FAILED"; exit 1; }

echo "===== verify"
"$PY" - "$CKPT" "$OUT" <<'PYEOF'
import os, sys, json, torch
ckpt, out = sys.argv[1], sys.argv[2]
lu = open(os.path.join(ckpt, "latest_universal")).read().strip()
print("  latest_universal ->", lu, "| matches:", lu == os.path.basename(out))
zero = os.path.join(out, "zero")
params = [d for d in os.listdir(zero) if os.path.isdir(os.path.join(zero, d))]
files = {f for d in params for f in os.listdir(os.path.join(zero, d))}
print(f"  zero/: {len(params)} parameter folders, per-param files {sorted(files)}")
mp = torch.load(os.path.join(out, "mp_rank_00_model_states.pt"), map_location="cpu", mmap=True, weights_only=False)
print("  mp_rank_00_model_states.pt copied:", "module" in mp, "| ds_version", mp.get("ds_version"), "| dp_world_size", mp.get("dp_world_size"))
n_expected = sum(len(g) for g in mp["param_shapes"])
print(f"  expected params from param_shapes: {n_expected} | present: {len(params)} | ok: {len(params) == n_expected}")
osd = torch.load(os.path.join(zero, "optimizer_state.pt"), map_location="cpu", weights_only=False)
print("  optimizer_state.pt keys:", sorted(osd.keys()))
ok = lu == os.path.basename(out) and len(params) == n_expected and {"fp32.pt", "exp_avg.pt", "exp_avg_sq.pt"} <= files
print("UNIVERSAL OK" if ok else "UNIVERSAL INCOMPLETE")
sys.exit(0 if ok else 1)
PYEOF
VRC=$?
du -sh "$OUT" 2>/dev/null
echo "===== DONE rc=$VRC $(date)"
exit $VRC
