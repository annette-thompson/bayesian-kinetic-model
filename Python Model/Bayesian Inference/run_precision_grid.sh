#!/bin/bash
# Full comparison grid on the 3-enzyme system: rtol x atol x precision x floor x device.
# Baseline for the final table is f64 / floor=0 / CPU, so that row must be present.
#
# PYTENSOR_FLAGS is set PER ROW, not globally -- exporting floatX=float32 for the
# whole script leaked float32 into the f64 rows of the previous run.
# JAX_PLATFORMS=cpu is how a row is pinned to CPU; jax otherwise grabs the GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1   # project dir, wherever this repo lives
# nate has miniforge in $HOME; Alpine provides it as a module.
if [ -r "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate Bayesian
else
  module purge && module load miniforge && mamba activate Bayesian
fi
CFG="Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json"
DEV="${1:-gpu}"          # gpu | cpu

# Trimmed from the original 4x3x2x2 sweep. Dropped:
#   rtol 1e-3  -- every combination there returns NaN gradients (kept once, as
#                 the "broken baseline" reference, because the speedup table
#                 needs to show what the old configs were actually costing)
#   f32        -- dies constructing the pytensor<->jax custom_vjp bridge, and
#                 does not fail fast: rows burn the full timeout producing
#                 nothing. Re-add once that bridge failure is understood.
#   atol 1e-6  -- NaN at every rtol tested.
for rtol in 1e-4 1e-5 1e-6; do
for atol in 1e-8 1e-10; do
for floor in 0 0.001; do
  extra=(); [ "$floor" != "0" ] && extra=(--floor "$floor")
  env_pre=(env); [ "$DEV" = "cpu" ] && env_pre=(env JAX_PLATFORMS=cpu)
  label="$DEV/f64 rtol$rtol atol$atol floor$floor"
  echo "######## $label ########"
  timeout 2700 "${env_pre[@]}" python -u Utilities/precision_probe.py --config "$CFG" \
      --utils Utilities --rtol "$rtol" --atol "$atol" --max-steps 20000 --evals 2 \
      --label "$label" "${extra[@]}" 2>&1 | grep -E "^RESULT" | head -1 \
    || echo "RESULT {\"label\": \"$label\", \"ok\": false, \"error\": \"timeout-or-crash\"}"
done; done; done

# The broken baseline, kept for the speedup table's reference row.
for floor in 0 0.001; do
  extra=(); [ "$floor" != "0" ] && extra=(--floor "$floor")
  env_pre=(env); [ "$DEV" = "cpu" ] && env_pre=(env JAX_PLATFORMS=cpu)
  label="$DEV/f64 rtol1e-3 atol1e-6 floor$floor (BROKEN-ref)"
  echo "######## $label ########"
  timeout 2700 "${env_pre[@]}" python -u Utilities/precision_probe.py --config "$CFG" \
      --utils Utilities --rtol 1e-3 --atol 1e-6 --max-steps 20000 --evals 2 \
      --label "$label" "${extra[@]}" 2>&1 | grep -E "^RESULT" | head -1 \
    || echo "RESULT {\"label\": \"$label\", \"ok\": false, \"error\": \"timeout-or-crash\"}"
done
echo "==> grid done ($DEV)"
