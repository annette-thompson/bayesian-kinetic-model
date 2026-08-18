#!/bin/bash
# 32-bit vs 64-bit, GPU vs CPU, at ONE tolerance so the four cells are comparable.
#
# Defaults to rtol 1e-4 / atol 1e-8. Every cell must be run at the same tolerance
# or the precision comparison is confounded by the tolerance, which is the single
# biggest lever on cost in this model.
#
# 32-bit comes from Utilities32 (Sync/make_utils32.sh), which STRIPS the explicit
# float64 rather than substituting float32 -- jax_enable_x64=False and pytensor's
# floatX default then agree with each other. No PYTENSOR_FLAGS is set here on
# purpose; specifying float32 on one side only is what broke the earlier attempt.
#
# Usage: "Bayesian Inference/run_32v64_matrix.sh" [rtol] [atol]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if [ -r "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate Bayesian
else
  module purge && module load miniforge && mamba activate Bayesian
fi

RTOL="${1:-1e-4}"; ATOL="${2:-1e-8}"
CFG="Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json"
HOST="$(hostname)"
[ -d Utilities32 ] || bash Sync/make_utils32.sh >/dev/null 2>&1

run() {  # utils device label
  echo ""
  echo "######## $HOST $3 rtol=$RTOL atol=$ATOL ########"
  local pre=(env)
  [ "$2" = "cpu" ] && pre=(env JAX_PLATFORMS=cpu)
  timeout 7200 "${pre[@]}" python -u Utilities/precision_probe.py --config "$CFG" \
    --utils "$1" --rtol "$RTOL" --atol "$ATOL" --max-steps 20000 --evals 2 \
    --label "$HOST $3 rtol$RTOL atol$ATOL"
}

run Utilities   gpu "f64-gpu"
run Utilities32 gpu "f32-gpu"
run Utilities   cpu "f64-cpu"
run Utilities32 cpu "f32-cpu"
echo ""
echo "==> 32v64 matrix done"
