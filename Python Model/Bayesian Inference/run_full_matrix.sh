#!/bin/bash
# Complete cell matrix at ONE tolerance: {f64,f32} x {gpu,cpu} x floors.
# Every cell at the same tolerance so precision/device/floor are the only variables.
#
# Usage: run_full_matrix.sh [rtol] [atol] [python]
#   python defaults to whatever `python` resolves to; pass an explicit interpreter
#   to pin a specific conda env (Alpine has several with different jax versions).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RTOL="${1:-1e-4}"; ATOL="${2:-1e-8}"; PY="${3:-python}"; ONLY="${4:-both}"  # gpu|cpu|both

# Activate an env only when no explicit interpreter was given. Passing a python
# path (Alpine, where several envs with different jax versions coexist) must win
# over whatever conda would activate.
if [ "$PY" = "python" ]; then
  if [ -r "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate Bayesian
  elif [ -r /curc/sw/install/miniforge3/24.11.3-0/etc/profile.d/conda.sh ]; then
    source /curc/sw/install/miniforge3/24.11.3-0/etc/profile.d/conda.sh; conda activate Bayesian
  fi
fi
command -v "$PY" >/dev/null || { echo "FATAL: no interpreter '$PY' on PATH"; exit 1; }
CFG="Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json"
HOST="$(hostname -s)"
[ -d Utilities32 ] || bash Sync/make_utils32.sh >/dev/null 2>&1

"$PY" -c "import jax,importlib.metadata as md;print('STACK jax',md.version('jax'),'diffrax',md.version('diffrax'),'pymc',md.version('pymc'),jax.devices())" 2>&1 | tail -1

cell() {  # utils device floor label
  echo ""; echo "######## $HOST $4 ########"
  local pre=(env); [ "$2" = "cpu" ] && pre=(env JAX_PLATFORMS=cpu)
  local extra=(); [ "$3" != "0" ] && extra=(--floor "$3")
  timeout 7200 "${pre[@]}" "$PY" -u Utilities/precision_probe.py --config "$CFG" \
    --utils "$1" --rtol "$RTOL" --atol "$ATOL" --max-steps 20000 --evals 2 \
    --label "$HOST $4" "${extra[@]}" 2>&1 | grep -E "^RESULT|FloatingPointError|TypeError|Error:" | head -2
}

for dev in gpu cpu; do
  [ "$ONLY" = both ] || [ "$ONLY" = "$dev" ] || continue
  for floor in 0 0.00001 0.0001 0.001 0.01; do
    cell Utilities "$dev" "$floor" "f64-$dev floor=$floor"
  done
  cell Utilities32 "$dev" 0 "f32-$dev floor=0"
done
echo ""; echo "==> full matrix done ($RTOL/$ATOL)"
