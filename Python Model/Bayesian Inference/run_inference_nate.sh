#!/bin/bash
# One BlackJAX inference run on nate -- a local lab Ubuntu workstation (Xeon
# Silver 4208 + RTX 3080) reached via `ssh nate`. There's no SLURM here, so
# there's no wall-clock kill to checkpoint around: MAX_HOURS is optional and
# omitting it runs straight through to the draw target. inference_runner.py's
# checkpointing still applies, so an interrupted run (ssh drop, reboot,
# Ctrl-C) resumes from <results_save_dir>/checkpoint/ if you just re-run this
# script with the same solver_params file.
#
# Normally launched via submit_inference_nate.sh, which runs this in the
# background over ssh (nohup, redirected stdio) so it survives the ssh
# session dropping. Can also be run directly on nate for a foreground test.
set -euo pipefail

_die() { echo "Error: $*" >&2; exit 1; }

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/solver_params.json|yaml [MAX_HOURS] [EXTRA_DRAWS]" >&2
  echo "  MAX_HOURS   optional wall-clock budget before checkpoint+exit; omit (or pass \"\")" >&2
  echo "              to run to completion (no wall-clock kill on nate)" >&2
  echo "  EXTRA_DRAWS optional: raise the draw target" >&2
  exit 1
fi

SOLVER_PARAMS_FILE="$1"
MAX_HOURS="${2:-}"
EXTRA_DRAWS="${3:-}"
[[ -r "$SOLVER_PARAMS_FILE" ]] || _die "Solver params file not readable: $SOLVER_PARAMS_FILE"

EXTRA_ARGS=()
[[ -n "$MAX_HOURS" ]] && EXTRA_ARGS+=(--max_hours "$MAX_HOURS")
if [[ -n "$EXTRA_DRAWS" && "$EXTRA_DRAWS" != "0" ]]; then
  EXTRA_ARGS+=(--extra_draws "$EXTRA_DRAWS")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ssh-launched non-login shells often don't source the rc file that conda's
# installer wrote its init hook into, so PATH may not have conda/mamba on it
# yet. Try common install locations before giving up.
if ! command -v conda >/dev/null 2>&1 && ! command -v mamba >/dev/null 2>&1; then
  for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/miniforge3"; do
    if [[ -x "$base/bin/conda" ]]; then
      export PATH="$base/bin:$PATH"
      break
    fi
  done
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate Bayesian
elif command -v mamba >/dev/null 2>&1; then
  source "$(mamba info --base)/etc/profile.d/conda.sh"
  conda activate Bayesian
else
  _die "Neither conda nor mamba found on PATH -- set up the 'Bayesian' env on nate first"
fi

echo "----------------------------------------------------------"
echo "==> Resumable BlackJAX inference run (nate, local GPU)"
echo "==> Solver params: $SOLVER_PARAMS_FILE"
echo "==> Max hours: ${MAX_HOURS:-unbounded (run to completion)}"
echo "==> Extra draws: ${EXTRA_DRAWS:-none}"
echo "==> Host: $(hostname)"
echo "==> PID: $$"
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
CPU_COUNT="$(nproc 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="$CPU_COUNT"
export OPENBLAS_NUM_THREADS="$CPU_COUNT"
export MKL_NUM_THREADS="$CPU_COUNT"
export NUMEXPR_NUM_THREADS="$CPU_COUNT"

echo "==> Python executable: $(command -v python)"
echo "==> Thread env: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# Pre-stamp arviz's "warn once per day" cache file before it gets imported.
python - <<'PY'
import datetime
from pathlib import Path
from platformdirs import user_cache_dir

cache_dir = Path(user_cache_dir("arviz", "arviz"))
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "daily_warning").write_text(datetime.date.today().isoformat())
PY

python - <<'PY'
import importlib.metadata as md
import platform

packages = ["arviz", "blackjax", "diffrax", "equinox", "jax", "jaxlib", "numpy", "pymc", "pytensor", "zarr"]
print(f"==> Python version: {platform.python_version()}")
print(f"==> Platform: {platform.platform()}")
for package in packages:
  try:
    print(f"==> {package}: {md.version(package)}")
  except md.PackageNotFoundError:
    print(f"==> {package}: not installed")

import jax
print(f"==> JAX devices: {jax.devices()}")
PY

time python -u "$PROJECT_DIR/Utilities/inference_runner.py" \
  --solver_params_file "$SOLVER_PARAMS_FILE" \
  "${EXTRA_ARGS[@]}"
