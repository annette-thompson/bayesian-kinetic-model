#!/bin/bash -l
#SBATCH --job-name=bayes_seg_gpu
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --account=ucb634_asc2

# `-l` (login shell) and `--export=NONE` below are both load-bearing, and this
# script silently failed without them. sbatch defaults to --export=ALL, so the
# job inherited the submitting shell's MODULEPATH -- which on the login node does
# NOT contain /curc/sw/alpine-modules/* -- and that overrode what the compute node
# would have set for itself, making `module load miniforge` report "unknown
# module". --export=NONE lets the node build its own environment; -l makes it
# actually run the profile that initialises Lmod.


_die() { echo "Error: $*" >&2; exit 1; }

# SLURM's [D-]HH:MM:SS time-limit string -> decimal hours.
_time_str_to_hours() {
  awk -F'[-:]' '{
    n = NF
    if (n == 4)      { d=$1; h=$2; m=$3; s=$4 }
    else if (n == 3) { d=0;  h=$1; m=$2; s=$3 }
    else if (n == 2) { d=0;  h=0;  m=$1; s=$2 }
    else             { d=0;  h=0;  m=0;  s=$1 }
    printf "%.4f", d*24 + h + m/60 + s/3600
  }' <<< "$1"
}

# This job's SLURM --time limit, in hours (falls back to this script's
# #SBATCH --time header if squeue is unavailable or the limit is unlimited).
_job_time_limit_hours() {
  local raw=""
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue >/dev/null 2>&1; then
    raw="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | head -n1)"
  fi
  if [[ -z "$raw" || "$raw" == "UNLIMITED" ]]; then
    raw="$(grep -m1 '^#SBATCH --time=' "$0" | cut -d= -f2)"
  fi
  [[ -n "$raw" ]] || { echo "24"; return; }
  _time_str_to_hours "$raw"
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/solver_params.json|yaml [MAX_HOURS] [EXTRA_DRAWS]" >&2
  echo "  MAX_HOURS   wall-clock budget before checkpoint+exit; omit (or pass \"\") to" >&2
  echo "              default to 95% of this job's SLURM --time limit" >&2
  echo "  EXTRA_DRAWS optional: raise the draw target (pass on the FIRST segment only)" >&2
  exit 1
fi

SOLVER_PARAMS_FILE="$1"
MAX_HOURS="${2:-}"
EXTRA_DRAWS="${3:-}"
[[ -r "$SOLVER_PARAMS_FILE" ]] || _die "Solver params file not readable: $SOLVER_PARAMS_FILE"

if [[ -z "$MAX_HOURS" ]]; then
  TIME_LIMIT_HOURS="$(_job_time_limit_hours)"
  MAX_HOURS="$(awk -v t="$TIME_LIMIT_HOURS" 'BEGIN{ if (t<=0.25) printf "%.4f", t; else printf "%.4f", t-0.25 }')"
  echo "==> MAX_HOURS not provided; defaulting to SLURM time limit (${TIME_LIMIT_HOURS}h) minus 15 min = ${MAX_HOURS}h"
fi

EXTRA_ARGS=()
if [[ -n "$EXTRA_DRAWS" && "$EXTRA_DRAWS" != "0" ]]; then
  EXTRA_ARGS+=(--extra_draws "$EXTRA_DRAWS")
fi

PROJECT_DIR="/projects/anth4580/Bayesian"

module purge
module load miniforge
mamba activate Bayesian

echo "----------------------------------------------------------"
echo "==> Resumable BlackJAX inference segment (GPU)"
echo "==> Solver params: $SOLVER_PARAMS_FILE"
echo "==> Max hours (this segment): $MAX_HOURS"
echo "==> Extra draws: ${EXTRA_DRAWS:-none}"
echo "==> SLURM job: ${SLURM_JOB_ID:-unset}  time limit: ${SBATCH_TIMELIMIT:-see --time}"
echo "==> SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "==> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> Python executable: $(command -v python)"
echo "==> Host: $(hostname)"
echo "==> Thread env: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"

# Pre-stamp arviz's "warn once per day" cache file before it gets imported.
# (Avoids a startup race when many jobs launch at once.)
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
  --max_hours "$MAX_HOURS" \
  "${EXTRA_ARGS[@]}"
