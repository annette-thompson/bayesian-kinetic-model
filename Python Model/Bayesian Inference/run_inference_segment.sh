#!/bin/bash
#SBATCH --job-name=bayes_seg
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --account=ucb634_asc2

# One resumable BlackJAX inference SEGMENT. Runs inference_runner.py with a
# wall-clock budget (--max_hours) so it checkpoints and exits cleanly before
# SLURM's --time limit kills it. Chain these with submit_inference_chain.sh; each
# segment resumes from <results_save_dir>/checkpoint/. A finished run makes later
# segments a fast no-op. Requires posterior_sampling.sampler == "blackjax".
#
# NOTE: confirm the partition/qos above actually permits a 24h --time on your
# account (submit_inference_chain.sh can override --time per job).

_die() { echo "Error: $*" >&2; exit 1; }

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/solver_params.json|yaml MAX_HOURS [EXTRA_DRAWS]" >&2
  echo "  MAX_HOURS   wall-clock budget before checkpoint+exit (set below SBATCH --time)" >&2
  echo "  EXTRA_DRAWS optional: raise the draw target (pass on the FIRST segment only)" >&2
  exit 1
fi

SOLVER_PARAMS_FILE="$1"
MAX_HOURS="$2"
EXTRA_DRAWS="${3:-}"
[[ -r "$SOLVER_PARAMS_FILE" ]] || _die "Solver params file not readable: $SOLVER_PARAMS_FILE"

EXTRA_ARGS=()
if [[ -n "$EXTRA_DRAWS" && "$EXTRA_DRAWS" != "0" ]]; then
  EXTRA_ARGS+=(--extra_draws "$EXTRA_DRAWS")
fi

PROJECT_DIR="/projects/anth4580/Bayesian"

source /etc/profile.d/lmod.sh
module load anaconda
conda activate Bayesian

echo "----------------------------------------------------------"
echo "==> Resumable BlackJAX inference segment"
echo "==> Solver params: $SOLVER_PARAMS_FILE"
echo "==> Max hours (this segment): $MAX_HOURS"
echo "==> Extra draws: ${EXTRA_DRAWS:-none}"
echo "==> SLURM job: ${SLURM_JOB_ID:-unset}  time limit: ${SBATCH_TIMELIMIT:-see --time}"
echo "==> SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

echo "==> Python executable: $(command -v python)"
echo "==> Host: $(hostname)"

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
PY

time python -u "$PROJECT_DIR/Utilities/inference_runner.py" \
  --solver_params_file "$SOLVER_PARAMS_FILE" \
  --max_hours "$MAX_HOURS" \
  "${EXTRA_ARGS[@]}"
