#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=NONE
#SBATCH --account=ucb634_asc2

# Throughput probe for ONE solver config: measures BlackJAX warmup/sampling
# draws-per-hour and the sec-per-gradient / leapfrog-per-draw decomposition, then
# writes throughput_probe.json next to the config's results. Independent per
# config, so submit one of these per Test* config in parallel to map scaling.
# Does NOT run a full inference and does NOT write posterior netcdf.

_die() { echo "Error: $*" >&2; exit 1; }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/solver_params.json|yaml [MINUTES] [extra benchmark_throughput.py args]" >&2
  echo "  MINUTES     full-NUTS probe wall budget (default 15). Use more for large networks." >&2
  echo "  --grad-only fast mode: time only logp+gradient evals (recommended for large/stiff" >&2
  echo "              configs where the full probe is too slow). MINUTES is then ignored." >&2
  echo "  e.g.  $0 'Results/Test FabD FabH FabG - a1/solver_params.json' --grad-only" >&2
  exit 1
fi

SOLVER_PARAMS_FILE="$1"; shift
MINUTES=15
if [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then MINUTES="$1"; shift; fi
EXTRA_ARGS=("$@")
[[ -r "$SOLVER_PARAMS_FILE" ]] || _die "Solver params file not readable: $SOLVER_PARAMS_FILE"

PROJECT_DIR="/projects/anth4580/Bayesian"

source /etc/profile.d/lmod.sh
module load anaconda
conda activate Bayesian

echo "----------------------------------------------------------"
echo "==> Throughput probe"
echo "==> Solver params: $SOLVER_PARAMS_FILE"
echo "==> Minutes: $MINUTES"
echo "==> SLURM job: ${SLURM_JOB_ID:-unset}  CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

# Pre-stamp arviz's "warn once per day" cache file before it gets imported.
python - <<'PY'
import datetime
from pathlib import Path
from platformdirs import user_cache_dir

cache_dir = Path(user_cache_dir("arviz", "arviz"))
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "daily_warning").write_text(datetime.date.today().isoformat())
PY

time python -u "$PROJECT_DIR/Utilities/benchmark_throughput.py" \
  --solver_params_file "$SOLVER_PARAMS_FILE" \
  --minutes "$MINUTES" \
  "${EXTRA_ARGS[@]}"
