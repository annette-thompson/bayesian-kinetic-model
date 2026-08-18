#!/bin/bash -l
#SBATCH --job-name=probe_gpu
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=NONE
#SBATCH --export=NONE
#SBATCH --account=ucb634_asc2

# `-l` (login shell) and `--export=NONE` below are both load-bearing, and this
# script silently failed without them. sbatch defaults to --export=ALL, so the
# job inherited the submitting shell's MODULEPATH -- which on the login node does
# NOT contain /curc/sw/alpine-modules/* -- and that overrode what the compute node
# would have set for itself, making `module load miniforge` report "unknown
# module". --export=NONE lets the node build its own environment; -l makes it
# actually run the profile that initialises Lmod.

# GPU counterpart to run_throughput_probe.sh, which pins JAX_PLATFORMS=cpu and so
# can only ever measure the CPU path. This one runs the same probe on a full A100,
# which is what the inference jobs actually use -- and what any nate-vs-Alpine
# comparison has to be measured against.
#
# Usage: sbatch run_throughput_probe_gpu.sh /path/to/solver_params.json [MINUTES] [args]
#   --grad-only   time only logp+gradient evals (fast; the right mode for
#                 comparing simulation cost across configs or machines)

_die() { echo "Error: $*" >&2; exit 1; }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/solver_params.json|yaml [MINUTES] [extra args]" >&2
  exit 1
fi

SOLVER_PARAMS_FILE="$1"; shift
MINUTES=15
if [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then MINUTES="$1"; shift; fi
EXTRA_ARGS=("$@")
[[ -r "$SOLVER_PARAMS_FILE" ]] || _die "Solver params file not readable: $SOLVER_PARAMS_FILE"

PROJECT_DIR="/projects/anth4580/Bayesian"

# miniforge, not anaconda -- see the note in run_throughput_probe.sh.
module purge
module load miniforge
mamba activate Bayesian

echo "----------------------------------------------------------"
echo "==> Throughput probe (GPU)"
echo "==> Solver params: $SOLVER_PARAMS_FILE"
echo "==> SLURM job: ${SLURM_JOB_ID:-unset}  host: $(hostname)"
echo "==> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Versions matter for cross-machine comparison, so record them alongside the
# timing rather than leaving the reader to guess which stack produced it.
python - <<'PY'
import importlib.metadata as md
import jax
for p in ["jax", "jaxlib", "diffrax", "equinox", "pymc", "pytensor", "numpy", "blackjax"]:
    try:
        print(f"==> {p}: {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"==> {p}: not installed")
print(f"==> JAX devices: {jax.devices()}")
PY

time python -u "$PROJECT_DIR/Utilities/benchmark_throughput.py" \
  --solver_params_file "$SOLVER_PARAMS_FILE" \
  --minutes "$MINUTES" \
  "${EXTRA_ARGS[@]}"
