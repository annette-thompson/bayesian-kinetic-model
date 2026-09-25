#!/bin/bash -l
#SBATCH --job-name=rtol_ladder
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100|h100"
#SBATCH --time=04:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/rtol_ladder/%x.%j.out
#SBATCH --export=NONE

# Measures rtol in {1e-3, 1e-4, 1e-5} for each system named on the command line and
# promotes the cheapest tolerance that still yields 10 endpoint conditions. Systems
# are passed as arguments so the 14-system set can be split across several jobs.
#
# Constraint matches the inference launcher: fp64 throughput matters here as much as
# it does there, since these are the same float64 Kvaerno5 solves.

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -ge 1 ]] || _die "Usage: sbatch rtol_ladder.sbatch <SYSTEM> [SYSTEM ...]"

PROJECT_DIR="/projects/anth4580/Bayesian"
JOBDIR="$PROJECT_DIR/job_files/rtol_ladder"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$PROJECT_DIR/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> systems: $*"
echo "==> host: $(hostname)  job: ${SLURM_JOB_ID:-unset}"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

time python -u "$JOBDIR/rtol_ladder.py" "$@"
