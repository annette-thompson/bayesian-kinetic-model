#!/bin/bash
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --account=ucb634_asc2
#
# run_model_error.sh — Alpine SLURM worker for one Monte Carlo model-error run.
#
# Usage:
#   sbatch run_model_error.sh /path/to/model_error_config.json

_die() { echo "Error: $*" >&2; exit 1; }

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/model_error_config.json" >&2
  exit 1
fi

CONFIG_FILE="$1"
[[ -r "$CONFIG_FILE" ]] || _die "Config file not readable: $CONFIG_FILE"

PROJECT_DIR="/projects/anth4580/Bayesian"

source /etc/profile.d/lmod.sh
module load anaconda
conda activate Bayesian

echo "----------------------------------------------------------"
echo "==> Model error worker"
echo "==> Config: $CONFIG_FILE"
echo "==> SLURM_NTASKS: ${SLURM_NTASKS:-unset}"
echo "==> SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "----------------------------------------------------------"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# This partition is CPU-only, but the shared "Bayesian" env has jax[cuda12] installed (for
# the GPU inference scripts), so JAX's plugin auto-discovery probes for a CUDA device on
# every process start (main + each spawned worker) and fails noisily before falling back to
# CPU. Forcing the platform here skips that probe entirely.
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu

echo "==> Python executable: $(command -v python)"
echo "==> Host: $(hostname)"
echo "==> Thread env: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"

time python -u "$PROJECT_DIR/Utilities/model_error_runner.py" --config "$CONFIG_FILE"
