#!/bin/bash -l
#SBATCH --job-name=tier1
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/tier1/%x.%j.out
#SBATCH --export=NONE

# Runs one Tier-1 prep script on CPU. These are single, unbatched ODE solves, which a
# GPU does not speed up, so no GPU is requested.
# The script path is relative to job_files/ (multiparam_tests is job_files/bench on the cluster):
#   sbatch --job-name=t1data_C12 cpu_python.sbatch tier1/make_tier1_data.py C12
#   sbatch --job-name=info_C12   cpu_python.sbatch bench/info_vs_conditions.py --system C12

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -ge 1 ]] || _die "usage: sbatch cpu_python.sbatch <script relative to job_files/> [args]"
SCRIPT="/projects/anth4580/Bayesian/job_files/$1"; shift
[[ -r "$SCRIPT" ]] || _die "script not found: $SCRIPT"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=/projects/anth4580/Bayesian/.jax_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> $(basename "$SCRIPT") $* | node=$(hostname) | job=${SLURM_JOB_ID:-unset}"
cd "$(dirname "$SCRIPT")" || _die "cannot cd"
time /projects/anth4580/software/anaconda/envs/Bayesian/bin/python -u "$SCRIPT" "$@"
