#!/bin/bash -l
#SBATCH --job-name=tier1_C8_a1c3_numpyro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/anth4580/Bayesian/job_files/tier1/%x.%j.out
#SBATCH --export=NONE
# Second-engine check (engine_check.py): R0's model refit with NumPyro's NUTS. NumPyro cannot
# resume, so this must run in one job that is not preempted: Alpine only, from job_files/:
#   ./gpu_submit.sh --alpine_only --time 24:00:00 --name tier1_C8_a1c3_numpyro -- tier1/engine_check.sbatch
BASE="/projects/anth4580/Bayesian"
source "$BASE/job_files/gpu_twin_claim.sh" || exit 0
[[ -n "$GPU_MODEL" ]] || { echo "no GPU visible -- submit through job_files/gpu_submit.sh" >&2; exit 1; }
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python   # the Bayesian env, without modules

# The same environment as tier1.sbatch, so the model is built and solved exactly as in R0.
export PYTHONUNBUFFERED=1
export BAYESIAN_BATCH_STRATEGY=vmap
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export EQX_ON_ERROR=nan

echo "==> engine check | node=$(hostname) | gpu=$GPU_MODEL | cluster=${SLURM_CLUSTER_NAME:-alpine}"
cd "$BASE/job_files/tier1" || exit 1
time "$PY" -u engine_check.py --run "Tier1 C8 - a1c3"
