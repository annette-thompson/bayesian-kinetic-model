#!/bin/bash -l
#SBATCH --job-name=tier1_parameter_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/anth4580/Bayesian/job_files/tier1/%x.%j.out
#SBATCH --export=NONE
# parameter_screen.py on the full-length system: all 18 scaling groups at the ME1 solution, the
# Tier-1 data design, 37 batched solves (the solution and each group nudged down and up). Not
# resumable, but short; from job_files/:
#   ./gpu_submit.sh --time 04:00:00 --name tier1_parameter_screen -- tier1/parameter_screen.sbatch [system]
SYSTEM="${1:-C20+unsat}"
BASE="/projects/anth4580/Bayesian"
source "$BASE/job_files/gpu_twin_claim.sh" || exit 0
[[ -n "$GPU_MODEL" ]] || { echo "no GPU visible -- submit through job_files/gpu_submit.sh" >&2; exit 1; }
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python   # the Bayesian env, without modules
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "==> parameter screen on $SYSTEM | node=$(hostname) | gpu=$GPU_MODEL | cluster=${SLURM_CLUSTER_NAME:-blanca}"
cd "$BASE/job_files/tier1" || exit 1
time "$PY" -u parameter_screen.py --system "$SYSTEM" 2>&1 | grep -v Warning
