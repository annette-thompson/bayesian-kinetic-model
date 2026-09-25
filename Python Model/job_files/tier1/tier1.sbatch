#!/bin/bash -l
#SBATCH --job-name=tier1
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:15:00
# No GPU request here: submit through job_files/gpu_submit.sh, which queues this on Blanca
# and Alpine at once; the first twin to start runs it and cancels the other.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/tier1/%x.%j.out
#SBATCH --export=NONE

# One Tier-1 run (IDs R0-R8 in Notes/tier1_experiment_plan.md section 2).
#   ../gpu_submit.sh --time 12:15:00 --name tier1_C8_a1c3 -- \
#       tier1/tier1.sbatch "Tier1 C8 - a1c3"
#
# Unlike chaintest.sbatch, the config here has early stopping ON, so the run ends when it
# converges rather than when a budget expires; it then finalizes and writes its figures.
# If it hits the wall clock first it checkpoints, and resubmitting the same command resumes.
# --a100_only is NOT passed by default: these runs are not being compared to each other on
# timing, so any GPU is fine and a wider pool means a shorter queue.

_die() { echo "Error: $*" >&2; exit 1; }
RUN="${1:?run directory name, e.g. \"Tier1 C8 - a1c3\"}"
MAXH="${2:-11.5}"   # per-segment wall budget, under the 12:15 SLURM limit
BASE="/projects/anth4580/Bayesian"
SOLVER_PARAMS="$BASE/Results/Tier1/$RUN/solver_params.json"
[[ -r "$SOLVER_PARAMS" ]] || _die "config not found: $SOLVER_PARAMS (run build_tier1_configs.py)"

# First of the Blanca/Alpine twins to start runs; the other exits here.
source "$BASE/job_files/gpu_twin_claim.sh" || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export BAYESIAN_BATCH_STRATEGY=vmap
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export EQX_ON_ERROR=nan

echo "==> $RUN | node=$(hostname) | gpu=$GPU_MODEL | cluster=${SLURM_CLUSTER_NAME:-blanca}"
time python -u "$BASE/Utilities/inference_runner.py" \
    --solver_params_file "$SOLVER_PARAMS" --max_hours "$MAXH"

# Recovery is the point of Tier 1, so report it as soon as the run has a posterior. Harmless
# while still mid-run: the script skips runs that have not finalized yet.
python -u "$BASE/job_files/tier1/recovery_report.py" --only "$RUN" || true
