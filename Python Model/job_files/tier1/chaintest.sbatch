#!/bin/bash -l
#SBATCH --job-name=chaintest
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:15:00
# No GPU request here: submit through job_files/gpu_submit.sh --a100_only, which queues
# this on Blanca and Alpine A100s at once (A100 only, so per-draw times are comparable
# across chain counts); the first twin to start runs it and cancels the other.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/tier1/%x.%j.out
#SBATCH --export=NONE

# One chain-count run: Notes/tier1_experiment_plan.md section 1.
#   ../gpu_submit.sh --a100_only --time 12:15:00 --name chaintest_16 -- \
#       tier1/chaintest.sbatch "Chain C12 - a1c3_16chains"
# The run stops when its sampling-time budget (max_sampling_hours in the config) is spent,
# leaving a checkpoint; it is deliberately not finalized, since the analysis reads the
# draws directly.

_die() { echo "Error: $*" >&2; exit 1; }
RUN="${1:?run directory name, e.g. \"Chain C12 - a1c3_16chains\"}"
MAXH="${2:-11.5}"   # per-segment wall budget, under the 12:15 SLURM limit
SOLVER_PARAMS="/projects/anth4580/Bayesian/Results/Chain Count Test/$RUN/solver_params.json"
[[ -r "$SOLVER_PARAMS" ]] || _die "config not found: $SOLVER_PARAMS (run build_chaintest_configs.py)"

# First of the Blanca/Alpine twins to start runs; the other exits here.
source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export BAYESIAN_BATCH_STRATEGY=vmap
export JAX_COMPILATION_CACHE_DIR="/projects/anth4580/Bayesian/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export EQX_ON_ERROR=nan

echo "==> $RUN | node=$(hostname) | gpu=$GPU_MODEL | cluster=${SLURM_CLUSTER_NAME:-blanca}"
time python -u /projects/anth4580/Bayesian/Utilities/inference_runner.py \
    --solver_params_file "$SOLVER_PARAMS" --max_hours "$MAXH"
