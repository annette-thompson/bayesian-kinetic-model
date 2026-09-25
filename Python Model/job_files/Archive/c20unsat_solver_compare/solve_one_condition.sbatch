#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.%j.out
#SBATCH --export=NONE

# CPU-only (no --gres=gpu) -- this is a plain diffrax ODE solve, no PyMC/BlackJAX
# sampling, so a GPU would be wasted allocation. One job = one (candidate,
# sweep-condition) pair: solve at that candidate's own PID/tolerance, then at
# the fixed tight-tolerance reference, write one JSON row to results/.

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || _die "Usage: sbatch solve_one_condition.sbatch <CANDIDATE:A-E> <INDEX:1-9>"
CANDIDATE="$1"
INDEX="$2"

PROJECT_DIR="/projects/anth4580/Bayesian"
JOBDIR="$PROJECT_DIR/job_files/c20unsat_solver_compare"

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> candidate=$CANDIDATE index=$INDEX  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
time python -u "$JOBDIR/solve_one_condition.py" --candidate "$CANDIDATE" --index "$INDEX"
