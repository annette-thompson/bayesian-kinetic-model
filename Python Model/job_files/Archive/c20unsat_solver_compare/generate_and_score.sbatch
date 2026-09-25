#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.%j.out
#SBATCH --export=NONE

# One job per candidate -- regenerates that candidate's 9-row sweep with the
# strict-tolerance filter (rejects rows that don't converge within 10,000
# steps at the exact reference settings), then scores all 10 resulting
# conditions. No GPU needed (plain diffrax ODE solve).

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -eq 1 ]] || _die "Usage: sbatch generate_and_score.sbatch <CANDIDATE:A-E>"
CANDIDATE="$1"

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

echo "==> candidate=$CANDIDATE  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
time python -u "$JOBDIR/generate_and_score.py" --candidate "$CANDIDATE"
