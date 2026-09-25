#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.%j.out
#SBATCH --export=NONE

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || _die "Usage: sbatch generate_and_score_ladder.sbatch <SYSTEM> <CANDIDATE:D|E>"
SYSTEM="$1"
CANDIDATE="$2"

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

echo "==> system=$SYSTEM candidate=$CANDIDATE  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
time python -u "$JOBDIR/generate_and_score_ladder.py" --system "$SYSTEM" --candidate "$CANDIDATE"
