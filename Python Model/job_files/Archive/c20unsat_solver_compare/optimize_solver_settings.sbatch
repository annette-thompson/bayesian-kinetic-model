#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.%j.out
#SBATCH --export=NONE

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

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -eq 1 ]] || _die "Usage: sbatch optimize_solver_settings.sbatch <PID_INDEX:0-4>"
PID_INDEX="$1"

echo "==> pid_index=$PID_INDEX  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
cd "$PROJECT_DIR"
time python -u "$JOBDIR/optimize_solver_settings.py" --pid_index "$PID_INDEX"
