#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100|A100|P100|h100
#SBATCH --time=02:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.out
#SBATCH --open-mode=append
#SBATCH --export=NONE

# NOTE: --job-name must be passed at submission time as ladder_<CANDIDATE> (same
# name for every system in that candidate's chain) so %x above resolves to ONE
# shared, append-mode log file per candidate across all 13 sequential systems.

_die() { echo "Error: $*" >&2; exit 1; }
[[ $# -eq 2 ]] || _die "Usage: sbatch --job-name=ladder_<CANDIDATE> generate_and_score_ladder2.sbatch <SYSTEM> <CANDIDATE>"
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

echo ""
echo "==================== SYSTEM=$SYSTEM CANDIDATE=$CANDIDATE  job=${SLURM_JOB_ID:-unset}  host=$(hostname)  $(date) ===================="
time python -u "$JOBDIR/generate_and_score_ladder2.py" --system "$SYSTEM" --candidate "$CANDIDATE" --min_total_kept 10
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "==================== FAILED at SYSTEM=$SYSTEM CANDIDATE=$CANDIDATE (exit $rc) -- chain stops here ===================="
fi
exit $rc
