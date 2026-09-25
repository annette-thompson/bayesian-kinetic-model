#!/bin/bash -l
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100|A100|h100
#SBATCH --time=02:00:00
#SBATCH --job-name=regen_data_final
#SBATCH --output=/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/logs/%x.%j.out
#SBATCH --export=NONE

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==> job=${SLURM_JOB_ID:-unset} host=$(hostname) $(date)"
time python -u "/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/regen_data_final.py"
