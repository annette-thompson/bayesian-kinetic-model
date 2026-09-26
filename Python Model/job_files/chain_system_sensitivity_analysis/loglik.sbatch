#!/bin/bash -l
#SBATCH --job-name=loglik
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100|h100"
#SBATCH --time=06:00:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/%x_%j.out
#SBATCH --export=NONE
module purge; module load miniforge; mamba activate Bayesian
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=/projects/anth4580/Bayesian/.jax_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "==> host $(hostname)  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
time python -u /projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/loglik_range_sweep.py "$@"
