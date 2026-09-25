#!/bin/bash -l
#SBATCH --job-name=dbg_d1d2
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/%x.%j.out
#SBATCH --export=NONE
module purge; module load miniforge; mamba activate Bayesian
export PYTHONUNBUFFERED=1
python -u /projects/anth4580/Bayesian/job_files/chain_system_sensitivity_analysis/debug_d1d2.py
