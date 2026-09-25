#!/bin/bash -l
#SBATCH --job-name=scaletest
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/anth4580/Bayesian/tmp_tests/%x.%j.out
#SBATCH --export=NONE
module purge
module load miniforge
mamba activate Bayesian
export JAX_PLATFORMS=cpu
cd /projects/anth4580/Bayesian/Utilities
/projects/anth4580/software/anaconda/envs/Bayesian/bin/python -u /projects/anth4580/Bayesian/tmp_tests/test_scale_transform.py
