#!/bin/bash
#SBATCH --job-name=import_test
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=A
#SBATCH --account=ucb634_asc2

echo "==> Starting import/module test at $(date)"

echo "==> Trying anaconda"
module purge
module load anaconda
conda activate Bayesian
python -c "import sys, numpy; print('Python:', sys.executable); print('NumPy version:', numpy.__version__)"
echo "==> Done with anaconda (exit code of python test: $?)"

echo "==> Trying miniforge"
module purge
module load miniforge
mamba activate Bayesian
python -c "import sys, numpy; print('Python:', sys.executable); print('NumPy version:', numpy.__version__)"
echo "==> Done with miniforge (exit code of python test: $?)"

echo "==> Finished at $(date)"