#!/bin/bash
#SBATCH --job-name=bayes_seg_gpu
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --output=/projects/anth4580/Bayesian/job_files/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --account=ucb634_asc2

echo "==> Starting import/module test at $(date)"

echo "==> Trying anaconda"
module purge
module load anaconda
conda activate Bayesian
python "/projects/anth4580/Bayesian/Utilities/inference_runner.py" --solver_params_file "/projects/anth4580/Bayesian/Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json" --max_hours 0.01 --checkpoint_every_steps 5
echo "==> Done with anaconda (exit code of python test: $?)"

echo "==> Trying miniforge"
module purge
module load miniforge
mamba activate Bayesian
python "/projects/anth4580/Bayesian/Utilities/inference_runner.py" --solver_params_file "/projects/anth4580/Bayesian/Results/GPU Scaling Tests/Test FabD FabH FabG - a2/solver_params.json" --max_hours 0.01 --checkpoint_every_steps 5
echo "==> Done with miniforge (exit code of python test: $?)"

echo "==> Finished at $(date)"