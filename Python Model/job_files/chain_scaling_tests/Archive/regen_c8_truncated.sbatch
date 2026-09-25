#!/bin/bash -l
#SBATCH --job-name=regenc8trunc
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="A100"
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --export=NONE

# One-off: C8's live run kept sampling to 3000 draws because its stranded
# chain (chain 2) wasn't caught by the live detector until late; recomputing
# r-hat/ESS over the other 7 chains from the start shows real convergence at
# draw 300 (2 consecutive passes). --end_draw 1400 = burn_in(1000) + 300 + a
# 100-draw buffer, matching the post-convergence margin every other run gets.
module purge
module load miniforge
mamba activate Bayesian
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
cd /projects/anth4580/Bayesian/Utilities
SP="/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain C8 - a1_0.1-10_no_floor/solver_params.json"
$PY finalize_window.py --solver_params_file "$SP" --end_draw 1400
$PY plot_predictive_check.py --solver_params_file "$SP"
$PY plot_trace_diagnostics.py --solver_params_file "$SP"
$PY plot_convergence_trajectory.py --solver_params_file "$SP"
echo "=== DONE ==="
