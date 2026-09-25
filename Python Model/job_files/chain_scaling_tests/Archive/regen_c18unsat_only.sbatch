#!/bin/bash -l
#SBATCH --job-name=regenc18u
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="A100"
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --export=NONE

module purge
module load miniforge
mamba activate Bayesian
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
BASE="/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
cd /projects/anth4580/Bayesian/Utilities

sp="$BASE/Chain C18+unsat - a1 tightest nofloor-eqxnan/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
$PY plot_predictive_check.py --solver_params_file "$sp"
$PY plot_trace_diagnostics.py --solver_params_file "$sp"
$PY plot_convergence_trajectory.py --solver_params_file "$sp"
echo "=== C18+unsat full finalize + plots done ==="
