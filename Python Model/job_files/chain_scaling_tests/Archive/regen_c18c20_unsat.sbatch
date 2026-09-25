#!/bin/bash -l
#SBATCH --job-name=regenc1820
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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

plot_only() {
  local sp="$1"
  $PY plot_predictive_check.py --solver_params_file "$sp"
  $PY plot_trace_diagnostics.py --solver_params_file "$sp"
  $PY plot_convergence_trajectory.py --solver_params_file "$sp"
}

echo "=== C18+unsat (tightest_nofloor): missing netcdf entirely, full finalize ==="
sp="$BASE/Chain C18+unsat - a1 tightest nofloor-eqxnan/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
plot_only "$sp"

echo "=== C20+unsat (tightest_nofloor): netcdf exists, plots missing ==="
sp="$BASE/Chain C20+unsat - a1 tightest nofloor-eqxnan/solver_params.json"
plot_only "$sp"

echo "=== ALL DONE ==="
