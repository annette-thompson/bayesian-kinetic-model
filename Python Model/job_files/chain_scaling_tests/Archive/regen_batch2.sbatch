#!/bin/bash -l
#SBATCH --job-name=regenbatch2
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

echo "=== C16+unsat (tightest_nofloor): missing netcdf, finalize default window ==="
sp="$BASE/Chain C16+unsat - a1_0.1-10_no_floor/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
plot_only "$sp"

echo "=== C18+unsat (tightest_nofloor): missing netcdf, finalize default window ==="
sp="$BASE/Chain C18+unsat - a1 tightest nofloor-eqxnan/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
plot_only "$sp"

echo "=== C14 (narrowest_nofloor): missing netcdf, finalize default window ==="
sp="$BASE/Chain C14 - a1 narrowest nofloor-eqxnan/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
plot_only "$sp"

echo "=== C10 (a1c3_no_floor): missing netcdf, finalize default window ==="
sp="$BASE/Chain C10 - a1c3_no_floor/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp"
plot_only "$sp"

echo "=== C6 (a1c3_no_floor): netcdf already exists, plots only ==="
sp="$BASE/Chain C6 - a1c3_no_floor/solver_params.json"
plot_only "$sp"

echo "=== C6 (narrowest_nofloor): stranded chains [2,4] -- recomputed convergence at draw 300 excluding them; end_draw=1400 (burn_in 1000 + 300 + 100 buffer) ==="
sp="$BASE/Chain C6 - a1 narrowest nofloor-eqxnan/solver_params.json"
$PY finalize_window.py --solver_params_file "$sp" --end_draw 1400
plot_only "$sp"

echo "=== ALL DONE ==="
