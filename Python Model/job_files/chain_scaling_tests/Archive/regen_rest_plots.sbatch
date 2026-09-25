#!/bin/bash -l
#SBATCH --job-name=regenrest
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

# Generates the 3 standard plots for every converged run EXCEPT the 2
# stranded-chain systems (those go through regen_stranded_system.sbatch
# instead, since they need an expensive re-finalize first). These 15 already
# have posterior_predictive/log_likelihood cached in their existing netcdf
# from when they first converged, so this is just reading + plotting -- no
# recomputation -- hence the much smaller time budget than the stranded jobs.
#   sbatch regen_rest_plots.sbatch

module purge
module load miniforge
mamba activate Bayesian

PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
BASE="/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
UTIL=/projects/anth4580/Bayesian/Utilities
cd "$UTIL"

declare -A SUFFIX
SUFFIX[tightest]="a1 tightest"
SUFFIX[tightest_nofloor]="a1 tightest nofloor-eqxnan"
SUFFIX[narrowest_matched]="a1 narrowest matched"
SUFFIX[narrowest_nofloor]="a1 narrowest nofloor-eqxnan"

declare -A NEW_SUFFIX
NEW_SUFFIX[tightest]="a1_0.1-10_floor"
NEW_SUFFIX[tightest_nofloor]="a1_0.1-10_no_floor"
NEW_SUFFIX[narrowest_matched]="a1_0.05-20_floor"
NEW_SUFFIX[narrowest_nofloor]="a1_0.05-20_no_floor"

resolve_suffix() {
  local set_="$1" sys="$2" new="${NEW_SUFFIX[$1]}"
  if [[ -n "$new" && -d "$BASE/Chain $sys - $new" ]]; then
    echo "$new"
  else
    echo "${SUFFIX[$set_]}"
  fi
}

# Same as regen_finished_plots.sbatch's old RUNS list, minus
# "narrowest_nofloor C10" and "tightest_nofloor C8" (the 2 stranded-chain
# systems, handled by regen_stranded_system.sbatch instead), plus
# tightest_nofloor's C20 (converged since that list was last written).
RUNS=(
  "tightest C4_NoFB" "tightest C6" "tightest C8"
  "tightest_nofloor C4_NoFB" "tightest_nofloor C6"
  "tightest_nofloor C10" "tightest_nofloor C12" "tightest_nofloor C12+unsat"
  "tightest_nofloor C14" "tightest_nofloor C14+unsat" "tightest_nofloor C16"
  "tightest_nofloor C16+unsat" "tightest_nofloor C18" "tightest_nofloor C20"
  "narrowest_matched C4_NoFB"
  "narrowest_nofloor C4_NoFB"
)

for pair in "${RUNS[@]}"; do
  read -r set_ system <<< "$pair"
  sp="$BASE/Chain $system - $(resolve_suffix "$set_" "$system")/solver_params.json"
  echo "--- $set_/$system ---"
  $PY plot_predictive_check.py --solver_params_file "$sp"
  $PY plot_trace_diagnostics.py --solver_params_file "$sp"
  $PY plot_convergence_trajectory.py --solver_params_file "$sp"
done
echo "=== ALL DONE ==="
