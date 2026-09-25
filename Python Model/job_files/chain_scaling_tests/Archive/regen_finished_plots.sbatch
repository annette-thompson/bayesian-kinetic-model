#!/bin/bash -l
#SBATCH --job-name=regenplots
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
# Bumped from 3h 2026-09-09: the 3h budget wasn't even enough for the FIRST of
# Step 1's 2 re-finalizations (narrowest_nofloor/C10 alone hit the time limit
# still mid-posterior-predictive, ~4820 posterior draws + a 10,000-draw prior
# predictive pass, each a full ODE forward solve). This script has no
# checkpointing of its own -- a time-limit kill restarts everything from
# scratch, not from where it left off -- so 12h (matching this project's
# other segment caps) rather than a smaller bump, to avoid a 4th failure.
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="A100"
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --export=NONE

# Re-finalizes the 2 converged runs that have a stranded chain (applying the
# 2026-09-09 exclusion fix to their saved netcdfs), then generates all 3
# standard plots (predictive check, trace diagnostics, convergence trajectory)
# for every currently-converged run, saved into each run own results folder.
#   sbatch regen_finished_plots.sbatch

module purge
module load miniforge
mamba activate Bayesian

PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
BASE="/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
UTIL=/projects/anth4580/Bayesian/Utilities
cd "$UTIL"

# Old, pre-2026-09-09-rename suffix, kept only as the fallback for a system
# that (for whatever reason) hasn't been renamed yet -- see resolve_suffix().
declare -A SUFFIX
SUFFIX[tightest]="a1 tightest"
SUFFIX[tightest_nofloor]="a1 tightest nofloor-eqxnan"
SUFFIX[narrowest_matched]="a1 narrowest matched"
SUFFIX[narrowest_nofloor]="a1 narrowest nofloor-eqxnan"

# New, prior-value-based suffix (warmup_status.py's NEW_SUFFIX). Every run in
# RUNS below is finished, so this should always be the one actually used --
# the SUFFIX fallback above only matters if renaming is somehow incomplete.
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

RUNS=(
  "tightest C4_NoFB" "tightest C6" "tightest C8"
  "tightest_nofloor C4_NoFB" "tightest_nofloor C6" "tightest_nofloor C8"
  "tightest_nofloor C10" "tightest_nofloor C12" "tightest_nofloor C12+unsat"
  "tightest_nofloor C14" "tightest_nofloor C14+unsat" "tightest_nofloor C16"
  "tightest_nofloor C16+unsat" "tightest_nofloor C18"
  "narrowest_matched C4_NoFB"
  "narrowest_nofloor C4_NoFB" "narrowest_nofloor C10"
)

echo "=== Step 1: re-finalize the 2 runs with stranded chains, applying the exclusion fix ==="
for pair in "narrowest_nofloor C10" "tightest_nofloor C8"; do
  read -r set_ system <<< "$pair"
  sp="$BASE/Chain $system - $(resolve_suffix "$set_" "$system")/solver_params.json"
  echo "--- re-finalizing $set_/$system ---"
  $PY finalize_window.py --solver_params_file "$sp"
done

echo
echo "=== Step 2: generate all 3 plot types for every converged run ==="
for pair in "${RUNS[@]}"; do
  read -r set_ system <<< "$pair"
  sp="$BASE/Chain $system - $(resolve_suffix "$set_" "$system")/solver_params.json"
  echo "--- $set_/$system ---"
  $PY plot_predictive_check.py --solver_params_file "$sp"
  $PY plot_trace_diagnostics.py --solver_params_file "$sp"
  $PY plot_convergence_trajectory.py --solver_params_file "$sp"
done
echo "=== ALL DONE ==="
