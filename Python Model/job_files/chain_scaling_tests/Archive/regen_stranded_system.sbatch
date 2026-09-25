#!/bin/bash -l
#SBATCH --job-name=regenstrand
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
# Bumped from 8h 2026-09-10: narrowest_nofloor/C10 (1205 sampling draws)
# finished in 4.5h, but tightest_nofloor/C8 (3000 sampling draws, 2.5x more)
# timed out at 8h still mid-computation. No internal checkpointing here, so a
# timeout restarts from zero -- sized generously to avoid a 3rd attempt.
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="A100"
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --export=NONE

# Re-finalizes ONE stranded-chain run (applying the 2026-09-09 exclusion fix)
# then generates its 3 standard plots. Split out from regen_finished_plots.sbatch
# 2026-09-09: that script's Step 1 (2 re-finalizations) alone didn't fit in a
# 3h budget, then a 12h budget, because narrowest_nofloor/C10's re-finalize
# (recomputing log-likelihood + a 10,000-draw prior-predictive + posterior-
# predictive, each a full ODE forward solve) is expensive enough on its own
# that bundling it with 16 other systems in one time budget meant one slow
# system could starve everything else. One system, one dedicated budget.
#   sbatch regen_stranded_system.sbatch narrowest_nofloor C10
#   sbatch regen_stranded_system.sbatch tightest_nofloor C8

_die() { echo "Error: $*" >&2; exit 1; }
SET_="${1:-}"
SYSTEM="${2:-}"
[[ -n "$SET_" && -n "$SYSTEM" ]] || _die "Usage: sbatch $0 <SET> <SYSTEM>  e.g. sbatch $0 narrowest_nofloor C10"

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

sp="$BASE/Chain $SYSTEM - $(resolve_suffix "$SET_" "$SYSTEM")/solver_params.json"
[[ -r "$sp" ]] || _die "Config not found: $sp"

echo "=== re-finalizing $SET_/$SYSTEM (applying stranded-chain exclusion) ==="
$PY finalize_window.py --solver_params_file "$sp"

echo
echo "=== generating all 3 plots for $SET_/$SYSTEM ==="
$PY plot_predictive_check.py --solver_params_file "$sp"
$PY plot_trace_diagnostics.py --solver_params_file "$sp"
$PY plot_convergence_trajectory.py --solver_params_file "$sp"
echo "=== DONE: $SET_/$SYSTEM ==="
