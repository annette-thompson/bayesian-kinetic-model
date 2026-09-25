#!/bin/bash -l
#SBATCH --job-name=finplot
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/projects/anth4580/Bayesian/job_files/masking_check/%x.%j.out
#SBATCH --export=NONE

# Finalize a converged run (netcdf) and/or draw its six standard figures.
#
# No GPU, time, CPU, or memory request is fixed here -- the caller passes them,
# because the two stages want different hardware. `finalize` re-solves the ODE
# once per posterior draw for the log-likelihood and again for the predictive,
# so it wants a GPU. `plots` only reads the netcdf and needs none; holding a GPU
# idle while matplotlib runs just blocks someone else's job.
#
#   finalize then figures (GPU, Blanca + Alpine twins; figures follow on Blanca CPU):
#     ../gpu_submit.sh --a100_hours 8 --mem 16G -- \
#         finalize_and_plot.sbatch "Chain C10 - a1c3_no_floor" finalize+plots
#   figures only, CPU:
#     sbatch --time=01:00:00 --cpus-per-task=4 --mem=8G \
#         finalize_and_plot.sbatch "Chain C10 - a1c3_no_floor" plots
#
# finalize+plots submits the figures job itself once the netcdf exists, instead of a
# --dependency=afterok chain: a dependency cannot follow a job to the other cluster,
# and it would wait forever on whichever twin was cancelled.
#
# finalize re-applies the stranded-chain exclusion from the run's status.json,
# so it is also the way to fix a netcdf written before that exclusion existed.
# Anything after the mode is passed straight to finalize_window.py -- e.g. to end
# the posterior at a run's real convergence point when the live detector let it
# keep sampling past it (full-timeline index: n_tune + sampling draws to keep):
#     ... finalize_and_plot.sbatch "Chain C10 - a1_0.05-20_no_floor" finalize --end_draw 1500

set -uo pipefail
_die() { echo "Error: $*" >&2; exit 1; }

RUN="${1:?run directory name, e.g. \"Chain C10 - a1c3_no_floor\"}"
MODE="${2:?mode: finalize, finalize+plots or plots}"
BASE="/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
UTIL=/projects/anth4580/Bayesian/Utilities
PY=/projects/anth4580/software/anaconda/envs/Bayesian/bin/python
SP="$BASE/$RUN/solver_params.json"
NC="$BASE/$RUN/posterior_samples_pm.nc"
[[ -r "$SP" ]] || _die "config not found: $SP"
# First of the Blanca/Alpine twins to start runs; the other exits here (no-op for a
# plain sbatch of the plots stage).
source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0

module purge
module load miniforge
mamba activate Bayesian
cd "$UTIL" || _die "cannot cd to $UTIL"

echo "==> $RUN | mode=$MODE | node=$(hostname) | gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

case "$MODE" in
  finalize|finalize+plots)
    [[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit finalize through job_files/gpu_submit.sh"
    "$PY" -u finalize_window.py --solver_params_file "$SP" "${@:3}" || _die "finalize_window.py failed"
    [[ -f "$NC" ]] || _die "finalize exited cleanly but wrote no netcdf"
    if [[ "$MODE" == finalize+plots ]]; then
      jid=$(module unload slurm >/dev/null 2>&1; module load slurm/blanca >/dev/null 2>&1
            sbatch --parsable --partition=blanca --qos=preemptable --time=01:00:00 \
                   --cpus-per-task=4 --mem=8G /projects/anth4580/Bayesian/job_files/masking_check/finalize_and_plot.sbatch "$RUN" plots 2>&1)
      echo "==> figures job (Blanca CPU): $jid"
    fi
    ;;
  plots)
    [[ -f "$NC" ]] || _die "no netcdf to plot -- run the finalize stage first"
    export JAX_PLATFORMS=cpu
    failed=0
    for script in plot_predictive_check.py plot_trace_diagnostics.py plot_convergence_trajectory.py; do
      "$PY" -u "$script" --solver_params_file "$SP" || { echo "!! $script failed"; failed=1; }
    done
    exit "$failed"
    ;;
  *)
    _die "mode must be 'finalize', 'finalize+plots' or 'plots', got '$MODE'"
    ;;
esac
