#!/bin/bash -l
#SBATCH --job-name=mparam
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:15:00
# No GPU request here: submit through job_files/gpu_submit.sh, which queues this
# script on Blanca (any full A100/V100/H100) and Alpine (full A100) at once; the
# first twin to start runs it and cancels the other (claim below). A bare sbatch
# gets no GPU and stops at the GPU check.
#SBATCH --output=/projects/anth4580/Bayesian/job_files/chain_scaling_tests/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

# Multi-parameter pilot: no-floor + EQX_ON_ERROR=nan only (floor deliberately
# not tested here -- see Notes/efficiency_logic_map.md and the plan this was
# built from). Config comes from build_multiparam_config.py, at
# "Chain $SYSTEM - ${PARAMSET}_no_floor/solver_params.json" -- tune=600,
# ess_threshold=null (convergence simplified to r-hat alone), rank-ECDF
# removed, everything else matching that system's existing single-parameter
# no-floor config.
#   ../gpu_submit.sh --time 12:15:00 -- multiparam_nofloor.sbatch C6 a1c2
#   ../gpu_submit.sh --time 12:15:00 -- multiparam_nofloor.sbatch C10 a1c3
# 12h segment budget (15 min margin under the 12:15 SLURM limit); checkpoints
# and exits cleanly if not done, self-resubmits via resubmit_if_needed.sh.

_die() { echo "Error: $*" >&2; exit 1; }
SYSTEM="${1:-}"
PARAMSET="${2:-}"
MAXH="${3:-12.0}"
[[ -n "$SYSTEM" && -n "$PARAMSET" ]] || _die "Usage: sbatch $0 <SYSTEM> <PARAMSET> [max_hours]  e.g. sbatch $0 C6 a1c2"
# resubmit_if_needed.sh passes the FULL label ("a1c3a2_no_floor") back as this
# argument, while a manual submit passes the bare one ("a1c3a2"). Appending
# unconditionally turned every resubmission into "..._no_floor_no_floor" and a
# config-not-found exit; strip first so both entry points land on the same dir.
PARAMSET="${PARAMSET%_no_floor}"
WIDTH="${PARAMSET}_no_floor"
SOLVER_PARAMS="/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain $SYSTEM - $WIDTH/solver_params.json"
[[ -r "$SOLVER_PARAMS" ]] || _die "Config not found: $SOLVER_PARAMS (run build_multiparam_config.py first)"
# First of the Blanca/Alpine twins to start runs; the other exits here.
source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"
# Record this job in its set's jobids file. warmup_status.py discovers systems
# ONLY through that file, which used to be written solely by a successful
# resubmission -- and none ever succeeded, so a manually submitted run never
# showed up in it. Idempotent: resubmit_if_needed.sh may have logged this ID
# already at submit time, and a duplicate would double its counted compute.
JOBIDS_FILE="/projects/anth4580/Bayesian/job_files/inference_${WIDTH}_jobids.txt"
# Alpine job IDs are written "alpine:<id>" so warmup_status.py asks the right cluster.
_JID="${SLURM_JOB_ID:-}"
[[ -n "$_JID" && "${SLURM_CLUSTER_NAME:-blanca}" != blanca ]] && _JID="${SLURM_CLUSTER_NAME}:$_JID"
if [[ -n "$_JID" ]] && ! grep -qx "$SYSTEM $_JID" "$JOBIDS_FILE" 2>/dev/null; then
    echo "$SYSTEM $_JID" >> "$JOBIDS_FILE"
fi

module purge
module load miniforge
mamba activate Bayesian

export PYTHONUNBUFFERED=1
export BAYESIAN_BATCH_STRATEGY=vmap
export JAX_COMPILATION_CACHE_DIR="/projects/anth4580/Bayesian/.jax_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export MASK_TEST_MODE=nofloor
export MASK_TEST_SOLVER_PARAMS="$SOLVER_PARAMS"
export MASK_TEST_MAX_HOURS="${MAXH}"
export EQX_ON_ERROR=nan

# ---- self-resubmission -------------------------------------------------- #
RESUB="/projects/anth4580/Bayesian/job_files/chain_scaling_tests/resubmit_if_needed.sh"
# Not "$(basename "${BASH_SOURCE[0]}")": under SLURM the batch script runs
# from a spool copy named "slurm_script", so resubmit_if_needed.sh built
# ".../masking_check/slurm_script" and EVERY automatic resubmission failed
# with "Unable to open file" -- a run that hit its segment budget just
# stopped, silently. Found 2026-09-14 after six runs sat idle ~2.5 days.
_SCRIPT_NAME="multiparam_nofloor.sbatch"
_resubmitted=0
_do_resubmit() {
    [[ $_resubmitted -eq 1 ]] && return 0
    _resubmitted=1
    bash "$RESUB" "$_SCRIPT_NAME" "$SYSTEM" "$WIDTH" "$MAXH" "${1:-normal}"
}
trap '_do_resubmit preempted; exit 143' TERM

echo "==> System: $SYSTEM  Paramset: $PARAMSET  (no-floor only)"
echo "==> Host: $(hostname)"
echo "==> GPU model: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

time python -u /projects/anth4580/Bayesian/job_files/chain_scaling_tests/truncated_warmup_test.py

_do_resubmit normal
