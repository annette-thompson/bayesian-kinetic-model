#!/bin/bash -l
#SBATCH --job-name=floorreal
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

# Production, unmodified (floor in place), real tune/draws, for a clean
# floor-vs-no-floor comparison at a specific prior width. Writes to
# "Chain X - a1 {width} matched" so it doesn't collide with any pre-existing
# differently-configured run at that width.
#   ../gpu_submit.sh --time 12:15:00 -- floor_real.sbatch C4_NoFB "narrowest matched"
#   ../gpu_submit.sh --time 12:15:00 -- floor_real.sbatch C6 "narrowest matched"

_die() { echo "Error: $*" >&2; exit 1; }
SYSTEM="${1:-}"
WIDTH="${2:-}"
MAXH="${3:-12.0}"
[[ -n "$SYSTEM" && -n "$WIDTH" ]] || _die "Usage: sbatch $0 SYSTEM WIDTH"
SOLVER_PARAMS="/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain $SYSTEM - a1 $WIDTH/solver_params.json"
[[ -r "$SOLVER_PARAMS" ]] || _die "Config not found: $SOLVER_PARAMS"
# First of the Blanca/Alpine twins to start runs; the other exits here.
source /projects/anth4580/Bayesian/job_files/gpu_twin_claim.sh || exit 0
[[ -n "$GPU_MODEL" ]] || _die "no GPU visible -- submit through job_files/gpu_submit.sh"
# Record this job in its set's jobids file. warmup_status.py discovers systems
# ONLY through that file, which used to be written solely by a successful
# resubmission -- and none ever succeeded, so a manually submitted run never
# showed up in it. Idempotent: resubmit_if_needed.sh may have logged this ID
# already at submit time, and a duplicate would double its counted compute.
JOBIDS_FILE="/projects/anth4580/Bayesian/job_files/inference_a1_${WIDTH// /_}_jobids.txt"
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

export MASK_TEST_MODE=floor
export MASK_TEST_SOLVER_PARAMS="$SOLVER_PARAMS"
export MASK_TEST_MAX_HOURS="${MAXH}"


# ---- self-resubmission -------------------------------------------------- #
# The successor is submitted from here rather than pre-chained with
# --dependency=afterany, so it exists only when the work is actually
# unfinished: resubmit_if_needed.sh checks the checkpoint phase and the
# per-chain acceptance first. The TERM trap covers preemption -- on this QOS a
# job can be requeued out from under us, and SIGTERM is the only warning.
RESUB="/projects/anth4580/Bayesian/job_files/chain_scaling_tests/resubmit_if_needed.sh"
# Not "$(basename "${BASH_SOURCE[0]}")": under SLURM the batch script runs
# from a spool copy named "slurm_script", so resubmit_if_needed.sh built
# ".../masking_check/slurm_script" and EVERY automatic resubmission failed
# with "Unable to open file" -- a run that hit its segment budget just
# stopped, silently. Found 2026-09-14 after six runs sat idle ~2.5 days.
_SCRIPT_NAME="floor_real.sbatch"
_resubmitted=0
_do_resubmit() {
    [[ $_resubmitted -eq 1 ]] && return 0
    _resubmitted=1
    bash "$RESUB" "$_SCRIPT_NAME" "$SYSTEM" "$WIDTH" "$MAXH" "${1:-normal}"
}
trap '_do_resubmit preempted; exit 143' TERM

echo "==> System: $SYSTEM  Width: $WIDTH  Mode: floor (production, unmodified)"
echo "==> Host: $(hostname)"
echo "==> GPU model: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

time python -u /projects/anth4580/Bayesian/job_files/chain_scaling_tests/truncated_warmup_test.py

_do_resubmit normal
